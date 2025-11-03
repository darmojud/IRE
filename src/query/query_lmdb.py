import lmdb
import json
import math
import re
from natsort import natsorted
from query.Parser import Parser, TermNode, NotNode, AndNode, OrNode


class QueryOnLMDB:
    def __init__(self, info="BOOLEAN", db_path="./index_lmdb"):
        self.loaded_index = None
        self.doc_titles = {}
        self.verbose = True
        self.info = info
        self.db_path = db_path

        # Open LMDB environment (read-only mode for safety)
        self.env = lmdb.open(
            db_path,
            readonly=True,
            lock=False,
            readahead=True,
            max_dbs=1,
        )

        self.index_pattern = "index:term"
        self.meta_pattern = "meta"

    def _get_from_db(self, key: str):
        """Helper to get JSON-decoded value from LMDB."""
        with self.env.begin() as txn:
            val = txn.get(key.encode("utf-8"))
            if not val:
                return None
            try:
                return json.loads(val.decode("utf-8"))
            except json.JSONDecodeError:
                print(f"[WARN] Corrupt JSON for key {key}")
                return None

    def get_postings_from_db(self, term: str) -> dict:
        if hasattr(self, "_cache") and term in self._cache:
            return self._cache[term]

        key = f"{self.index_pattern}:{term.lower()}"
        postings = self._get_from_db(key) or {}

        if not hasattr(self, "_cache"):
            self._cache = {}
        self._cache[term] = postings
        return postings

    @staticmethod
    def tokenize_query(query):
        pattern = r"\"[^\"]+\"|\(|\)|AND|OR|NOT|[a-zA-Z0-9]+"
        tokens = re.findall(pattern, query, flags=re.IGNORECASE)
        return [
            t.upper() if t.upper() in {"AND", "OR", "NOT"} else t.strip('"')
            for t in tokens
        ]

    @staticmethod
    def combine_adjacent_terms(tokens):
        combined = []
        i = 0
        while i < len(tokens):
            if tokens[i] not in {"AND", "OR", "NOT", "(", ")"}:
                phrase = [tokens[i]]
                while i + 1 < len(tokens) and tokens[i + 1] not in {
                    "AND",
                    "OR",
                    "NOT",
                    "(",
                    ")",
                }:
                    phrase.append(tokens[i + 1])
                    i += 1
                combined.append(" ".join(phrase))
            else:
                combined.append(tokens[i])
            i += 1
        return combined

    def get_terms(self, node):
        if isinstance(node, TermNode):
            return [node.term]
        elif isinstance(node, NotNode):
            return [f"NOT({self.get_terms(node.child)})"]
        elif isinstance(node, AndNode) or isinstance(node, OrNode):
            return self.get_terms(node.left) + self.get_terms(node.right)
        return []

    def phrase_query(self, phrase):
        if hasattr(self, "_cache"):
            cached_terms = [t for t in phrase if t in self._cache]
        else:
            self._cache = {}
            cached_terms = []

        terms_to_fetch = [t for t in phrase if t not in cached_terms]

        for term in terms_to_fetch:
            postings = self._get_from_db(f"{self.index_pattern}:{term.lower()}") or {}
            self._cache[term] = postings

        postings_dicts = [self._cache[t] for t in phrase]
        list_of_docs = set(postings_dicts[0].keys())
        for postings in postings_dicts[1:]:
            list_of_docs &= set(postings.keys())

        if not list_of_docs:
            return []

        if len(phrase) == 1:
            return natsorted(list(list_of_docs))

        term_positions_per_doc = {doc_id: [] for doc_id in list_of_docs}

        for i, term in enumerate(phrase):
            postings = self._cache[term]
            for doc_id in list_of_docs:
                term_entry = postings.get(doc_id, {})
                if isinstance(term_entry, dict) and "positions" in term_entry:
                    adjusted = [p - i for p in term_entry["positions"]]
                    term_positions_per_doc[doc_id].append(adjusted)

        final_docs = []
        for doc_id, pos_lists in term_positions_per_doc.items():
            if not pos_lists:
                continue
            intersection = set(pos_lists[0])
            for plist in pos_lists[1:]:
                intersection &= set(plist)
            if intersection:
                final_docs.append(doc_id)
        return natsorted(final_docs)

    def get_all_doc_ids(self):
        meta = self._get_from_db(f"{self.meta_pattern}:doc:merged")
        if not meta:
            return set()
        return set(meta.get("doc_titles", {}).keys())

    def evaluate(self, node):
        all_docs = set(self.get_all_doc_ids())

        if isinstance(node, TermNode):
            if " " in node.term:
                docs = self.phrase_query(node.term.split())
            else:
                docs = self.phrase_query([node.term])
            return docs

        elif isinstance(node, NotNode):
            child_docs = self.evaluate(node.child)
            return list(set(all_docs) - set(child_docs))

        elif isinstance(node, AndNode):
            left = self.evaluate(node.left)
            right = self.evaluate(node.right)
            return list(set(left).intersection(right))

        elif isinstance(node, OrNode):
            left = self.evaluate(node.left)
            right = self.evaluate(node.right)
            return list(set(left).union(right))

        return []

    def rank_by_tfidf(self, terms_in_query, matched_docs):
        scores = {}
        meta = self._get_from_db(f"{self.meta_pattern}:doc:merged")
        total_docs = len(meta.get("doc_titles", {})) if meta else 1

        clean_terms = []
        for t in terms_in_query:
            if t.lower() in {"and", "or", "not"}:
                continue
            clean_terms.extend(t.split())

        for term in clean_terms:
            postings = self.get_postings_from_db(term)
            df_t = len(postings)
            if df_t == 0:
                continue

            idf = math.log((total_docs) / (df_t + 1))
            for doc_id, data in postings.items():
                if doc_id not in matched_docs:
                    continue
                tf = len(data.get("positions", [])) if isinstance(data, dict) else 1
                scores[doc_id] = scores.get(doc_id, 0.0) + tf * idf

        if scores:
            max_score = max(scores.values())
            scores = {d: s / max_score for d, s in scores.items()}

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    def rank_by_count(self, terms_in_query, matched_docs):
        scores = {}
        for term in terms_in_query:
            if term.lower() in {"and", "or", "not"}:
                continue
            for subterm in term.split():
                postings = self.get_postings_from_db(subterm)
                if not postings:
                    continue
                for doc_id, data in postings.items():
                    if doc_id not in matched_docs:
                        continue
                    count = (
                        len(data.get("positions", [])) if isinstance(data, dict) else 1
                    )
                    scores[doc_id] = scores.get(doc_id, 0) + count
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    def response_json(
        self, query_str, terms_in_query, negated_terms, matched_docs, ranked_results
    ):
        meta = self._get_from_db(f"{self.meta_pattern}:doc:merged")
        titles = meta.get("doc_titles", {}) if meta else {}

        summary = {
            "query": query_str,
            "terms": terms_in_query,
            "negated_terms": negated_terms,
            "matched_docs": len(matched_docs),
            "ranked_results": len(ranked_results),
            "top_n_ranked_results": [
                {"doc_id": d, "score": s, "title": titles.get(d, "")}
                for d, s in ranked_results[:10]
            ],
        }
        return json.dumps(summary, indent=4)

    def query(self, query_str: str) -> str:
        tokens = self.tokenize_query(query_str)
        tokens = self.combine_adjacent_terms(tokens)
        parser = Parser(tokens)
        ast = parser.parse()

        matched_docs = self.evaluate(ast)
        terms_in_query = re.findall(r'"([^"]+)"|(\w+)', query_str)
        terms_in_query = [t[0] if t[0] else t[1] for t in terms_in_query]
        negated_terms = re.findall(r'NOT\s+"([^"]+)"', query_str, flags=re.IGNORECASE)

        is_phrase = len(terms_in_query) == 1 and " " in terms_in_query[0]
        if is_phrase:
            matched_docs = self.phrase_query(terms_in_query[0].split())

        if self.info.upper() == "TFIDF":
            ranked_results = self.rank_by_tfidf(terms_in_query, matched_docs)
        elif self.info.upper() == "WORDCOUNT":
            ranked_results = self.rank_by_count(terms_in_query, matched_docs)
        else:
            ranked_results = [(d, 1.0) for d in matched_docs]

        return self.response_json(
            query_str, terms_in_query, negated_terms, matched_docs, ranked_results
        )
