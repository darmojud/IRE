from query.Parser import Parser, TermNode, NotNode, AndNode, OrNode
import base64
import json
import lz4
from natsort import natsorted
import re
import redis
import math


class QueryOnRedis:

    def __init__(
        self,
        info="BOOLEAN",
    ):
        self.loaded_index = None
        self.doc_titles = {}
        self.verbose = False
        self.info = info
        self.db = redis.StrictRedis(host="localhost", port=6379, db=0)
        self.index_pattern = f"index:term"
        self.meta_pattern = f"meta"

    def get_postings_from_db(self, term: str) -> dict:
        # Return cached result if already fetched in this query
        if hasattr(self, "_cache") and term in self._cache:
            return self._cache[term]

        # Build key
        key = f"{self.index_pattern}:{term.lower()}"

        # Fetch from Redis
        val = self.db.get(key)
        postings = {}
        if val:
            try:
                postings = json.loads(val.decode("utf-8"))
            except json.JSONDecodeError:
                print(f"[WARN] Corrupt postings JSON for term '{term}'")

        # Cache it for later reuse
        if not hasattr(self, "_cache"):
            self._cache = {}
        self._cache[term] = postings

        return postings

    @staticmethod
    def tokenize_query(query):
        # pattern = pattern = r"\"[^\"]+\"|\(|\)|AND|OR|NOT|\w+"
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

    def matched_docs(self, query_terms):

        postings_list = []

        for term in query_terms:
            # Fetch postings from Redis
            postings = self.get_postings_from_db(term)
            if not postings:
                print(f"[MISS] No postings found for term '{term}'")
                return []
            postings_list.append(list(postings.keys()))

        if not postings_list:
            return []

        # Sort postings by length to optimize intersection
        ordered_postings = sorted(postings_list, key=len)
        intersection_set = set(ordered_postings[0])
        for postings in ordered_postings[1:]:
            intersection_set &= set(postings)

        # Return naturally sorted list of document IDs
        return natsorted(intersection_set)

    def phrase_query(self, phrase):

        if hasattr(self, "_cache"):
            cached_terms = [t for t in phrase if t in self._cache]
        else:
            self._cache = {}
            cached_terms = []

        terms_to_fetch = [t for t in phrase if t not in cached_terms]

        if terms_to_fetch:
            pipe = self.db.pipeline()
            for term in terms_to_fetch:
                pipe.get(f"{self.index_pattern}:{term.lower()}")
            results = pipe.execute()

            for term, val in zip(terms_to_fetch, results):
                postings = {}
                if val:
                    try:
                        postings = json.loads(val.decode("utf-8"))
                    except json.JSONDecodeError:
                        print(f"[WARN] Corrupt postings JSON for term '{term}'")
                self._cache[term] = postings

        postings_dicts = [self._cache[t] for t in phrase]

        # Get document IDs common to all terms
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
                    positions = term_entry["positions"]
                    # Adjust offset by position in phrase
                    adjusted = [p - i for p in positions]
                    term_positions_per_doc[doc_id].append(adjusted)

        # Keep docs where all terms align (same adjusted position)
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

    def get_all_doc_ids(self) -> set[str]:
        data = self.db.get(f"{self.meta_pattern}:doc:merged")
        if not data:
            return set()
        meta = json.loads(data)
        return set(meta.get("doc_titles", {}).keys())

    def evaluate(self, node):
        # self.verbose = True  # Enable detailed logging
        all_docs = set(self.get_all_doc_ids())

        def preview(result: list[str]) -> str:
            if not result:
                return "[] (count=0)"
            preview = result[:5]
            more = f"... (+{len(result) - 5} more)" if len(result) > 5 else ""
            return f"{preview} (count={len(result)}) {more}"

        if isinstance(node, TermNode):
            # Handle phrase queries like "modern art"
            if " " in node.term:
                phrase_terms = node.term.split()
                docs = self.phrase_query(phrase_terms)
                if self.verbose:
                    print(f"Phrase '{node.term}' -> {preview(docs)}")
            else:
                docs = self.phrase_query([node.term])
                if self.verbose:
                    print(f"Term '{node.term}' -> {preview(docs)}")
            return docs

        elif isinstance(node, NotNode):
            child_docs = self.evaluate(node.child)
            result = list(set(all_docs) - set(child_docs))

            # Determine name of child node for printing
            child_name = self.get_terms(node.child)
            if self.verbose:
                print(f"NOT '{child_name}' -> {preview(result)}")
            return result

        elif isinstance(node, AndNode):
            left = self.evaluate(node.left)
            right = self.evaluate(node.right)
            result = list(set(left).intersection(right))
            if self.verbose:
                print(
                    f"AND Left: {self.get_terms(node.left)} Right: {self.get_terms(node.right)} -> {preview(result)}"
                )
            return result

        elif isinstance(node, OrNode):
            left = self.evaluate(node.left)
            right = self.evaluate(node.right)
            result = list(set(left).union(right))
            if self.verbose:
                print(
                    f"OR Left: {self.get_terms(node.left)} Right: {self.get_terms(node.right)} -> {preview(result)}"
                )
            return result

        return []

    def rank_by_tfidf(self, terms_in_query, matched_docs):
        """
        Rank documents by TF-IDF, computing values on the fly from Redis.
        """
        scores = {}

        # Get total number of documents
        meta_data = self.db.get(f"{self.meta_pattern}:doc:merged")
        if meta_data:
            try:
                meta = json.loads(meta_data.decode("utf-8"))
                total_docs = len(meta.get("doc_titles", {}))
            except json.JSONDecodeError:
                total_docs = 1  # fallback
        else:
            total_docs = 1  # fallback

        # Preprocess terms: ignore logical operators, expand phrases
        clean_terms = []
        for t in terms_in_query:
            if t.lower() in {"and", "or", "not"}:
                continue
            clean_terms.extend(t.split())

        # Compute TF-IDF for each term
        for term in clean_terms:
            postings = self.get_postings_from_db(term)
            df_t = len(postings)
            if df_t == 0:
                continue

            idf = math.log((total_docs) / (df_t + 1))  # add 1 to avoid div by 0

            for doc_id, data in postings.items():
                if doc_id not in matched_docs:
                    continue

                # Term frequency (raw count)
                tf = 0
                if isinstance(data, dict):
                    tf = len(data.get("positions", []))
                elif isinstance(data, (int, float)):
                    tf = data

                scores[doc_id] = scores.get(doc_id, 0.0) + tf * idf

        # Normalize scores to [0,1]
        if scores:
            max_score = max(scores.values())
            scores = {doc_id: score / max_score for doc_id, score in scores.items()}

        # Return sorted list
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    def rank_by_count(self, terms_in_query, matched_docs):
        scores = {}

        for term in terms_in_query:
            # Skip logic operators
            if term.lower() in {"and", "or", "not"}:
                continue

            # Handle phrase terms
            subterms = term.split() if " " in term else [term]

            for subterm in subterms:
                postings = self.get_postings_from_db(subterm)
                if not postings:
                    continue

                for doc_id, data in postings.items():
                    if doc_id not in matched_docs:
                        continue
                    count = (
                        len(data.get("positions", []))
                        if isinstance(data, dict)
                        else len(data)
                    )
                    scores[doc_id] = scores.get(doc_id, 0) + count

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    def print_query_summary(
        self,
        query_str: str,
        terms_in_query: list[str],
        negated_terms: list[str],
        matched_docs: set[str],
        ranked_results: list[tuple[str, float]],
    ) -> None:

        if hasattr(self, "db") is False or self.db is None:
            print("[WARN] Redis handle not available — cannot fetch metadata.")
            return

        print("\n--- Query Summary ---")
        print(f"Query: {query_str}")
        print(f"Terms: {terms_in_query}")
        print(f"Negated Terms: {negated_terms}")
        print(f"Total Matched Docs: {len(matched_docs)}")
        print(f"Ranked Results Count: {len(ranked_results)}")

        if ranked_results:
            print("\n--- Top Ranked Results ---")
            print(f"{'Doc ID':<10} {'Score':<10} {'Title'}")
            print("-" * 60)

            for doc_id, score in ranked_results[:10]:
                doc_data = self.db.get(f"{self.meta_pattern}:doc:merged")
                if doc_data:
                    try:
                        metadata = json.loads(doc_data.decode("utf-8"))
                        title = metadata.get("doc_titles", {}).get(doc_id, "")
                    except json.JSONDecodeError:
                        title = ""
                else:
                    print("[WARN] Unable to fetch document metadata from Redis.")
                    title = ""
                print(f"{doc_id:<10} {score:<10.6f} {title}")
        else:
            print("No results found.")

    def response_json(
        self, query_str, terms_in_query, negated_terms, matched_docs, ranked_results
    ):
        summary = {
            "query": query_str,
            "terms": terms_in_query,
            "negated_terms": negated_terms,
            "matched_docs": len(matched_docs),
            "ranked_results": len(ranked_results),
            "top_n_ranked_results": [],
        }

        for doc_id, score in ranked_results[:10]:
            doc_data = self.db.get(f"{self.meta_pattern}:doc:merged")
            if doc_data:
                try:
                    metadata = json.loads(doc_data.decode("utf-8"))
                    title = metadata.get("doc_titles", {}).get(doc_id, "")
                except json.JSONDecodeError:
                    title = ""
            else:
                # print("[WARN] Unable to fetch document metadata from Redis.")
                title = ""
            summary["top_n_ranked_results"].append(
                {"doc_id": doc_id, "score": score, "title": title}
            )
        return json.dumps(summary, indent=4)

    def query(self, query_str: str) -> str:

        tokens = self.tokenize_query(query_str)
        tokens = self.combine_adjacent_terms(tokens)
        parser = Parser(tokens)
        # build abstract syntax tree - ast
        ast = parser.parse()

        # if self.loaded_index is None:
        #     print("[QUERY] Using RocksDB backend — no need to load index.")

        matched_docs = self.evaluate(ast)

        # Extract terms and negated terms
        terms_in_query = re.findall(r'"([^"]+)"|(\w+)', query_str)
        terms_in_query = [t[0] if t[0] else t[1] for t in terms_in_query]
        negated_terms = re.findall(r'NOT\s+"([^"]+)"', query_str, flags=re.IGNORECASE)

        # Determine if query is a phrase
        is_phrase = len(terms_in_query) == 1 and " " in terms_in_query[0]

        if is_phrase:
            phrase_terms = terms_in_query[0].split()
            matched_docs = self.phrase_query(phrase_terms)
            if self.verbose:
                print(
                    f"[QUERY] Phrase query detected: {phrase_terms} -> {len(matched_docs)} docs"
                )

        if self.info.upper() == "TFIDF":
            ranked_results = self.rank_by_tfidf(
                phrase_terms if is_phrase else terms_in_query, matched_docs
            )
        elif self.info.upper() == "WORDCOUNT":
            ranked_results = self.rank_by_count(
                phrase_terms if is_phrase else terms_in_query, matched_docs
            )
        else:
            ranked_results = [(doc_id, 1.0) for doc_id in matched_docs]

        return self.response_json(
            query_str=query_str,
            terms_in_query=terms_in_query,
            negated_terms=negated_terms,
            matched_docs=matched_docs,
            ranked_results=ranked_results,
        )
