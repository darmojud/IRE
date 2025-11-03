from query.Parser import Parser, TermNode, NotNode, AndNode, OrNode
import base64
import json
import lz4
from natsort import natsorted
import re


class QueryProcessor:

    def __init__(
        self,
        loaded_index=None,
        doc_titles=None,
        compressor=None,
        compr="NONE",
        optim="Skipping",
        info="TFIDF",
        qproc="TERMatat",
    ):
        self.loaded_index = loaded_index or {}
        self.doc_titles = doc_titles or {}
        self.compressor = compressor
        self.compr = compr  # 'NONE', 'CODE', 'CLIB'
        self.optim = optim  # 'Skipping' or other
        self.info = info  # 'TFIDF', 'WORDCOUNT', etc.
        self.qproc = qproc  # 'TERMatat' or 'DOCatat'
        print(f"[QUERY PROCESSOR] Initialized with qproc={self.qproc}")
        self.verbose = False

        print(self.compr)

    def tokenize_query(self, query):
        pattern = r"\"[^\"]+\"|\(|\)|AND|OR|NOT|[a-zA-Z0-9]+"
        tokens = re.findall(pattern, query, flags=re.IGNORECASE)
        return [
            t.upper() if t.upper() in {"AND", "OR", "NOT"} else t.strip('"')
            for t in tokens
        ]

    def combine_adjacent_terms(self, tokens):
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
        """Recursively collect all terms under a node (for printing)."""
        if isinstance(node, TermNode):
            return [node.term]
        elif isinstance(node, NotNode):
            return [f"NOT({self.get_terms(node.child)})"]
        elif isinstance(node, AndNode) or isinstance(node, OrNode):
            return self.get_terms(node.left) + self.get_terms(node.right)
        return []

    def get_positions(self, term, doc_id) -> list[int]:
        """
        Returns a list of integer positions for the given term and document.
        Works with compressed or uncompressed indexes.
        """
        entry = self.loaded_index[term][doc_id]

        # Compressed positions
        if self.compr == "CODE" and self.compressor:
            compressed_bytes = base64.b64decode(entry["positions"])
            positions = self.compressor.decompress(compressed_bytes)
            # positions = self.compressor.gap_decode(
            #     self.compressor.vb_decode(compressed_bytes)
            # )
            return [int(p) for p in positions]  # ensure integers
        elif self.compr == "CLIB":
            compressed_bytes = base64.b64decode(entry["positions"])
            decompressed_json = lz4.frame.decompress(compressed_bytes).decode("utf-8")
            positions = json.loads(decompressed_json)
            return [int(p) for p in positions]  # ensure integers

        # Uncompressed positions
        elif "positions" in entry:
            return [int(p) for p in entry.get("positions", [])]  # ensure integers

        return []

    def matched_docs_with_skips(self, query_terms: list[str]) -> list[str]:
        print("Matched docs with skip pointers optimization.")
        if not query_terms:
            return []

        # Step 1: Fetch posting lists for all terms
        postings_list = []
        for term in query_terms:
            if term not in self.loaded_index:
                return []  # Term not in index
            term_postings = list(self.loaded_index[term].keys())
            term_postings.sort()  # Ensure ascending order
            postings_list.append(term_postings)

        # Step 2: Sort by length to intersect smaller lists first
        postings_list.sort(key=len)

        # Step 3: Intersect using skip pointers
        # Start with first (smallest) posting list
        result = postings_list[0]
        for plist in postings_list[1:]:
            result = self.intersect_with_skip_pointers(result, plist)

            if not result:
                return []

        return natsorted(result)

    def intersect_with_skip_pointers(
        self, list1: list[int], list2: list[int]
    ) -> list[int]:
        answer = []
        i = j = 0
        len1, len2 = len(list1), len(list2)

        while i < len1 and j < len2:
            doc1, doc2 = list1[i], list2[j]

            if doc1 == doc2:
                answer.append(doc1)
                i += 1
                j += 1
            elif doc1 < doc2:
                # Check if we can skip
                skip_doc = self.get_skip_doc(list1[i:], doc2)
                if skip_doc:
                    i += skip_doc
                else:
                    i += 1
            else:
                # doc2 < doc1
                skip_doc = self.get_skip_doc(list2[j:], doc1)
                if skip_doc:
                    j += skip_doc
                else:
                    j += 1

        return answer

    def get_skip_doc(self, posting_sublist: list[int], target: int) -> int:
        # Simple skip: jump ahead in fixed block, e.g., sqrt(len(list))
        n = len(posting_sublist)
        if n <= 1:
            return None
        skip_block = int(n**0.5)
        if posting_sublist[skip_block - 1] < target:
            return skip_block
        return None

    def matched_docs(self, query_terms):
        postings_list = []
        for term in query_terms:
            if term in self.loaded_index:
                # Handle both dict and list postings
                postings = self.loaded_index[term]
                if isinstance(postings, dict):
                    postings_list.append(list(postings.keys()))
                elif isinstance(postings, list):
                    # Convert list of doc_ids to consistent format
                    postings_list.append(postings)
                else:
                    continue
            else:
                return []  # Term not found at all

        if not postings_list:
            return []

        # Sort by length to optimize intersection
        ordered_postings = sorted(postings_list, key=len)
        intersection_set = set(ordered_postings[0])
        for postings in ordered_postings[1:]:
            intersection_set &= set(postings)

        return natsorted(intersection_set)

    def phrase_query(self, phrase: list[str]) -> list[str]:

        # Step 1: Get candidate documents containing all terms
        if self.optim == "Skipping":
            list_of_docs = self.matched_docs_with_skips(phrase)
        else:
            list_of_docs = self.matched_docs(phrase)
        if not list_of_docs:
            return []

        # Step 2: If single word, return matched docs
        if len(phrase) == 1:
            return list_of_docs

        # Step 3: Check phrase alignment
        final_docs = []

        for doc_id in list_of_docs:
            term_positions = []
            for term in phrase:
                if self.compr != "NONE":
                    positions = self.get_positions(term, doc_id)
                    term_positions.append(set(positions))  # set of ints
                else:
                    # Convert positions to int
                    positions = self.loaded_index[term][doc_id].get("positions", [])
                    positions = [int(p) for p in positions]  # <-- critical
                    term_positions.append(set(positions))

            # Step 4: Check phrase occurrence
            first_term_positions = term_positions[0]
            found = False
            for p in first_term_positions:
                if all((p + i) in term_positions[i] for i in range(1, len(phrase))):
                    found = True
                    break
            if found:
                final_docs.append(doc_id)

        return natsorted(final_docs)

    def evaluate(self, node):
        # self.verbose = True  # Enable detailed logging
        all_docs = list(
            {
                doc_id
                for term_docs in self.loaded_index.values()
                for doc_id in term_docs.keys()
            }
        )

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
            if self.qproc == "DOCatat":
                # Collect all terms under this AND node
                terms = self.get_terms(node)
                docs = self.matched_docs_daat(terms)
                if self.verbose:
                    print(f"DAAT AND terms {terms} -> {preview(docs)}")
                return docs
            else:
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
        scores = {}

        if not matched_docs:
            print("[RANK] No matched docs — skipping ranking.")
            return []

        for term in terms_in_query:
            postings = self.loaded_index.get(term, {})
            if not postings:
                continue
            for doc_id, data in postings.items():
                if doc_id not in matched_docs:
                    continue
                # Handle dict or list entries gracefully
                tfidf_val = 0.0
                if isinstance(data, dict):
                    tfidf_val = data.get("tfidf", 0.0)
                elif isinstance(data, (int, float)):
                    tfidf_val = data
                scores[doc_id] = scores.get(doc_id, 0.0) + tfidf_val

        ranked_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        if not ranked_results:
            print(f"[RANK] No TF–IDF scores found for terms {terms_in_query}")

        # Attach title info (from loaded index or doc_metadata)

        return ranked_results

    def rank_by_count(self, terms_in_query, matched_docs):
        scores = {}

        if not matched_docs:
            print("[RANK] No matched docs — skipping ranking.")
            return []

        for term in terms_in_query:
            postings = self.loaded_index.get(term, {})
            for doc_id, data in postings.items():
                if doc_id not in matched_docs:
                    continue

                # Count occurrences as score
                count = 0
                if isinstance(data, dict):
                    count = len(data.get("positions", []))
                elif isinstance(data, list):
                    count = len(data)

                scores[doc_id] = scores.get(doc_id, 0) + count

        # Sort descending by count
        ranked_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        if not ranked_results:
            print(f"[RANK] No counts found for terms {terms_in_query}")

        return ranked_results

    def matched_docs_daat(self, query_terms: list[str]) -> list[str]:
        print("Matched docs with DAAT approach.")
        if not query_terms:
            return []

        # Step 1: Fetch posting lists
        posting_lists = []
        for term in query_terms:
            if term not in self.loaded_index:
                return []  # term not found
            docs = list(self.loaded_index[term].keys())
            docs.sort()
            posting_lists.append(docs)

        # Step 2: Initialize pointers
        pointers = [0] * len(posting_lists)
        results = []

        while True:
            # Step 3: Check if any pointer is at the end
            if any(p >= len(posting_lists[i]) for i, p in enumerate(pointers)):
                break

            # Step 4: Get current doc_ids
            current_docs = [
                posting_lists[i][pointers[i]] for i in range(len(posting_lists))
            ]
            min_doc = min(current_docs)
            max_doc = max(current_docs)

            if min_doc == max_doc:
                # All lists point to same doc -> add to result
                results.append(min_doc)
                # Advance all pointers
                pointers = [p + 1 for p in pointers]
            else:
                # Advance pointers where current_doc < max_doc
                for i, p in enumerate(pointers):
                    if current_docs[i] < max_doc:
                        # Check skip pointer if available
                        term = query_terms[i]
                        current_doc = posting_lists[i][p]
                        skip_to_doc = self.loaded_index[term][current_doc].get(
                            "skip_to"
                        )
                        if (
                            skip_to_doc
                            and skip_to_doc <= max_doc
                            and skip_to_doc in self.loaded_index[term]
                        ):
                            # Jump to skip_to index
                            try:
                                new_pointer = posting_lists[i].index(skip_to_doc)
                                pointers[i] = new_pointer
                            except ValueError:
                                # fallback: advance by 1
                                pointers[i] += 1
                        else:
                            pointers[i] += 1

        return natsorted(results)

    def print_query_summary(
        self,
        query_str: str,
        terms_in_query: list[str],
        negated_terms: list[str],
        matched_docs: set[str],
        ranked_results: list[tuple[str, float]],
    ) -> None:
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
                if doc_id in self.doc_titles:
                    title = self.doc_titles[doc_id]
                else:
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
            if doc_id in self.doc_titles:
                title = self.doc_titles[doc_id]
            else:
                title = ""
            summary["top_n_ranked_results"].append(
                {"doc_id": doc_id, "score": score, "title": title}
            )
        return json.dumps(summary, indent=4)

    def query(self, loaded_index, query_str: str) -> str:

        tokens = self.tokenize_query(query_str)
        tokens = self.combine_adjacent_terms(tokens)
        parser = Parser(tokens)
        # build abstract syntax tree - ast
        ast = parser.parse()

        if self.loaded_index is None:
            print("[QUERY] Loading index into memory...")
            # self.load_index(str(self.index_dir))
            self.loaded_index = loaded_index
        else:
            print("[QUERY] Using already loaded index in memory.")

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
