import json
import math
from pathlib import Path
from typing import Dict, List, Any
from core.utils import safe_write_json, is_valid_json
from collections import defaultdict
import re
from typing import Iterable, Tuple
from core.preprocessor import Preprocessor
import redis


class IndexBuilderRedis:

    def __init__(self, info: str = "BOOLEAN"):
        self.preprocessor = Preprocessor()
        self.N = 0  # total number of documents
        self.doc_titles = {}
        self.doc_lengths = {}
        # ----- Data store setup -----
        self.dstore = "DB1"
        self.db = redis.StrictRedis(host="localhost", port=6379, db=0)
        print(f"[INIT] Redis initialized on localhost:6379")
        self.info = info
        # self.index_pattern = f"index:{self.info.lower()}:term"
        # self.meta_pattern = f"meta:{self.info.lower()}"
        self.index_pattern = f"index:term"
        self.meta_pattern = f"meta"

    def create_index(
        self, index_id: str, files: Iterable[Tuple[str, str, str]]
    ) -> None:
        missing = self.is_missing_batch(index_id)
        if not missing:
            print(f"[-SKIP-] Batch - {index_id} indexed already.")
            return

        term_posting_list = self.built_postinglists(files)
        self.write_this_batch_to_redis(index_id, term_posting_list)
        self.save_batch_metadata(index_id)
        self.save_document_metadata(index_id)

    def is_missing_batch(self, batch_id: int) -> bool:
        # return missing batch ids
        key = f"meta:batch:"
        val = self.db.get(key)

        if val:
            try:
                batch_info = json.loads(val.decode("utf-8"))
                completed_batches = set(batch_info.get("completed_batches", []))
                return batch_id not in completed_batches
            except json.JSONDecodeError:
                print(
                    f"[WARN] Corrupt Redis metadata for batch {batch_id}, treating as missing."
                )
                return True
        else:
            return True

    def write_this_batch_to_redis(self, shard_id: str, shard_data: dict) -> int:
        if not shard_data:
            print(f"[EMPTY] Shard {shard_id} has no data, skipping.")
            return 0

        pipeline = self.db.pipeline()
        total_terms = 0

        for term, postings in shard_data.items():
            key = f"{self.index_pattern}:{str(term).lower()}"

            existing_val = self.db.get(key)
            if existing_val:
                existing = json.loads(existing_val.decode("utf-8"))
                for doc_id, info in postings.items():
                    if doc_id not in existing:
                        existing[doc_id] = info
                    else:
                        existing_positions = existing[doc_id].get("positions", [])
                        info_positions = info.get("positions", [])
                        existing[doc_id]["positions"] = (
                            existing_positions + info_positions
                        )

                        # Merge other keys
                        for k, v in info.items():
                            if k != "positions":
                                existing[doc_id][k] = v
                val = json.dumps(existing)
            else:
                val = json.dumps(postings)

            pipeline.set(key, val)
            total_terms += 1

        pipeline.execute()
        print(f"[REDIS] Batch {shard_id} written with {len(shard_data)} terms.")
        return total_terms

    def delete_redis_index(self):
        if not hasattr(self, "db") or self.db is None:
            raise RuntimeError("Redis handle not initialized.")

        meta_pattern = f"meta:*"
        index_pattern = f"index:*"
        pipeline = self.db.pipeline()
        for key in self.db.scan_iter(match=meta_pattern):
            pipeline.delete(key)

        pipeline.execute()
        print("Deleted MetaData")

        for key in self.db.scan_iter(match=index_pattern):
            pipeline.delete(key)
        pipeline.execute()
        print("Deleted Index")

    def built_postinglists(self, files: Iterable[Tuple[str, str, str]]) -> dict:

        term_postings = {}
        self.doc_titles = {}
        self.doc_lengths = {}

        for file_id, title, text in files:
            # Tokenize
            try:
                tokens = self.preprocessor.tokenize(title + " " + text)
            except Exception:
                tokens = re.findall(r"\w+", str(title + " " + text or "").lower())

            clean_title = title.strip() if title else f"doc_{file_id}"
            self.doc_titles[file_id] = clean_title
            self.doc_lengths[file_id] = len(tokens)

            for pos, term in enumerate(tokens):
                term_lower = term.lower()
                if term_lower not in term_postings:
                    term_postings[term_lower] = {}

                if file_id not in term_postings[term_lower]:
                    term_postings[term_lower][file_id] = {"positions": []}

                term_postings[term_lower][file_id]["positions"].append(pos)

        self.N = len(self.doc_lengths)
        return term_postings

    def save_batch_metadata(self, batch_id: int):

        if not hasattr(self, "db") or self.db is None:
            raise RuntimeError("Redis handle not initialized. Cannot update metadata.")

        key = f"meta:batch:"
        val = self.db.get(key)
        if val:
            try:
                batch_info = json.loads(val.decode("utf-8"))
                print(f"[REDIS] Loaded existing metadata for batch {batch_id}.")
            except json.JSONDecodeError:
                batch_info = {}
                print(f"[WARN] Corrupt metadata for batch {batch_id}, resetting.")
        else:
            batch_info = {}
            print(f"[REDIS] No existing metadata for batch {batch_id}, creating new.")

        # Update completed_batches
        completed = set(batch_info.get("completed_batches", []))
        completed.add(batch_id)
        batch_info["completed_batches"] = sorted(list(completed))

        # Write back to Redis
        self.db.set(key, json.dumps(batch_info))
        print(f"[REDIS] Updated metadata for batch {batch_id} -> {batch_info}")

    def save_document_metadata(self, batch_id: int) -> None:

        if not hasattr(self, "db") or self.db is None:
            print("[WARN] Redis handle not available — cannot merge metadata.")
            return

        key = f"{self.meta_pattern}:doc:batch:{batch_id}"
        val = self.db.get(key)
        if not val:
            print(
                f"[WARN] Document metadata does not exist for {batch_id}. Creating new metadata entry."
            )
        else:
            print(f"[SAVE] Saving document metadata to Redis key '{key}'")

        merged_metadata = {
            "doc_titles": self.doc_titles,
            "doc_lengths": self.doc_lengths,
            "N": self.N,
        }
        pipeline = self.db.pipeline()
        self.db.set(key, json.dumps(merged_metadata))
        pipeline.execute()
        print(
            f"[SAVE] Stored metadata in '{key}' with {len(self.doc_lengths)} documents (N={self.N})."
        )

    def update_global_metadata(self, batch_ids) -> None:

        if not hasattr(self, "db") or self.db is None:
            print("[WARN] Redis handle not available — cannot merge metadata.")
            return

        merged_titles = {}
        merged_lengths = {}
        total_docs = 0
        any_valid = False

        for bid in batch_ids:
            key = f"{self.meta_pattern}:doc:batch:{bid}"
            val = self.db.get(key)

            if not val:
                # print(f"[WARN] No metadata found for batch {bid}, skipping.")
                continue

            try:
                print(f"[MERGE] Loading metadata from '{key}'")
                data = json.loads(val.decode("utf-8"))
            except (json.JSONDecodeError, AttributeError):
                print(f"[WARN] Corrupt metadata in {key}; skipping.")
                continue

            any_valid = True

            # Merge data safely
            if "doc_titles" in data and isinstance(data["doc_titles"], dict):
                merged_titles.update(data["doc_titles"])
            if "doc_lengths" in data and isinstance(data["doc_lengths"], dict):
                merged_lengths.update(data["doc_lengths"])
            if "N" in data and isinstance(data["N"], int):
                total_docs += data["N"]

        if not any_valid:
            print("[-SKIP-] No valid metadata found in any batch.")
            return

        # Update internal state
        self.doc_titles = merged_titles
        self.doc_lengths = merged_lengths
        self.N = total_docs or len(merged_lengths)

        # Prepare merged data
        merged_metadata = {
            "doc_titles": self.doc_titles,
            "doc_lengths": self.doc_lengths,
            "N": self.N,
        }

        # Write unified data to single static key
        unified_key = f"{self.meta_pattern}:doc:merged"
        self.db.set(unified_key, json.dumps(merged_metadata))

        print(
            f"[MERGE] Stored merged metadata in '{unified_key}' "
            f"with {len(merged_lengths)} documents (N={self.N})."
        )

        pattern = f"{self.meta_pattern}:doc:batch:*"
        for key in self.db.scan_iter(match=pattern):
            print(f"[CLEANUP] Deleting temporary metadata key: {key.decode()}")
            self.db.delete(key)

    def read_document_metadata(self):
        if not hasattr(self, "db") or self.db is None:
            print("[WARN] Redis handle not available — cannot merge metadata.")
            return
        pattern = f"{self.meta_pattern}:doc:merged"
        for key in self.db.scan_iter(match=pattern):
            print(self.db.get(key))

    def delete_redis_index(self):
        if not hasattr(self, "db") or self.db is None:
            raise RuntimeError("Redis handle not initialized.")

        meta_pattern = f"meta:*"
        index_pattern = f"index:*"
        pipeline = self.db.pipeline()
        for key in self.db.scan_iter(match=meta_pattern):
            pipeline.delete(key)

        pipeline.execute()
        print("Deleted MetaData")

        for key in self.db.scan_iter(match=index_pattern):
            pipeline.delete(key)
        pipeline.execute()
        print("Deleted Index")
