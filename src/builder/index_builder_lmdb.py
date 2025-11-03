import json
import lmdb
import re
import math
from pathlib import Path
from typing import Iterable, Tuple
from core.preprocessor import Preprocessor
from pathlib import Path
import shutil


class IndexBuilderLMDB:
    def __init__(self, info: str = "BOOLEAN", db_path: str = "lmdb_store"):
        self.preprocessor = Preprocessor()
        self.N = 0
        self.doc_titles = {}
        self.doc_lengths = {}
        self.info = info
        self.dstore = "DB1"

        Path(db_path).mkdir(parents=True, exist_ok=True)
        self.env = lmdb.open(
            db_path,
            map_size=10 * 1024 * 1024 * 1024,
            subdir=True,
            lock=True,
            writemap=True,
            sync=False,
        )
        print(f"[INIT] LMDB initialized at '{db_path}'")

        self.index_prefix = "index:term:"
        self.meta_prefix = "meta:"
        self.db_path = db_path

    # -------------------------------------------
    def create_index(
        self, index_id: str, files: Iterable[Tuple[str, str, str]]
    ) -> None:
        missing = self.is_batch_missing(index_id)
        if not missing:
            print(f"[-SKIP-] Batch - {index_id} indexed already.")
            return

        term_posting_list = self.build_term_postings(files)
        self.write_shards_to_lmdb(index_id, term_posting_list)
        self.update_batch_metadata(index_id)
        self.save_document_metadata(index_id)

    def is_batch_missing(self, batch_id: int) -> bool:
        with self.env.begin() as txn:
            val = txn.get(f"{self.meta_prefix}batch:".encode())
            if val:
                try:
                    meta = json.loads(val.decode("utf-8"))
                    return batch_id not in meta.get("completed_batches", [])
                except json.JSONDecodeError:
                    print(f"[WARN] Corrupt metadata for batch {batch_id}.")
                    return True
            else:
                return True

    def write_shards_to_lmdb(self, shard_id: str, shard_data: dict) -> int:
        total_terms = 0
        with self.env.begin(write=True) as txn:
            for term, postings in shard_data.items():
                key = f"{self.index_prefix}{term.lower()}".encode()
                existing_val = txn.get(key)

                if existing_val:
                    existing = json.loads(existing_val.decode("utf-8"))
                    for doc_id, info in postings.items():
                        if doc_id not in existing:
                            existing[doc_id] = info
                        else:
                            existing[doc_id]["positions"] += info.get("positions", [])
                else:
                    existing = postings

                txn.put(key, json.dumps(existing).encode("utf-8"))
                total_terms += 1

        print(f" Batch {shard_id} written with {len(shard_data)} terms.")
        return total_terms

    def build_term_postings(self, files: Iterable[Tuple[str, str, str]]) -> dict:
        term_postings = {}
        self.doc_titles = {}
        self.doc_lengths = {}

        for file_id, title, text in files:
            try:
                tokens = self.preprocessor.tokenize(title + " " + text)
            except Exception:
                tokens = re.findall(r"\w+", (title + " " + text).lower())

            clean_title = title.strip() if title else f"doc_{file_id}"
            self.doc_titles[file_id] = clean_title
            self.doc_lengths[file_id] = len(tokens)

            for pos, term in enumerate(tokens):
                t = term.lower()
                if t not in term_postings:
                    term_postings[t] = {}
                if file_id not in term_postings[t]:
                    term_postings[t][file_id] = {"positions": []}
                term_postings[t][file_id]["positions"].append(pos)

        self.N = len(self.doc_lengths)
        return term_postings

    def update_batch_metadata(self, batch_id: int):
        with self.env.begin(write=True) as txn:
            key = f"{self.meta_prefix}batch:".encode()
            val = txn.get(key)
            if val:
                try:
                    data = json.loads(val.decode("utf-8"))
                except json.JSONDecodeError:
                    data = {}
            else:
                data = {}

            completed = set(data.get("completed_batches", []))
            completed.add(batch_id)
            data["completed_batches"] = sorted(list(completed))
            txn.put(key, json.dumps(data).encode("utf-8"))

        print(f" Updated metadata for batch {batch_id}")

    def save_document_metadata(self, batch_id: int):
        meta = {
            "doc_titles": self.doc_titles,
            "doc_lengths": self.doc_lengths,
            "N": self.N,
        }

        with self.env.begin(write=True) as txn:
            key = f"{self.meta_prefix}doc:batch:{batch_id}".encode()
            txn.put(key, json.dumps(meta).encode("utf-8"))

        print(f" Saved doc metadata for batch {batch_id} (N={self.N})")

    def update_global_metadata(self, batch_ids) -> None:
        merged_titles, merged_lengths = {}, {}
        total_docs = 0

        with self.env.begin() as txn:
            for bid in batch_ids:
                key = f"{self.meta_prefix}doc:batch:{bid}".encode()
                val = txn.get(key)
                if not val:
                    continue
                try:
                    data = json.loads(val.decode("utf-8"))
                except json.JSONDecodeError:
                    continue

                merged_titles.update(data.get("doc_titles", {}))
                merged_lengths.update(data.get("doc_lengths", {}))
                total_docs += data.get("N", 0)

        unified = {
            "doc_titles": merged_titles,
            "doc_lengths": merged_lengths,
            "N": total_docs or len(merged_lengths),
        }

        with self.env.begin(write=True) as txn:
            txn.put(
                f"{self.meta_prefix}doc:merged".encode(),
                json.dumps(unified).encode("utf-8"),
            )

        print(
            f" Stored merged metadata with {len(merged_lengths)} docs (N={total_docs})."
        )

    def read_doc_meta(self):
        with self.env.begin() as txn:
            val = txn.get(f"{self.meta_prefix}doc:merged".encode())
            if val:
                print(json.loads(val.decode("utf-8")))
            else:
                print("[WARN] No merged metadata found.")

    def delete_db_data(self):
        with self.env.begin(write=True) as txn:
            cursor = txn.cursor()
            count = sum(1 for _ in cursor)
            cursor.first()
            deleted = 0
            while cursor.next():
                cursor.delete()
                deleted += 1
        print(f" Deleted {deleted} entries (out of {count})")

    def delete_index(self, index_id: str = "ALL") -> None:

        print(f" Deleting index: {index_id}")

        if index_id == "ALL":
            try:
                self.env.close()
            except Exception:
                pass

            db_path = Path(self.db_path)
            if db_path.exists():
                shutil.rmtree(db_path, ignore_errors=True)
                print(f" Deleted all LMDB data at {db_path}")
            else:
                print(f" No LMDB data found at {db_path}")
            return

        print(" Partial deletion not implemented; use index_id='ALL'.")
