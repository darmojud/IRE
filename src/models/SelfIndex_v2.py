#!/usr/bin/env python3

import warnings

warnings.filterwarnings("ignore")


import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import os
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Tuple, List
import re
from typing import List, Set, Dict, Any
import math
from datetime import datetime

from unidecode import unidecode

# External dependencies you mentioned (should be importable)
from core.preprocessor import Preprocessor
from core.index_base import IndexBase  # your abstract class

from query.Parser import Parser, TermNode, NotNode, AndNode, OrNode
from core.compressor import PostingCompressor
from builder.shard_builder import ShardBuilder
from builder.merger import Merger
from query.query_processor import QueryProcessor
from natsort import natsorted
import random
import re
from builder.index_builder_redis import IndexBuilderRedis
from query.query_redis import QueryOnRedis
from builder.index_builder_lmdb import IndexBuilderLMDB


import base64
import lz4.frame

# --------------- Config ----------------
INDEX_DIR = Path("./wiki_index")
NUM_SHARDS = 8  # number of total shards

INDEX_DIR.mkdir(parents=True, exist_ok=True)


class SelfIndex(IndexBase):

    def __init__(
        self,
        core="SelfIndex",
        info="BOOLEAN",
        dstore="CUSTOM",
        qproc="TERMatat",
        compr="NONE",
        optim="Null",
    ):
        super().__init__(core, info, dstore, qproc, compr, optim)
        self.info = info
        self.dstore = dstore
        self.qproc = qproc
        self.optim = optim
        if self.optim == "Skipping":
            print("[INFO] Skipping optimization enabled.")
            self.compr = "NONE"  # ensure no compression for skipping

        self.compr = compr
        self.compressor = None
        if self.compr == "CODE":
            self.compressor = PostingCompressor()

        self.index_dir = Path(
            f"indexes_{dstore.lower()}/{info.lower()}/compress_{compr.lower()}/optim_{optim}/{INDEX_DIR}"
        )
        Path(self.index_dir).mkdir(parents=True, exist_ok=True)
        self.loaded_index = None
        self.doc_lengths = {}
        self.doc_titles = {}
        self.preprocessor = Preprocessor()
        if self.dstore == "CUSTOM":
            self.index_builder = ShardBuilder(index_dir=self.index_dir, num_shards=8)
        elif self.dstore == "DB1":
            self.index_builder = IndexBuilderRedis(self.info)
        elif self.dstore == "DB2":
            self.index_builder = IndexBuilderLMDB(
                self.info, db_path=str(self.index_dir / "lmdb_store")
            )
        else:
            raise ValueError(f"Unknown data store type: {dstore}")
        self.merger = Merger(
            self.index_dir, self.compr, self.optim, self.info, num_shards=8
        )

    def create_index(
        self, index_id: str, files: Iterable[Tuple[str, str, str]]
    ) -> None:
        return self.index_builder.create_index(index_id, files)

    def merge_index(self, batch_ids):
        if self.dstore == "DB1" or self.dstore == "DB2":
            return self.index_builder.update_global_metadata(batch_ids)
        else:
            return self.merger.merge_shards(batch_ids)

    def query(self, query_str: str) -> str:

        if self.dstore == "DB1":
            query_processor = QueryOnRedis(self.info)
            # print("[QUERY] Using Redis-based query processor.")
            return query_processor.query(query_str)
        elif self.dstore == "DB2":
            from query.query_lmdb import QueryOnLMDB

            query_processor = QueryOnLMDB(
                self.info, db_path=str(self.index_dir / "lmdb_store")
            )
            return query_processor.query(query_str)
        else:
            if self.loaded_index is None:
                # print("[QUERY] Loading index into memory...")
                self.load_index(str(self.index_dir))
            # else:
            #     # print("[QUERY] Using already loaded index in memory.")
            query_processor = QueryProcessor(
                self.loaded_index,
                self.doc_titles,
                self.compressor,
                self.compr,
                self.optim,
                self.info,
                self.qproc,
            )

            return query_processor.query(self.loaded_index, query_str)

    def load_index(self, serialized_index_dump: str) -> dict:

        metadata_path = self.index_dir / "doc_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            self.doc_lengths = metadata.get("doc_lengths", {})
            self.doc_titles = metadata.get("doc_titles", {})
            self.N = metadata.get("N", len(self.doc_lengths))
            print(
                f"[LOAD] Metadata loaded: {len(self.doc_lengths)} documents, N={self.N}."
            )

        path = Path(serialized_index_dump)
        if not path.exists():
            raise FileNotFoundError(f"Index path not found: {serialized_index_dump}")

        self.loaded_index = {}

        for p in path.glob("cumulative_shard_*.json"):
            with open(p, "r", encoding="utf-8") as fh:
                data = json.load(fh)

            for term, postings in data.items():
                if term not in self.loaded_index:
                    self.loaded_index[term] = {}

                for doc_id, entry in postings.items():
                    # Handle list-type (old positional index)
                    if isinstance(entry, list):
                        self.loaded_index[term][doc_id] = {
                            "positions": sorted(set(entry)),
                            "tf": 0.0,
                            "idf": 0.0,
                            "tfidf": 0.0,
                            "title": "",
                        }
                    # Handle dict-type (new TF–IDF index)
                    elif isinstance(entry, dict):
                        doc_entry = {
                            "tf": entry.get("tf", 0.0),
                            "idf": entry.get("idf", 0.0),
                            "tfidf": entry.get("tfidf", 0.0),
                            "title": entry.get("title", ""),
                        }

                        if self.compr != "NONE":
                            # Keep positions compressed as Base64 string
                            doc_entry["positions"] = entry["positions"]
                        else:
                            # Legacy / uncompressed
                            doc_entry["positions"] = sorted(
                                set(entry.get("positions", []))
                            )

                        self.loaded_index[term][doc_id] = doc_entry

                    else:
                        # Fallback (unknown format)
                        self.loaded_index[term][doc_id] = {
                            "positions": [],
                            "tf": 0.0,
                            "idf": 0.0,
                            "tfidf": 0.0,
                            "title": "",
                        }

        print(
            f"[LOAD] Loaded {len(self.loaded_index)} terms into memory (TF–IDF compatible)"
        )

        return self.loaded_index

    def update_index(
        self,
        index_id: str,
        remove_files: Iterable[Tuple[str, str]],
        add_files: Iterable[Tuple[str, str]],
    ) -> None:
        if add_files:
            # add_files -> create a temporary batch id and merge it
            tmp_batch_id = f"update_{index_id}"
            self.create_index(tmp_batch_id, add_files)
            self.merge_shards([tmp_batch_id])

        if remove_files:
            # removal: remove doc ids from cumulative shards
            removed_ids = {rid for rid, _ in remove_files}
            for sid in range(NUM_SHARDS):
                cumulative_path = self.index_dir / f"cumulative_shard_{sid}.json"
                if not cumulative_path.exists():
                    continue
                with open(cumulative_path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                changed = False
                for term, postings in list(data.items()):
                    new_post = [d for d in postings if d not in removed_ids]
                    if len(new_post) != len(postings):
                        data[term] = new_post
                        changed = True
                    if not new_post:
                        del data[term]
                if changed:
                    with open(cumulative_path, "w", encoding="utf-8") as fh:
                        json.dump(data, fh, ensure_ascii=False)
                    print(f"[UPDATE] Removed files from cumulative_shard_{sid}.json")

    def delete_index(self, index_id: str) -> None:

        if index_id == "ALL":
            for p in self.index_dir.iterdir():
                p.unlink()
            print("[DELETE] Removed all index files.")
            return

        # delete batch files that include index_id in their name
        for p in self.index_dir.glob(f"*{index_id}*.json"):
            p.unlink()
            print(f"[DELETE] Removed {p}")

        merged_log_path = self.index_dir / "merged_batches.json"

        if merged_log_path.exists():
            with open(merged_log_path, "r", encoding="utf-8") as f:
                merged_batches = json.load(f)
            if index_id in merged_batches:
                merged_batches.remove(index_id)
                with open(merged_log_path, "w", encoding="utf-8") as f:
                    json.dump(merged_batches, f, indent=2)
                print(f"[DELETE] Removed {index_id} from merged_batches.json")

    def list_indices(self) -> Iterable[str]:
        batch_ids = set()
        for p in self.index_dir.glob("index_shard_*_batch_*.json"):
            parts = p.stem.split("_")
            # format: index_shard_{sid}_batch_{batchid}
            if "batch" in parts:
                batch_idx = parts.index("batch")
                batch_ids.add("_".join(parts[batch_idx + 1 :]))
        return sorted(list(batch_ids))

    def list_indexed_files(self, index_id: str) -> Iterable[str]:
        docs = set()
        for p in self.index_dir.glob("cumulative_shard_*.json"):
            with open(p, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            for postings in data.values():
                docs.update(postings)
        return sorted(list(docs))
