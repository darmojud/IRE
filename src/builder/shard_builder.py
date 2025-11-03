import json
import math
from pathlib import Path
from typing import Dict, List, Any
from core.utils import safe_write_json, is_valid_json
from collections import defaultdict
import re
from typing import Iterable, Tuple
from core.preprocessor import Preprocessor


class ShardBuilder:

    def __init__(self, index_dir: Path, num_shards: int = 8):
        self.index_dir = index_dir
        self.preprocessor = Preprocessor()
        self.N = 0  # total number of documents
        self.doc_titles = {}
        self.doc_lengths = {}
        self.num_shards = num_shards

    def get_shard(self, term):
        return hash(term) % self.num_shards

    def create_index(self, index_id, files):

        # Identify missing shards
        missing_shards = self.get_missing_shards(index_id)
        if not missing_shards:
            print(f"[-SKIP-] All shards for batch {index_id} already valid.")
            return

        # build shard data for missing shards
        shard_data = self.build_shard_data(files, missing_shards)

        # write shard files to the disk
        self.write_shard_files(shard_data, index_id)

        # updata batch metadata and document metadata
        self.save_batch_metadata(index_id, missing_shards)
        self.save_document_metadata(index_id)

    def get_missing_shards(self, index_id: str) -> list[int]:

        # detect missing shards based on the completed shards inforamation in batch_metadata.json file
        shard_paths = [
            self.index_dir / f"index_shard_{sid}_batch_{index_id}.json"
            for sid in range(self.num_shards)
        ]
        metadata_path = self.index_dir / "batch_metadata.json"
        missing_shards = []

        if metadata_path.exists():
            try:
                with open(metadata_path, "r", encoding="utf-8") as fh:
                    metadata = json.load(fh)
                index = metadata.get("index", {})
                batch_info = index.get(index_id, {})
                completed = set(batch_info.get("completed_shards", []))
                missing_shards = [
                    sid for sid in range(self.num_shards) if sid not in completed
                ]
                print(
                    f"[META] Found metadata for batch {index_id} → Missing shards: {missing_shards}"
                )
            except json.JSONDecodeError:
                print(
                    f"[WARN] Corrupt metadata for batch {index_id}, falling back to file scan."
                )
                missing_shards = [
                    sid for sid, p in enumerate(shard_paths) if not is_valid_json(p)
                ]
        else:
            missing_shards = [
                sid for sid, p in enumerate(shard_paths) if not is_valid_json(p)
            ]

        return missing_shards

    def build_shard_data(self, files, missing_shards):

        shard_data = {
            sid: defaultdict(lambda: defaultdict(lambda: {"positions": []}))
            for sid in missing_shards
        }
        self.doc_titles = {}
        self.doc_lengths = {}

        for file_id, title, text in files:
            try:
                tokens = self.preprocessor.tokenize(title + " " + text)
            except Exception:
                tokens = re.findall(r"\w+", str(title + " " + text or "").lower())

            clean_title = title.strip() if title else f"doc_{file_id}"
            self.doc_titles[file_id] = clean_title
            self.doc_lengths[file_id] = len(tokens)

            for pos, term in enumerate(tokens):
                sid = self.get_shard(term)
                if sid not in shard_data:
                    continue
                shard_data[sid][term][file_id]["positions"].append(pos)

        self.N = len(self.doc_lengths)
        return shard_data

    def write_shard_files(self, shard_data: dict, index_id: str):

        for sid, data in shard_data.items():
            out_path = self.index_dir / f"index_shard_{sid}_batch_{index_id}.json"
            if not data:
                print(f"[EMPTY] Shard {sid} has no terms; skipping write.")
                continue
            try:
                safe_write_json(data, out_path)
            except Exception as e:
                print(f"[ERROR] Failed to write {out_path}: {e}")

    def save_batch_metadata(self, index_id: str, completed_shards: list[int]):
        meta_path = self.index_dir / "batch_metadata.json"
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
            except Exception:
                metadata = {}
        else:
            metadata = {}

        index = metadata.setdefault("index", {})
        batch_info = index.setdefault(index_id, {})
        shards_set = set(batch_info.get("completed_shards", []))
        shards_set.update(completed_shards)
        batch_info["completed_shards"] = sorted(list(shards_set))

        try:
            safe_write_json(metadata, meta_path)
            print(
                f"[META] Updated metadata for batch {index_id} → {meta_path} "
                f"({len(batch_info['completed_shards'])} shards completed)"
            )
        except Exception as e:
            print(f"[ERROR] Failed to write metadata: {e}")

    def save_document_metadata(self, index_id: str):
        doc_meta_path = self.index_dir / f"doc_metadata_{index_id}.json"
        doc_metadata = {
            "N": self.N,
            "doc_titles": self.doc_titles,
            "doc_lengths": self.doc_lengths,
        }

        safe_write_json(doc_metadata, doc_meta_path)
        print(f"[META] Global document metadata saved → {doc_meta_path}")
