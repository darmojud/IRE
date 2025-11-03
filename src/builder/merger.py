import base64
import json
import lz4
import math
import os
from core.compressor import PostingCompressor


class Merger:

    def __init__(
        self,
        index_dir,
        compr,
        optim,
        info,
        num_shards: int = 8,
    ):
        self.index_dir = index_dir
        self.num_shards = num_shards
        self.N = 0
        self.doc_lengths = {}
        self.doc_titles = {}
        self.compr = compr
        if self.compr == "CODE":
            self.compressor = PostingCompressor()
        self.optim = optim
        self.info = info

    def merge_shards(self, batch_ids: list[str]) -> None:

        self.merge_metadata(batch_ids)
        if not batch_ids:
            print("[-SKIP-] No batch IDs provided to merge.")
            return

        metadata = self.load_metadata()
        merged_batches = set(metadata.get("merged_batches", []))
        unmerged_batches = [bid for bid in batch_ids if bid not in merged_batches]

        if not unmerged_batches:
            print(f"[-SKIP-] All batches {batch_ids} already merged.")
            return

        print(f"[MERGE] Starting merge for batches: {unmerged_batches}")

        for sid in range(self.num_shards):
            merged = self.merge_shard_for_batches(sid, unmerged_batches)
            self.write_cumulative_shard(sid, merged)

        # Update merged batches metadata
        self.update_merged_batches_metadata(metadata, unmerged_batches)

        try:
            self.merge_metadata(unmerged_batches)
            print(f"[META] doc_metadata.json merged for batches: {unmerged_batches}")
        except Exception as e:
            print(f"[META][ERROR] Failed to merge doc_metadata: {e}")

        # Cleanup individual batch shard files
        self.cleanup_batch_shard_files(unmerged_batches)

    def load_metadata(self) -> dict:
        metadata_path = self.index_dir / "batch_metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"[WARN] Failed to read metadata: {e}")
                return {}
        return {}

    def compress_pos(self, positions: list[int]) -> str:
        if self.compr == "CODE" and self.compressor:
            compressed_bytes = self.compressor.compress(positions)
            return base64.b64encode(compressed_bytes).decode("ascii")
        elif self.compr == "CLIB":
            compressed_bytes = lz4.frame.compress(json.dumps(positions).encode("utf-8"))
            return base64.b64encode(compressed_bytes).decode("ascii")
        else:
            return positions  # raw positions if no compression

    def decompress_pos(self, stored) -> list[int]:
        if self.compr == "CODE" and self.compressor:
            compressed_bytes = base64.b64decode(stored)
            return self.compressor.decompress(compressed_bytes)
        elif self.compr == "CLIB":
            compressed_bytes = base64.b64decode(stored)
            return json.loads(lz4.frame.decompress(compressed_bytes))
        else:
            return stored  # raw positions

    def merge_shard_for_batches(self, sid: int, batch_ids: list[str]) -> dict:
        cumulative_path = self.index_dir / f"cumulative_shard_{sid}.json"
        merged: dict[str, dict[str, dict]] = {}

        # Load existing cumulative shard
        if cumulative_path.exists():
            try:
                with open(cumulative_path, "r", encoding="utf-8") as fh:
                    old = json.load(fh)
                for term, postings in old.items():
                    merged[term] = {}
                    for doc_id, data in postings.items():
                        positions = self.decompress_pos(data.get("positions", []))
                        merged[term][doc_id] = {"positions": positions}
            except json.JSONDecodeError:
                print(f"[WARN] Corrupted {cumulative_path}, rebuilding from scratch.")

        # Merge new batch shards
        for batch_id in batch_ids:
            batch_path = self.index_dir / f"index_shard_{sid}_batch_{batch_id}.json"
            if not batch_path.exists():
                continue
            try:
                with open(batch_path, "r", encoding="utf-8") as fh:
                    new = json.load(fh)
                for term, postings in new.items():
                    merged.setdefault(term, {})
                    for doc_id, data in postings.items():
                        entry = merged[term].setdefault(doc_id, {"positions": []})
                        entry["positions"].extend(data.get("positions", []))
            except Exception as e:
                print(f"[ERROR] Reading {batch_path}: {e}")

        # Deduplicate positions & compute TF/TF-IDF if needed
        N = getattr(self, "N", 1)

        for term, postings in merged.items():
            df = len(postings)
            idf = math.log(N / df, 10) if df > 0 else 0

            # skip pointers generation
            if self.optim == "Skipping":
                sorted_doc_ids = sorted(map(int, postings.keys()))
                skip_interval = (
                    int(math.sqrt(len(sorted_doc_ids)))
                    if len(sorted_doc_ids) > 1
                    else 0
                )
            else:
                sorted_doc_ids = []
                skip_interval = 0

            # fill informantion
            for idx, doc_id in enumerate(
                sorted_doc_ids if self.optim == "Skipping" else postings.keys()
            ):
                doc_id_str = str(doc_id)
                info = postings[doc_id_str]
                positions = sorted(set(info.get("positions", [])))
                info["positions"] = self.compress_pos(positions)

                if (
                    self.optim == "Skipping"
                    and skip_interval
                    and (idx + skip_interval) < len(sorted_doc_ids)
                ):
                    info["skip_to"] = sorted_doc_ids[idx + skip_interval]

                if self.info.upper() == "TFIDF":
                    doc_len = self.doc_lengths.get(doc_id_str, 1)
                    tf = len(positions) / doc_len
                    tfidf = tf * idf
                    info["tf"] = round(tf, 6)
                    info["idf"] = round(idf, 6)
                    info["tfidf"] = round(tfidf, 6)

        return merged

    def write_cumulative_shard(self, sid: int, merged: dict):
        cumulative_path = self.index_dir / f"cumulative_shard_{sid}.json"
        try:
            with open(cumulative_path, "w", encoding="utf-8") as fh:
                json.dump(merged, fh, ensure_ascii=False, indent=4)
            print(f"[MERGED] cumulative_shard_{sid}.json (terms={len(merged)})")
        except Exception as e:
            print(f"[ERROR] Writing {cumulative_path}: {e}")

    def update_merged_batches_metadata(
        self, metadata: dict, unmerged_batches: list[str]
    ):
        merged_batches = set(metadata.get("merged_batches", []))
        merged_batches.update(unmerged_batches)
        metadata["merged_batches"] = sorted(list(merged_batches))

        metadata_path = self.index_dir / "batch_metadata.json"
        try:
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
            print(f"[DONE] Updated metadata → {metadata_path}")
        except Exception as e:
            print(f"[ERROR] Writing merged batches metadata: {e}")

    def cleanup_batch_shard_files(self, batch_ids: list[str]):
        for bid in batch_ids:
            for sid in range(self.num_shards):
                batch_path = self.index_dir / f"index_shard_{sid}_batch_{bid}.json"
                if batch_path.exists():
                    try:
                        os.remove(batch_path)
                        print(f"[CLEANUP] Removed metadata for merged batch {bid}.")
                    except Exception as e:
                        print(f"[WARN] Failed to remove metadata for batch {bid}: {e}")

    def merge_metadata(self, batch_ids: list[str]) -> None:
        merged_doc_lengths = {}
        total_docs = 0

        merged_metadata_path = self.index_dir / "doc_metadata.json"
        if merged_metadata_path.exists():
            with open(merged_metadata_path, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
            merged_doc_lengths = existing_data.get("doc_lengths", {})
            self.doc_titles.update(existing_data.get("doc_titles", {}))
            total_docs = existing_data.get("N", 0)
            print(
                f"[LOAD] Existing doc_metadata.json loaded with {len(merged_doc_lengths)} docs."
            )
        else:
            print("[INFO] No existing metadata file found; creating a new one.")

        valid_files_found = False

        for bid in batch_ids:
            path = self.index_dir / f"doc_metadata_{bid}.json"
            print(f"[MERGE] Loading metadata from {path}")
            if not path.exists():
                print(f"[WARN] Metadata file {path} does not exist; skipping.")
                continue

            valid_files_found = True
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Merge titles
            if "doc_titles" in data:
                self.doc_titles.update(data["doc_titles"])

            # Merge doc_lengths
            if "doc_lengths" in data:
                merged_doc_lengths.update(data["doc_lengths"])

            # Increment total doc count
            if "N" in data:
                total_docs += data["N"]

        if not valid_files_found:
            print("[-SKIP-] No valid metadata files found to merge.")
            return

        self.doc_lengths = merged_doc_lengths
        self.N = total_docs or len(merged_doc_lengths)

        merged_metadata = {
            "doc_titles": self.doc_titles,
            "doc_lengths": merged_doc_lengths,
            "N": self.N,
        }

        with open(merged_metadata_path, "w", encoding="utf-8") as f:
            json.dump(merged_metadata, f, ensure_ascii=False, indent=4)

        print(
            f"[MERGE] Updated doc_metadata.json with {len(merged_doc_lengths)} documents "
            f"(total N={self.N})."
        )

        for bid in batch_ids:
            temp_path = self.index_dir / f"doc_metadata_{bid}.json"
            try:
                os.remove(temp_path)
                print(f"[CLEANUP] Removed temporary metadata file: {temp_path}")
            except Exception as e:
                print(
                    f"[WARN] Failed to remove temporary metadata file {temp_path}: {e}"
                )
