import warnings

warnings.filterwarnings("ignore")


from datasets import load_from_disk
import multiprocessing as mp
import sys
import time
import psutil
import json
from pathlib import Path
from typing import List
from models.SelfIndex_v2 import SelfIndex
from models.ESIndex import ESIndex
import os

sys.path.append(os.path.abspath(".."))

# ---------------- CONFIG ----------------
DATA_PATH = "wikidata/train"
# BATCH_SIZE = 2000
# MAX_PROCESSES = 10
# MAX_DOCS = 5000  # cap for demo runs
METRICS_FILE = "index_metrics.json"  # output file


# ---------------- Helper Functions ----------------
def get_memory_usage_mb():
    process = psutil.Process()
    mem_info = process.memory_info()
    return mem_info.rss / (1024 * 1024)


def calculate_index_size(index_dir: Path) -> float:
    total_size = 0
    for path in index_dir.glob("**/*"):
        if path.is_file():
            total_size += path.stat().st_size
    return total_size / (1024 * 1024)


def load_dataset_in_parts(data_path, start, end):
    dataset = load_from_disk(data_path)
    return dataset.select(range(start, end))


def calculate_es_index_size(index_name="wiki_index"):
    es = ESIndex()
    stats = es.es.indices.stats(index=index_name, metric="store")
    size_bytes = stats["_all"]["total"]["store"]["size_in_bytes"]
    return round(size_bytes / (1024 * 1024), 2)


# ---------------- Worker ----------------
def worker_process(
    indexing: str,
    batch_id: int,
    start_idx: int,
    dataset_slice,
    index_dir: str,
    info_type: str,
    dstore: str,
    compr_type: str,
    optim_type: str,
):
    if indexing == "self":
        idx = SelfIndex(
            core="SelfIndex",
            info=info_type,
            dstore=dstore,
            qproc="TERMatat",
            compr=compr_type,
            optim=optim_type,
        )
    elif indexing == "es":
        idx = ESIndex(
            core="ESIndex",
            info=info_type,
            dstore=dstore,
            qproc="TERMatat",
            compr=compr_type,
            optim=optim_type,
        )
    else:
        print("ERROR")
        return
    files = []
    for i, example in enumerate(dataset_slice):
        doc_id = start_idx + i
        text = example.get("text", "") if isinstance(example, dict) else str(example)
        title = example.get("title", "") if isinstance(example, dict) else ""
        files.append((doc_id, title, text))

    start_create = time.time()
    idx.create_index(str(batch_id), files)
    end_create = time.time()
    print(f"[WORKER] Batch {batch_id} created in {end_create - start_create:.2f}s")


# ---------------- Main controller ----------------
def run_indexing(
    max_docs: int,
    max_processes: int,
    batch_size: int,
    indexing: str,
    info_type: str,
    dstore_type: str = "CUSTOM",
    compr_type: str = "NONE",
    optim_type: str = "Null",
):

    print(f"\n{'='*60}")
    print(f"Starting Indexing Mode: {indexing} {info_type}")
    print(f"{'='*60}")

    start_mem = get_memory_usage_mb()
    start_time = time.time()

    dataset = load_from_disk(DATA_PATH)
    total_docs = len(dataset)
    # del dataset
    cap = max_docs if max_docs else total_docs
    total_docs = min(total_docs, cap)
    print(f"Total documents to index: {total_docs}")

    batch_start = 0
    batch_id = 0

    if indexing == "self":
        idx_mgr = SelfIndex(
            core="SelfIndex",
            info=info_type,
            dstore=dstore_type,
            qproc="TERMatat",
            compr=compr_type,
            optim=optim_type,
        )
    elif indexing == "es":
        idx_mgr = ESIndex(
            core="ESIndex",
            info=info_type,
            dstore=dstore_type,
            qproc="TERMatat",
            compr=compr_type,
            optim=optim_type,
        )
    else:
        print("ERROR")
        return

    if indexing == "self":

        index_dir = Path(idx_mgr.index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
    else:
        index_dir = "-"

    i = 0
    while batch_start < total_docs:
        group_end = min(batch_start + (batch_size * max_processes), total_docs)
        processes = []
        group_batch_ids: List[str] = []

        print(
            f"----------- Iteration {i} ({batch_start}:{group_end}) -----------------"
        )

        for _ in range(max_processes):
            start_idx = batch_start + len(group_batch_ids) * batch_size
            if start_idx >= total_docs:
                break
            end_idx = min(start_idx + batch_size, total_docs)

            # dataset_slice = load_dataset_in_parts(DATA_PATH, start_idx, end_idx)
            dataset_slice = dataset.select(range(start_idx, end_idx))
            this_batch_id = batch_id

            p = mp.Process(
                target=worker_process,
                args=(
                    indexing,
                    this_batch_id,
                    start_idx,
                    dataset_slice,
                    index_dir,
                    info_type,
                    dstore_type,
                    compr_type,
                    optim_type,
                ),
            )
            p.start()
            processes.append(p)
            group_batch_ids.append(str(this_batch_id))
            print(f"[START] Batch {this_batch_id} ({start_idx}:{end_idx})")
            batch_id += 1

        for p in processes:
            p.join()

        # print(f"[MERGE] Merging metadata for batches: {group_batch_ids}")
        # idx_mgr.merge_metadata(group_batch_ids)

        if indexing == "self":
            print("[MERGE]  Merging shards for batches:", group_batch_ids)
            idx_mgr.merge_index(group_batch_ids)

        batch_start = group_end
        i += 1

    end_time = time.time()
    end_mem = get_memory_usage_mb()
    if indexing == "self":
        index_size_mb = calculate_index_size(Path(index_dir))
    elif indexing == "es":
        index_size_mb = calculate_es_index_size(index_name="wiki_index")

    metrics = {
        "index_type": indexing,
        "info_type": info_type,
        "total_docs": total_docs,
        "batch_size": batch_size,
        "max_processes": max_processes,
        "index_directory": str(index_dir),
        "total_time_sec": round(end_time - start_time, 2),
        "memory_used_mb": round(end_mem - start_mem, 2),
        "index_size_mb": round(index_size_mb, 2),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    print(f"[DONE] {info_type} indexing complete.")
    print(json.dumps(metrics, indent=4))

    # Append metrics to JSON file
    existing_metrics = []
    if Path(METRICS_FILE).exists():
        with open(METRICS_FILE, "r") as f:
            try:
                existing_metrics = json.load(f)
            except json.JSONDecodeError:
                existing_metrics = []

    existing_metrics.append(metrics)

    with open(METRICS_FILE, "w") as f:
        json.dump(existing_metrics, f, indent=4)

    print(f"[LOG] Metrics written to {METRICS_FILE}\n")


# ---------------- Run All Configurations ----------------
if __name__ == "__main__":
    mp.set_start_method("spawn")

    max_docs = int(sys.argv[1])
    max_processes = int(sys.argv[2])
    batch_size = int(sys.argv[3])
    config_path = sys.argv[4]

    # index_type = sys.argv[4]

    # info=sys.argv[5],
    # dstore=sys.argv[6],
    # qproc=sys.argv[7],
    # compr=sys.argv[8],
    # optim=sys.argv,

    # configs = [
    #     ("BOOLEAN", "NONE"),
    #     ("WORDCOUNT", "NONE"),
    #     ("TFIDF", "NONE"),
    #     ("BOOLEAN", "CODE"),
    #     ("WORDCOUNT", "CODE"),
    #     ("TFIDF", "CODE"),
    #     ("BOOLEAN", "CLIB"),
    #     ("WORDCOUNT", "CLIB"),
    #     ("TFIDF", "CLIB"),
    # ]

    # for info_type, index_dir in configs:
    #     run_indexing(info_type, index_dir)

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    for cfg in config:
        try:
            index = cfg.get("index_type", "self")
            info_type = cfg.get("info", "BOOLEAN")
            dstore_type = cfg.get("dstore", "CUSTOM")
            compr_type = cfg.get("compr", "NONE")
            optim_type = cfg.get("optim", "Null")
            index_type = cfg.get("index_type", "self")

            print(f"Running config: {cfg}")

            run_indexing(
                max_docs,
                max_processes,
                batch_size,
                indexing=index,
                info_type=info_type,
                dstore_type=dstore_type,
                compr_type=compr_type,
                optim_type=optim_type,
            )

        except Exception as e:
            print(f"Error while running config {cfg}: {e}")
            continue

    # run_indexing("BOOLEAN", "DB1", "CODE", "Skipping")
