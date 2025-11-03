from models.SelfIndex_v2 import SelfIndex
import json
from builder.index_builder_redis import IndexBuilderRedis
from models.SelfIndex_v2 import SelfIndex
from models.ESIndex import ESIndex

import time
from datetime import datetime
import numpy as np
import os
import csv
import sys
import tracemalloc


def print_results(response_json):

    response = json.loads(response_json)

    print("\n--- Query Summary ---")
    print(f"Query: {response['query']}")
    print(f"Terms: {response['terms']}")
    print(f"Negated Terms: {response['negated_terms']}")
    print(f"Total Matched Docs: {response['matched_docs']}")
    print(f"Ranked Results Count: {response['ranked_results']}")

    if response["ranked_results"]:
        print("\n--- Top Ranked Results ---")
        print(f"{'Doc ID':<10} {'Score':<10} {'Title'}")
        print("-" * 60)
        for doc in response["top_n_ranked_results"]:
            print(f"{doc['doc_id']:<10} {doc['score']:<10.6f} {doc['title']}")
    else:
        print("No results found.")


def pretty_response(response):
    if len(response["hits"]["hits"]) == 0:
        print("Your search returned no results.")
    else:
        for hit in response["hits"]["hits"]:
            id = hit["_id"]
            # publication_date = hit["_source"]["publish_date"]
            score = hit["_score"]
            title = hit["_source"]["title"]
            pretty_output = f"\nID: {id}\nTitle: {title}\nScore: {score}"
            print(pretty_output)


def measure_latency(index_type, index, query_str, runs=1):
    latencies = []
    for _ in range(runs):
        tracemalloc.start()
        start = time.perf_counter()
        response = index.query(query_str)
        end = time.perf_counter()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        if _ == 0:
            if index_type == "self":
                print_results(response)
            elif index_type == "es":
                pretty_response(response)
        latencies.append((end - start) * 1000)
    return latencies, current / 1024, peak / 1024


def percentile(values, p):
    if not values:
        return None
    return np.percentile(values, p)


def log_results_to_csv(rows, output_file="query_latency_log.csv"):
    file_exists = os.path.exists(output_file)
    with open(output_file, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(
                [
                    "timestamp",
                    "index_type",
                    "info",
                    "dstore",
                    "qproc",
                    "compr",
                    "optim",
                    "query",
                    "query_type",
                    "run_id",
                    "latency_ms",
                    "current_mem_kb",
                    "peak_mem_kb",
                ]
            )
        writer.writerows(rows)


if __name__ == "__main__":

    config_path = sys.argv[1]

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    for cfg in config:
        index_type = cfg.get("index_type", "self")
        info_type = cfg.get("info", "BOOLEAN")
        dstore_type = cfg.get("dstore", "CUSTOM")
        compr_type = cfg.get("compr", "NONE")
        optim_type = cfg.get("optim", "Null")
        qproc_type = cfg.get("qproc", "TERMatat")

        if index_type == "self":

            # Instantiate index
            index = SelfIndex(
                core="SelfIndex",
                info=info_type,
                dstore=dstore_type,
                qproc=qproc_type,
                compr=compr_type,
                optim=optim_type,
            )
        elif index_type == "es":
            index = ESIndex()

        test_queries = [
            # Single terms
            ("america", "term"),
            ("washington", "term"),
            ("constitution", "term"),
            # Basic boolean
            ("america AND washington", "boolean"),
            ("america OR washington", "boolean"),
            ("america AND NOT washington", "boolean"),
            ("NOT america", "boolean"),
            # Phrase queries
            ('"united states of america"', "phrase"),
            ('"american constitution"', "phrase"),
            ('"american independence"', "phrase"),
            # Mixed
            ('"united states of america" AND "american constitution"', "mixed"),
            ('"american independence" OR america', "mixed"),
            ('"united states of america" AND NOT constitution', "mixed"),
            ('"north america" OR "south america"', "mixed"),
            # Grouped
            ('("america" AND "washington") OR ("congress" AND "president")', "grouped"),
            ('("america" OR "washington") AND ("congress" OR "president")', "grouped"),
            (
                '("america" AND NOT "washington") OR ("washington" AND NOT "america")',
                "grouped",
            ),
            ('(america OR washington) AND NOT "congress"', "grouped"),
            # Nested
            (
                '(("america" AND "washington") OR "president") AND NOT "congress"',
                "nested",
            ),
            ('(("america" OR "washington") AND ("congress" OR "president"))', "nested"),
            ('("america" AND ("washington" OR "president"))', "nested"),
            (
                '("america" AND ("washington" AND ("congress" OR "president")))',
                "nested",
            ),
            # Edge / tricky
            ("america washington", "edge"),
            ('"united states of america" OR america', "edge"),
            ('"washington" AND NOT "washington"', "edge"),
            ("(america)", "edge"),
            ('NOT ("america" AND "washington")', "edge"),
        ]

        all_latencies = []
        log_rows = []
        for q, type in test_queries:
            print("\n" + "=" * 80)
            latencies, current, peak = measure_latency(index_type, index, q)
            all_latencies.extend(latencies)

            for run_id, latency in enumerate(latencies):
                log_rows.append(
                    [
                        datetime.utcnow().isoformat(),
                        index_type,
                        index.info if index_type == "self" else " ",
                        index.dstore if index_type == "self" else " ",
                        index.compr if index_type == "self" else " ",
                        index.qproc if index_type == "self" else " ",
                        index.optim if index_type == "self" else " ",
                        q,
                        type,
                        run_id,
                        round(latency, 3),
                        round(current, 2),
                        round(peak, 2),
                    ]
                )
                print(
                    f"Avg: {np.mean(latencies):.2f} ms | p95: {percentile(latencies, 95):.2f} | p99: {percentile(latencies, 99):.2f}"
                )

        log_results_to_csv(log_rows)
