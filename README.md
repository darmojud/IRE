
# Assignment - 1
This implementation explores how **Information Retrieval (IR) systems** work internally by building a custom Python-based search engine (`SelfIndex`) and comparing it with an **Elasticsearch-based baseline (`ESIndex`)**.  
The goal is to analyze indexing, querying, and performance trade-offs under different configurations.

---

## Directory Structure

```plaintext
📁 src
├── builder/
│   ├── __init__.py
│   ├── index_builder_lmdb.py
│   ├── index_builder_redis.py
│   ├── merger.py
│   └── shard_builder.py
│
├── configs/
│   ├── config_es.json
│   ├── config_i.json
│   ├── config_q1.json
│   ├── config_x.json
│   ├── config_xy.json
│   ├── config_xy1.json
│   ├── config_xy2.json
│   └── config_xz.json
│
├── core/
│   ├── __init__.py
│   ├── compressor.py
│   ├── index_base.py
│   ├── preprocessor.py
│   └── utils.py
│
├── models/
│   ├── __init__.py
│   ├── ESIndex.py
│   └── SelfIndex_v2.py
│
├── query/
│   ├── __init__.py
│   ├── Parser.py
│   ├── query_lmdb.py
│   ├── query_processor.py
│   └── query_redis.py
│
├── index_helper.py
├── query_helper.py
├── index_metrics.json
├── query_latency_log.csv
└── wikiindex_analysis.ipynb  <<<=== The main notebook where the analysis begins
```
---

# Running the Index Builder (index_helper.py)

This script builds the index using the provided configuration.

Command:
```plain text
python3 index_helper.py <num_docs> <num_processes> <batch_size> <configfile>
```
Example:
```plain text
python3 index_helper.py 10000 10 2000  <configfile>
```
---

# Running the Query Engine (query_helper.py)

After building the index, you can run queries using the query helper script.

Command:
```plain text
python3 query_helper.py <configfile>
```

Example:
```plain text
python3 query_helper.py configs/config_x.json
```
