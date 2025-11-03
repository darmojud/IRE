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

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

import base64
import lz4.frame

# # --------------- Config ----------------
# INDEX_DIR = Path("./wiki_index")
# NUM_SHARDS = 8  # number of total shards

# INDEX_DIR.mkdir(parents=True, exist_ok=True)


# ---------------- IndexManager (concrete implementation) ----------------
class ESIndex(IndexBase):

    def __init__(
        self,
        core="ESIndex",
        info="BOOLEAN",
        dstore="CUSTOM",
        qproc="TERMatat",
        compr="NONE",
        optim="Null",
    ):
        super().__init__(core, info, dstore, qproc, compr, optim)

        self.es = Elasticsearch(hosts=["http://localhost:9200"])
        print(self.es.ping())
        print(self.es.info())
        self.preprocessor = Preprocessor()

    def create_index(
        self, index_id: str, files: Iterable[Tuple[str, str, str]]
    ) -> None:
        mappings = {
            "properties": {
                "id": {"type": "keyword"},
                "title": {"type": "text"},
                "text": {"type": "text"},
            }
        }

        if not self.es.indices.exists(index="wiki_index"):
            self.es.indices.create(index="wiki_index", mappings=mappings)
            print("Created index named wiki_index")
        else:
            print("Index wiki_index already exists, skipping creation")

        bulk_set = [
            {
                "_index": "wiki_index",
                "_id": doc_id,
                "_source": {
                    "id": doc_id,
                    "title": title,
                    "text": self.preprocessor.preprocess(text),
                },
            }
            for (doc_id, title, text) in files
        ]

        bulk(self.es, bulk_set)
        return

    def merge_index(self, batch_ids):
        pass

    def query(self, query_str: str) -> str:

        query = {"query_string": {"query": query_str, "fields": ["text"]}}

        return self.es.search(index="wiki_index", query=query)

    def load_index(self, serialized_index_dump: str) -> dict:
        pass

    def update_index(
        self,
        index_id: str,
        remove_files: Iterable[Tuple[str, str]],
        add_files: Iterable[Tuple[str, str]],
    ) -> None:

        if add_files:
            for i, (title, text) in enumerate(add_files):
                body = {
                    "doc": {
                        "title": title,
                        "text": text,
                    }
                }
                self.es.update(index="wiki_index", id=index_id, body=body)

        if remove_files:
            pass

    def delete_index(self, index_id: str) -> None:

        self.es.delete(index="wiki_index", id=1000)

    def delete_whole_index(self):
        self.es.indices.delete(index="wiki_index")

    def list_indices(self) -> Iterable[str]:
        pass

    def list_indexed_files(self, index_id: str) -> Iterable[str]:
        pass
