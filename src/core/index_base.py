from abc import ABC, abstractmethod
from typing import Iterable
from pathlib import Path
from enum import Enum
import json


# Identifier enums for variants for index
# Tailor to add specifics that are implemented
class IndexInfo(Enum):
    BOOLEAN_NO_POS = 0
    BOOLEAN = 1
    WORDCOUNT = 2
    TFIDF = 3


class DataStore(Enum):
    CUSTOM = 1
    DB1 = 2
    DB2 = 3


class Compression(Enum):
    NONE = 1
    CODE = 2
    CLIB = 3


class QueryProc(Enum):
    TERMatat = "T"
    DOCatat = "D"


class Optimizations(Enum):
    Null = "0"
    Skipping = "sp"
    Thresholding = "th"
    EarlyStopping = "es"


class IndexBase(ABC):
    """
    Base index class with abstract methods to inherit for specific implementations.
    """

    def __init__(self, core, info, dstore, qproc, compr, optim):
        """
        Sample usage:
            idx = IndexBase(core='ESIndex', info='BOOLEAN', dstore='DB1', compr='NONE', qproc='TERMatat', optim='Null')
            print (idx)
        """
        assert core in ("ESIndex", "SelfIndex")
        long = [
            IndexInfo[info],
            DataStore[dstore],
            Compression[compr],
            QueryProc[qproc],
            Optimizations[optim],
        ]
        short = [k.value for k in long]
        self.identifier_long = (
            "core={}|index={}|datastore={}|compressor={}|qproc={}|optim={}".format(
                *[core] + long
            )
        )
        self.identifier_short = "{}_i{}d{}c{}q{}o{}".format(*[core] + short)

    def __repr__(self):
        return f"{self.identifier_short}: {self.identifier_long}"

    @abstractmethod
    def create_index(index_id: str, files: Iterable[tuple[str, str]]) -> None:
        """Creates and index for the given files
        Args:
            index_id: The unique identifier for the index.
            files: An iterable (list-like object) of tuples, where each tuple contains the file id and its content.
        """
        # DUMMY IMPLEMENTATION, only stores the index_id
        """
        with open(INDEX_STORAGE_PATH) as f:
            data: list[str] = json.load(f)
    
        data.append(index_id)
    
        with open(INDEX_STORAGE_PATH, "w") as f:
            json.dump(data, f)
        """
        pass

    @abstractmethod
    def load_index(serialized_index_dump: str) -> None:
        """Loads an already created index into memory from disk.
        Args:
            serialized_index_dump: Path to dump of serialized index
        """
        pass

    @abstractmethod
    def update_index(
        index_id: str,
        remove_files: Iterable[tuple[str, str]],
        add_files: Iterable[tuple[str, str]],
    ) -> None:
        """Updates an index. First removes files from the index, then adds files to the index.
        Args:
            index_id: The unique identifier for the index.
            remove_files: An iterable (list-like object) of tuples, where each tuple contains the file id and its content to be removed.
            add_files: An iterable (list-like object) of tuples, where each tuple contains the file id and its content to be added.
        """
        pass

    @abstractmethod
    def query(query: str) -> str:
        """Queries the already loaded index to generate a results json and return as str
        Args:
            query: Input query in str format
        Returns:
            results: Output json str with results
        """
        pass

    @abstractmethod
    def delete_index(index_id: str) -> None:
        """Deletes the index with the given index_id."""
        # Remove index files from disk
        pass

    @abstractmethod
    def list_indices() -> Iterable[str]:
        """Lists all indices.

        Returns:
            An iterable (list) of index ids.
        """
        pass

    @abstractmethod
    def list_indexed_files(index_id: str) -> Iterable[str]:
        """Lists all files indexed in the given index.

        Returns:
            An iterable (list-like object) of file ids.
        """
        # DUMMY IMPLEMENTATION, only returns a fixed set of paths
        return ["documents/example.txt", "documents/example2.txt"]
