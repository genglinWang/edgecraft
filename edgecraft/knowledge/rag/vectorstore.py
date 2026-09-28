"""ChromaDB vector store wrapper for EdgeCraft."""
import os
import sys
import contextlib
from typing import List, Dict, Any, Optional

# Suppress ONNX Runtime TensorRT/CUDA provider warnings before importing chromadb
# ChromaDB uses ONNX Runtime for embeddings, which tries to load TensorRT by default
# The C++ warnings can only be suppressed by redirecting stderr during import
os.environ["ORT_LOGGING_LEVEL"] = "3"

try:
    import onnxruntime as ort
    ort.set_default_logger_severity(3)  # 3 = ERROR only
except ImportError:
    pass

# Suppress stderr during chromadb import to hide C++ level TensorRT warnings
@contextlib.contextmanager
def _suppress_stderr():
    """Temporarily redirect stderr to suppress C++ warnings."""
    devnull = open(os.devnull, 'w')
    old_stderr = sys.stderr
    sys.stderr = devnull
    try:
        yield
    finally:
        sys.stderr = old_stderr
        devnull.close()

with _suppress_stderr():
    import chromadb
    from chromadb.config import Settings as ChromaSettings

from edgecraft.config.settings import settings


class VectorStore:
    """Wrapper for ChromaDB vector store."""

    def __init__(
        self,
        collection_name: str = "edgecraft_knowledge",
        persist_directory: str = None
    ):
        """Initialize the vector store.

        Args:
            collection_name: Name of the ChromaDB collection.
            persist_directory: Directory to persist the database.
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory or settings.CHROMA_PERSIST_DIRECTORY

        # Ensure directory exists
        os.makedirs(self.persist_directory, exist_ok=True)

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=ChromaSettings(anonymized_telemetry=False)
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add_documents(
        self,
        documents: List[str],
        metadatas: List[Dict[str, Any]] = None,
        ids: List[str] = None
    ) -> None:
        """Add documents to the vector store.

        Args:
            documents: List of document texts.
            metadatas: List of metadata dicts for each document.
            ids: List of unique IDs for each document.
        """
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(documents))]

        if metadatas is None:
            metadatas = [{} for _ in documents]

        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

    def query(
        self,
        query_text: str,
        n_results: int = 5,
        where: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Query the vector store.

        Args:
            query_text: Query string.
            n_results: Number of results to return.
            where: Optional filter conditions.

        Returns:
            Dict with 'documents', 'metadatas', 'distances'.
        """
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"]
        )

        return {
            "ids": results["ids"][0] if results["ids"] else [],
            "documents": results["documents"][0] if results["documents"] else [],
            "metadatas": results["metadatas"][0] if results["metadatas"] else [],
            "distances": results["distances"][0] if results["distances"] else []
        }

    def delete(self, ids: List[str]) -> None:
        """Delete documents by ID."""
        self.collection.delete(ids=ids)

    def count(self) -> int:
        """Return the number of documents in the collection."""
        return self.collection.count()

    def reset(self) -> None:
        """Delete all documents in the collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
