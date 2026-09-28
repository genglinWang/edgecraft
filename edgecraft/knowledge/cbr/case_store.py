"""Case store for Case-Based Reasoning."""
import json
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime
from pathlib import Path

from edgecraft.core.modality import Modality, TaskType
from edgecraft.knowledge.rag.vectorstore import VectorStore
from edgecraft.config.settings import settings
from edgecraft.knowledge.cbr.scope import case_store_namespace


class Case(BaseModel):
    """A case representing a past edge AI synthesis task and its solution."""

    id: str = Field(..., description="Unique case identifier")

    # Problem description
    intent: str = Field(..., description="User's natural language intent")
    modality: Modality
    task_type: TaskType
    dataset_description: str = Field(default="", description="Description of the dataset")
    dataset_signature: Dict[str, Any] = Field(default_factory=dict)
    layout_contract: str = Field(default="")
    failure_signature: str = Field(default="")
    constraints: Dict[str, Any] = Field(default_factory=dict)
    target_device: str = Field(default="")

    # Solution
    model_name: str = Field(default="", description="Selected model (e.g., yolo11n, mobilenetv3)")
    model_source: str = Field(default="", description="Model source (ultralytics/timm/huggingface)")
    training_config: Dict[str, Any] = Field(default_factory=dict)
    optimization_config: Dict[str, Any] = Field(default_factory=dict)

    # Outcome
    metrics: Dict[str, float] = Field(default_factory=dict, description="Final performance metrics")
    success: bool = Field(default=True)
    lessons_learned: str = Field(default="", description="What was learned from this case")

    # Provenance metadata
    source: str = Field(default="edgecraft", description="Case source: kaggle/github/jetson/edgecraft")
    source_url: str = Field(default="", description="URL to the original source")
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())

    def to_embedding_text(self) -> str:
        """Convert case to text for embedding."""
        return f"""
Intent: {self.intent}
Modality: {self.modality.value}
Task: {self.task_type.value}
Device: {self.target_device}
Constraints: {json.dumps(self.constraints)}
Model: {self.model_name}
Model Source: {self.model_source}
Dataset Signature: {json.dumps(self.dataset_signature)}
Layout Contract: {self.layout_contract}
Failure Signature: {self.failure_signature}
Optimization: {json.dumps(self.optimization_config)}
Metrics: {json.dumps(self.metrics)}
Success: {self.success}
Lessons: {self.lessons_learned}
""".strip()

    def to_prompt_text(self) -> str:
        """Convert case to a readable prompt format for LLM context."""
        constraints_str = ", ".join(f"{k}={v}" for k, v in self.constraints.items()) if self.constraints else "None"
        metrics_str = ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in self.metrics.items()) if self.metrics else "None"
        opt_str = ", ".join(f"{k}={v}" for k, v in self.optimization_config.items()) if self.optimization_config else "None"

        return f"""**Past Case: {self.id}**
- Task: {self.modality.value} {self.task_type.value}
- Device: {self.target_device or 'Not specified'}
- Constraints: {constraints_str}
- Solution: {self.model_name} (from {self.model_source or 'unknown'})
- Layout Contract: {self.layout_contract or 'unknown'}
- Failure Signature: {self.failure_signature or 'none'}
- Optimization: {opt_str}
- Achieved Metrics: {metrics_str}
- Success: {'Yes' if self.success else 'No'}
- Lessons: {self.lessons_learned or 'None recorded'}
"""


class CaseStore:
    """Tenant-private CBR cases using ChromaDB with JSON persistence."""

    def __init__(
        self,
        collection_name: str = "edgecraft_cases",
        *,
        tenant_id: str = "default",
    ):
        """Initialize the case store.

        Args:
            collection_name: Base name of the ChromaDB collection.
            tenant_id: Owning tenant. The raw value is never used as a path.
        """
        self.tenant_id = str(tenant_id or "default")
        self.collection_name = case_store_namespace(
            self.tenant_id,
            collection_name=collection_name,
        )
        self.vectorstore = VectorStore(collection_name=self.collection_name)
        self.read_only = bool(settings.CBR_READ_ONLY)
        self._cases: Dict[str, Case] = {}
        configured_path = str(settings.CBR_CASE_PATH or "").strip()
        configured = Path(configured_path).expanduser().resolve() if configured_path else None
        self._json_path = (
            configured.with_name(
                f"{configured.stem}_{self.collection_name}{configured.suffix or '.json'}"
            )
            if configured is not None
            else Path(settings.CHROMA_PERSIST_DIRECTORY) / f"{self.collection_name}.json"
        )
        self._load_cases_from_json()

    def _load_cases_from_json(self) -> None:
        """Load cases from JSON file for full object access."""
        if self._json_path.exists():
            try:
                with open(self._json_path, "r") as f:
                    data = json.load(f)
                for case_data in data:
                    case = Case(**case_data)
                    self._cases[case.id] = case
            except (json.JSONDecodeError, Exception) as e:
                pass  # Start fresh if JSON is corrupted

    def _save_cases_to_json(self) -> None:
        """Persist cases to JSON file."""
        self._json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._json_path, "w") as f:
            json.dump([c.model_dump(mode="json") for c in self._cases.values()], f, indent=2)

    def add_case(self, case: Case) -> None:
        """Add a case to the store.

        Args:
            case: The case to add.
        """
        if self.read_only:
            return
        # Store in vector store for similarity search
        self.vectorstore.add_documents(
            documents=[case.to_embedding_text()],
            metadatas=[{
                "modality": case.modality.value,
                "task_type": case.task_type.value,
                "target_device": case.target_device,
                "success": str(case.success),
                "model_name": case.model_name,
                "source": case.source,
                "layout_contract": case.layout_contract,
                "failure_signature": case.failure_signature,
            }],
            ids=[case.id]
        )

        # Keep in memory and persist
        self._cases[case.id] = case
        self._save_cases_to_json()

    def add_cases_batch(self, cases: List[Case]) -> int:
        """Add multiple cases in batch.

        Args:
            cases: List of cases to add.

        Returns:
            Number of cases added.
        """
        if not cases:
            return 0
        if self.read_only:
            return 0

        documents = [c.to_embedding_text() for c in cases]
        metadatas = [{
            "modality": c.modality.value,
            "task_type": c.task_type.value,
            "target_device": c.target_device,
            "success": str(c.success),
            "model_name": c.model_name,
            "source": c.source,
            "layout_contract": c.layout_contract,
            "failure_signature": c.failure_signature,
        } for c in cases]
        ids = [c.id for c in cases]

        self.vectorstore.add_documents(documents=documents, metadatas=metadatas, ids=ids)

        for case in cases:
            self._cases[case.id] = case
        self._save_cases_to_json()

        return len(cases)

    def get_case(self, case_id: str) -> Optional[Case]:
        """Get a case by ID."""
        return self._cases.get(case_id)

    def search_similar(
        self,
        query: str,
        n_results: int = 3,
        modality: Modality = None,
        task_type: TaskType = None,
        source: str = None
    ) -> List[Case]:
        """Search for similar cases.

        Args:
            query: Query text (e.g., user intent).
            n_results: Number of results.
            modality: Optional filter by modality.
            task_type: Optional filter by task type.
            source: Optional filter by source (kaggle/github/jetson/edgecraft).

        Returns:
            List of similar cases.
        """
        # Build filter
        where_conditions = []
        if modality:
            where_conditions.append({"modality": modality.value})
        if task_type:
            where_conditions.append({"task_type": task_type.value})
        if source:
            where_conditions.append({"source": source})

        where = None
        if len(where_conditions) == 1:
            where = where_conditions[0]
        elif len(where_conditions) > 1:
            where = {"$and": where_conditions}

        results = self.vectorstore.query(
            query_text=query,
            n_results=n_results,
            where=where
        )

        # Return full case objects by matching IDs from metadata
        cases = []
        # ChromaDB returns IDs in results
        if "ids" in results and results["ids"]:
            for case_id in results["ids"]:
                if case_id in self._cases:
                    cases.append(self._cases[case_id])
        else:
            # Fallback: match by document content
            for doc in results.get("documents", []):
                for case_id, case in self._cases.items():
                    if case.to_embedding_text() == doc and case not in cases:
                        cases.append(case)
                        break

        return cases

    def count(self) -> int:
        """Return number of cases."""
        return self.vectorstore.count()

    def count_by_source(self) -> Dict[str, int]:
        """Return case counts grouped by source."""
        counts = {}
        for case in self._cases.values():
            counts[case.source] = counts.get(case.source, 0) + 1
        return counts

    def list_cases(self, modality: Modality = None, source: str = None) -> List[Case]:
        """List all cases, optionally filtered by modality or source."""
        cases = list(self._cases.values())
        if modality:
            cases = [c for c in cases if c.modality == modality]
        if source:
            cases = [c for c in cases if c.source == source]
        return cases

    def delete_case(self, case_id: str) -> bool:
        """Delete a case by ID.

        Returns:
            True if deleted, False if not found.
        """
        if self.read_only or case_id not in self._cases:
            return False

        self.vectorstore.delete([case_id])
        del self._cases[case_id]
        self._save_cases_to_json()
        return True

    def clear(self) -> int:
        """Clear all cases from the store.

        Returns:
            Number of cases deleted.
        """
        if self.read_only:
            return 0
        count = len(self._cases)
        self.vectorstore.reset()
        self._cases.clear()
        self._save_cases_to_json()
        return count
