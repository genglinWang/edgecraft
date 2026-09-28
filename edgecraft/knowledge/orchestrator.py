"""Knowledge Orchestrator for EdgeCraft.

This module provides a unified interface for retrieving knowledge from both:
- RAG (Retrieval-Augmented Generation): Web, arXiv, Kaggle real-time search
- CBR (Case-Based Reasoning): Past edge AI cases from the case store

The orchestrator combines both sources to provide comprehensive knowledge
for the planning phase of model synthesis.
"""
from typing import Dict, Any, Optional
from dataclasses import dataclass
from loguru import logger

from edgecraft.core.task import UserSpec
from edgecraft.knowledge.rag import MultiSourceRetriever
from edgecraft.knowledge.cbr import CaseRetriever, Case


@dataclass
class RetrievedKnowledge:
    """Container for retrieved knowledge from all sources."""

    rag_knowledge: str
    """Summarized knowledge from RAG (web, papers, Kaggle)."""

    similar_case: Optional[Case]
    """Best matching case from CBR, or None if no cases found."""

    combined_prompt: str
    """Combined knowledge formatted for injection into planner prompt."""

    def has_knowledge(self) -> bool:
        """Check if any knowledge was retrieved."""
        return bool(self.rag_knowledge) or self.similar_case is not None


class KnowledgeOrchestrator:
    """Unified orchestrator for RAG and CBR knowledge retrieval."""

    def __init__(
        self,
        rag_retriever: MultiSourceRetriever = None,
        cbr_retriever: CaseRetriever = None
    ):
        """Initialize the orchestrator.

        Args:
            rag_retriever: Optional RAG retriever instance.
            cbr_retriever: Optional CBR retriever instance.
        """
        self.rag_retriever = rag_retriever or MultiSourceRetriever()
        self.cbr_retriever = cbr_retriever or CaseRetriever()

    def retrieve_knowledge(
        self,
        user_spec: UserSpec,
        target_device: str = "",
        enable_rag: bool = True,
        enable_cbr: bool = True,
        cbr_candidates: int = 5
    ) -> RetrievedKnowledge:
        """Retrieve knowledge from all enabled sources.

        Args:
            user_spec: User intent specification with task details.
            target_device: Target deployment device.
            enable_rag: Whether to retrieve from RAG sources.
            enable_cbr: Whether to retrieve from CBR case store.
            cbr_candidates: Number of CBR candidates to consider before reranking.

        Returns:
            RetrievedKnowledge containing all retrieved information.
        """
        rag_knowledge = ""
        similar_case = None

        # Retrieve from RAG
        if enable_rag:
            logger.debug("Retrieving knowledge from RAG sources...")
            try:
                rag_knowledge = self.rag_retriever.retrieve_all(
                    user_spec=user_spec,
                    target_device=target_device
                )
                if rag_knowledge:
                    logger.debug(f"RAG retrieved {len(rag_knowledge)} chars of knowledge")
            except Exception as e:
                logger.warning(f"RAG retrieval failed: {e}")

        # Retrieve from CBR
        if enable_cbr:
            logger.debug("Retrieving similar cases from CBR...")
            try:
                similar_case = self.cbr_retriever.find_best_case(
                    user_spec=user_spec,
                    target_device=target_device,
                    n_candidates=cbr_candidates
                )
                if similar_case:
                    logger.debug(f"CBR found best case: {similar_case.id}")
            except Exception as e:
                logger.warning(f"CBR retrieval failed: {e}")

        # Build combined prompt
        combined_prompt = self._build_combined_prompt(
            rag_knowledge=rag_knowledge,
            similar_case=similar_case
        )

        return RetrievedKnowledge(
            rag_knowledge=rag_knowledge,
            similar_case=similar_case,
            combined_prompt=combined_prompt
        )

    def _build_combined_prompt(
        self,
        rag_knowledge: str,
        similar_case: Optional[Case]
    ) -> str:
        """Build a combined prompt section from all retrieved knowledge.

        Args:
            rag_knowledge: Knowledge from RAG sources.
            similar_case: Best matching case from CBR.

        Returns:
            Formatted prompt section for injection into planner.
        """
        sections = []

        if rag_knowledge:
            sections.append(f"""## Retrieved Knowledge (from Web, Papers, Kaggle)
{rag_knowledge}
""")

        if similar_case:
            sections.append(f"""## Similar Past Case (from Case Store)
{similar_case.to_prompt_text()}

**Recommendation**: Consider using a similar approach to the past case above.
The model '{similar_case.model_name}' achieved {self._format_metrics(similar_case.metrics)} on a similar task.
""")

        if not sections:
            return """## Retrieved Knowledge
No relevant knowledge or past cases found. Proceeding with default recommendations.
"""

        return "\n".join(sections)

    def _format_metrics(self, metrics: Dict[str, float]) -> str:
        """Format metrics dict into readable string."""
        if not metrics:
            return "metrics not recorded"

        parts = []
        for k, v in metrics.items():
            if isinstance(v, float):
                parts.append(f"{k}={v:.3f}")
            else:
                parts.append(f"{k}={v}")

        return ", ".join(parts)

    def get_case_store_status(self) -> Dict[str, Any]:
        """Get status of the case store.

        Returns:
            Dict with case store statistics.
        """
        return {
            "total_cases": self.cbr_retriever.case_store.count(),
            "by_source": self.cbr_retriever.case_store.count_by_source()
        }


def get_knowledge_for_planning(
    user_spec: UserSpec,
    target_device: str = "",
    enable_rag: bool = True,
    enable_cbr: bool = True
) -> RetrievedKnowledge:
    """Convenience function to retrieve knowledge for planning.

    Args:
        user_spec: User intent specification (parsed task/constraints).
        target_device: Target device.
        enable_rag: Whether to use RAG.
        enable_cbr: Whether to use CBR.

    Returns:
        Retrieved knowledge.
    """
    orchestrator = KnowledgeOrchestrator()
    return orchestrator.retrieve_knowledge(
        user_spec=user_spec,
        target_device=target_device,
        enable_rag=enable_rag,
        enable_cbr=enable_cbr
    )
