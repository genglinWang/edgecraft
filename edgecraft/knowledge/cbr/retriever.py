"""Case retriever for CBR-based solution adaptation."""
import re
from typing import List, Dict, Any, Optional
from loguru import logger

from langchain_core.messages import HumanMessage, SystemMessage

from .case_store import CaseStore, Case
from edgecraft.core.modality import Modality, TaskType
from edgecraft.core.task import UserSpec
from edgecraft.utils.llm import create_chat_llm


RERANK_PROMPT = """You are an expert at evaluating the relevance of past edge AI cases to a new task.

Given the current task specification and {n_cases} candidate cases from past projects,
rank them by relevance, informativeness, and helpfulness for solving the new task.

## Current Task
- Description: {description}
- Task Type: {task_type}
- Modality: {modality}
- Target Device: {target_device}
- Constraints: {constraints}

## Candidate Cases
{cases_text}

## Instructions
Rank the cases from most to least relevant based on:
1. Similarity of task type and modality
2. Similarity of target device and constraints
3. Success of the past solution
4. Applicability of lessons learned

Output format: [1] > [2] > [3] > ... (using the case numbers above)
Only output the ranking, no explanations.
"""


class CaseRetriever:
    """Retriever for finding and adapting past cases with LLM reranking."""

    def __init__(self, case_store: CaseStore = None, llm = None):
        """Initialize the case retriever.

        Args:
            case_store: Optional existing case store.
            llm: Optional LLM for reranking.
        """
        self.case_store = case_store or CaseStore()

        self.llm = llm or create_chat_llm(temperature=0.1, purpose="cbr_rerank")

    def find_similar_cases(
        self,
        intent: str,
        modality: Modality = None,
        task_type: TaskType = None,
        n_results: int = 3
    ) -> List[Case]:
        """Find cases similar to the given intent.

        Args:
            intent: User's intent description.
            modality: Optional modality filter.
            task_type: Optional task type filter.
            n_results: Number of cases to return.

        Returns:
            List of similar cases.
        """
        return self.case_store.search_similar(
            query=intent,
            n_results=n_results,
            modality=modality,
            task_type=task_type
        )

    def rerank_cases(
        self,
        cases: List[Case],
        user_spec: UserSpec,
        target_device: str = ""
    ) -> Optional[Case]:
        """Rerank cases using LLM and return the best one.

        This implements the retrieve-then-rerank pattern from DS-Agent.

        Args:
            cases: List of candidate cases from vector search.
            user_spec: Current task specification.
            target_device: Target deployment device.

        Returns:
            The best matching case, or None if no cases provided.
        """
        if not cases:
            return None

        if len(cases) == 1:
            return cases[0]

        # Format cases for the prompt
        cases_text_parts = []
        for i, case in enumerate(cases, 1):
            cases_text_parts.append(f"""[{i}] **{case.id}**
- Intent: {case.intent}
- Task: {case.modality.value} / {case.task_type.value}
- Device: {case.target_device or 'Not specified'}
- Model: {case.model_name} (from {case.model_source or 'unknown'})
- Constraints: {case.constraints}
- Metrics: {case.metrics}
- Success: {'Yes' if case.success else 'No'}
- Lessons: {case.lessons_learned or 'None'}
""")

        cases_text = "\n".join(cases_text_parts)

        # Build constraints string
        constraints_str = ""
        if user_spec.constraints:
            constraints_str = ", ".join(
                f"{c.metric} {c.comparison} {c.target}{c.unit or ''}"
                for c in user_spec.constraints
            )

        prompt = RERANK_PROMPT.format(
            n_cases=len(cases),
            description=user_spec.description,
            task_type=user_spec.task_type.value if user_spec.task_type else "unknown",
            modality=user_spec.input_type.value if user_spec.input_type else "unknown",
            target_device=target_device or "Not specified",
            constraints=constraints_str or "None specified",
            cases_text=cases_text
        )

        try:
            response = self.llm.invoke([
                SystemMessage(content="You are an expert at matching past solutions to new problems."),
                HumanMessage(content=prompt)
            ])

            # Parse ranking from response
            ranking_text = response.content.strip()
            ranking_numbers = re.findall(r'\[(\d+)\]', ranking_text)

            if ranking_numbers:
                # Get the first (best) case
                best_idx = int(ranking_numbers[0]) - 1
                if 0 <= best_idx < len(cases):
                    logger.debug(f"LLM reranked cases, best: {cases[best_idx].id}")
                    return cases[best_idx]

            # Fallback to first case if parsing fails
            return cases[0]

        except Exception as e:
            logger.warning(f"LLM reranking failed: {e}, using first case")
            return cases[0]

    def find_best_case(
        self,
        user_spec: UserSpec,
        target_device: str = "",
        n_candidates: int = 5
    ) -> Optional[Case]:
        """Find the best matching case using vector search + LLM reranking.

        This is the main entry point for CBR retrieval.

        Args:
            user_spec: Current task specification.
            target_device: Target deployment device.
            n_candidates: Number of candidates to retrieve before reranking.

        Returns:
            The best matching case, or None if no cases found.
        """
        # First, vector search for candidates
        candidates = self.case_store.search_similar(
            query=user_spec.description,
            n_results=n_candidates,
            modality=user_spec.input_type,
            task_type=user_spec.task_type
        )

        if not candidates:
            logger.debug("No similar cases found in case store")
            return None

        logger.debug(f"Found {len(candidates)} candidate cases, reranking...")

        # Then, LLM rerank
        return self.rerank_cases(candidates, user_spec, target_device)

    def adapt_solution(
        self,
        similar_case: Case,
        new_constraints: Dict[str, Any],
        new_device: str
    ) -> Dict[str, Any]:
        """Adapt a past solution to new constraints.

        Args:
            similar_case: A similar past case.
            new_constraints: New constraint requirements.
            new_device: New target device.

        Returns:
            Adapted solution configuration.
        """
        # Start with the successful solution
        adapted = {
            "model_name": similar_case.model_name,
            "training_config": similar_case.training_config.copy(),
            "optimization_config": similar_case.optimization_config.copy(),
            "source_case_id": similar_case.id,
            "adaptations": []
        }

        # Adapt based on constraint differences
        old_constraints = similar_case.constraints

        # If new latency constraint is tighter, suggest smaller model
        if "latency_ms" in new_constraints:
            new_lat = new_constraints["latency_ms"]
            old_lat = old_constraints.get("latency_ms", float("inf"))

            if new_lat < old_lat * 0.8:  # Significantly tighter
                adapted["adaptations"].append({
                    "reason": f"Latency constraint tightened from {old_lat}ms to {new_lat}ms",
                    "suggestion": "Consider a smaller model variant (e.g., nano instead of small)"
                })

        # If new memory constraint is tighter, suggest more aggressive quantization
        if "memory_mb" in new_constraints:
            new_mem = new_constraints["memory_mb"]
            old_mem = old_constraints.get("memory_mb", float("inf"))

            if new_mem < old_mem * 0.8:
                adapted["adaptations"].append({
                    "reason": f"Memory constraint tightened from {old_mem}MB to {new_mem}MB",
                    "suggestion": "Use INT8 quantization instead of FP16"
                })
                adapted["optimization_config"]["precision"] = "int8"

        # If accuracy constraint is higher, suggest larger model or more epochs
        if "accuracy" in new_constraints:
            new_acc = new_constraints["accuracy"]
            old_acc = old_constraints.get("accuracy", 0)

            if new_acc > old_acc + 0.05:
                adapted["adaptations"].append({
                    "reason": f"Accuracy requirement increased from {old_acc} to {new_acc}",
                    "suggestion": "Consider a larger model or more training epochs"
                })
                adapted["training_config"]["epochs"] = adapted["training_config"].get("epochs", 50) * 2

        return adapted

    def get_recommended_config(
        self,
        intent: str,
        modality: Modality,
        task_type: TaskType,
        constraints: Dict[str, Any],
        target_device: str
    ) -> Optional[Dict[str, Any]]:
        """Get a recommended configuration based on similar cases.

        Args:
            intent: User's intent.
            modality: Task modality.
            task_type: Task type.
            constraints: Constraints.
            target_device: Target device.

        Returns:
            Recommended configuration or None if no similar cases found.
        """
        similar_cases = self.find_similar_cases(
            intent=intent,
            modality=modality,
            task_type=task_type,
            n_results=1
        )

        if not similar_cases:
            return None

        best_case = similar_cases[0]

        # Only use successful cases
        if not best_case.success:
            return None

        return self.adapt_solution(
            similar_case=best_case,
            new_constraints=constraints,
            new_device=target_device
        )

    def record_case(
        self,
        intent: str,
        modality: Modality,
        task_type: TaskType,
        constraints: Dict[str, Any],
        target_device: str,
        solution: Dict[str, Any],
        metrics: Dict[str, float],
        success: bool,
        lessons: str = ""
    ) -> Case:
        """Record a new case from a completed task.

        Args:
            intent: User's original intent.
            modality: Task modality.
            task_type: Task type.
            constraints: Task constraints.
            target_device: Target device.
            solution: The solution that was used.
            metrics: Final metrics achieved.
            success: Whether constraints were met.
            lessons: Lessons learned.

        Returns:
            The created case.
        """
        import uuid

        case = Case(
            id=f"case_{uuid.uuid4().hex[:8]}",
            intent=intent,
            modality=modality,
            task_type=task_type,
            constraints=constraints,
            target_device=target_device,
            model_name=solution.get("model_name", ""),
            training_config=solution.get("training_config", {}),
            optimization_config=solution.get("optimization_config", {}),
            metrics=metrics,
            success=success,
            lessons_learned=lessons
        )

        self.case_store.add_case(case)
        return case
