"""AgentState for the EdgeCraft constraint-aware synthesis agent."""
from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict

from edgecraft.core.modality import Modality, TaskType
from edgecraft.core.task import RuntimeConfig, UserSpec

# Real imports needed so LangGraph can resolve type hints at runtime
# (get_type_hints called internally by StateGraph)
from edgecraft.agent.search.solution_variant import SolutionVariant
from edgecraft.agent.search.trial_result import TrialResult
from edgecraft.agent.search.trial_bank import TrialBank
from edgecraft.agent.search.refinement_tree import RefinementTree


class AgentState(TypedDict):
    """State passed between LangGraph nodes.

    Key changes from v1
    -------------------
    - ``trial_bank`` replaces ``tool_results`` and is *never* cleared.
    - ``search_tree`` holds private candidate lineage and measured state.
    - ``pending_variants`` buffers the k variants generated in one LLM call
      so the executor only runs one per graph iteration.
    - ``plan`` / ``current_step`` / ``reflection`` (free-text) are removed.
    - ``user_spec`` carries structured fields parsed from user intent only.
    - ``runtime_config`` constrains the search space.
    """

    # ------------------------------------------------------------------
    # User inputs (set once at startup)
    # ------------------------------------------------------------------
    raw_user_intent: str
    dataset_path: Optional[str]  # dataset root directory (not the resolved yaml path)
    dataset_report: Optional[str]  # unstructured exploration text from preflight
    dataset_info: Optional[Dict[str, Any]]
    target_device: str
    device_ip: Optional[str]
    ssh_key: Optional[str]
    # Pre-built Docker image on the edge (from CLI / caller); used by EdgeRunner.
    docker_image: Optional[str]

    # ------------------------------------------------------------------
    # Parsed / inferred (set during preflight or first planner call)
    # ------------------------------------------------------------------
    user_spec: Optional[UserSpec]
    modality: Optional[Modality]
    task_type: Optional[TaskType]
    runtime_config: Optional[RuntimeConfig]
    retrieved_case_ids: List[str]
    retrieved_rule_ids: List[str]

    # ------------------------------------------------------------------
    # Search identity
    # ------------------------------------------------------------------
    run_id: str
    tenant_id: str
    continuation_parent_trial_id: Optional[str]

    # ------------------------------------------------------------------
    # Constraint-aware search state (complex objects — no checkpointer used)
    # ------------------------------------------------------------------
    trial_bank: TrialBank
    search_tree: RefinementTree
    best_feasible_trial: Optional[TrialResult]

    # ------------------------------------------------------------------
    # Per-iteration execution
    # ------------------------------------------------------------------
    # Variants queued for execution (ProposalGenerator fills this with k
    # candidates; PipelineExecutor consumes one per graph iteration).
    pending_variants: List[SolutionVariant]
    current_variant: Optional[SolutionVariant]
    current_trial_result: Optional[TrialResult]

    # Structured LLM diagnosis from the last Reflector call
    reflector_diagnosis: Optional[Dict[str, Any]]
    branch_selection_history: List[Dict[str, Any]]
    l1_forced_audit_trial_ids: List[str]

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------
    iteration: int           # incremented by Scorer after each trial
    max_iterations: int
    branching_factor: int    # number of children proposed per expansion
    status: str              # "searching" | "completed" | "failed"
    error: Optional[str]

    # ------------------------------------------------------------------
    # Conversation history (for LLM context accumulation)
    # ------------------------------------------------------------------
    messages: List[Dict[str, str]]
