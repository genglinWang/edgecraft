"""Planner node for the EdgeCraft agent."""
import json
from typing import Dict, Any, Optional
from loguru import logger
from langchain_core.messages import HumanMessage, SystemMessage

from edgecraft.agent.state import AgentState
from edgecraft.agent.prompts.system import SYSTEM_PROMPT, PLANNER_PROMPT, PLANNER_PROMPT_WITH_KNOWLEDGE, MODALITY_DETECTION_PROMPT
from edgecraft.core.modality import Modality, TaskType
from edgecraft.core.task import UserSpec
from edgecraft.knowledge import KnowledgeOrchestrator, RetrievedKnowledge
from edgecraft.utils.llm import create_chat_llm


def detect_modality(state: AgentState, llm) -> tuple[Modality, TaskType]:
    """Use LLM to detect modality and task type from user intent."""
    prompt = MODALITY_DETECTION_PROMPT.format(
        user_intent=state["raw_user_intent"],
        dataset_path=state.get("dataset_path", "Not provided"),
        sample_files="(to be analyzed)"
    )

    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt)
    ])

    try:
        # Extract JSON from response
        content = response.content
        # Try to parse JSON
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        result = json.loads(content.strip())
        modality = Modality(result.get("modality", "vision"))
        task_type = TaskType(result.get("task_type", "classification"))
        return modality, task_type
    except (json.JSONDecodeError, ValueError):
        # Default to vision classification
        return Modality.VISION, TaskType.CLASSIFICATION


VALID_TOOLS = {
    "ultralytics_trainer",
    "timm_trainer",
    "tensorrt_exporter",
    "edge_runner",
    "dataset_analyzer"
}


def validate_plan(plan: list) -> bool:
    """Check if all tools in the plan are valid."""
    for step in plan:
        tool_name = step.get("tool_name", "")
        if tool_name not in VALID_TOOLS:
            return False
    return True


def create_plan(
    state: AgentState,
    llm: ChatOpenAI,
    retrieved_knowledge: Optional[RetrievedKnowledge] = None
) -> list[Dict[str, Any]]:
    """Create execution plan based on detected modality and user intent.

    Args:
        state: Current agent state.
        llm: LLM for plan generation.
        retrieved_knowledge: Optional knowledge from RAG/CBR.

    Returns:
        List of plan steps.
    """
    # For MVP: If we already know modality and task type, use default plan
    # This avoids LLM generating invalid tool names
    if state.get("modality") and state.get("task_type"):
        return get_default_plan(state, retrieved_knowledge)

    # Choose prompt based on whether we have retrieved knowledge
    if retrieved_knowledge and retrieved_knowledge.has_knowledge():
        prompt = PLANNER_PROMPT_WITH_KNOWLEDGE.format(
            user_intent=state["raw_user_intent"],
            dataset_path=state.get("dataset_path", "Not provided"),
            target_device=state["target_device"],
            constraints=json.dumps(state.get("constraints", {})),
            modality=state.get("modality", "unknown"),
            task_type=state.get("task_type", "unknown"),
            tool_results=json.dumps(state.get("tool_results", [])),
            edge_feedback=json.dumps(state.get("edge_feedback", {})),
            retrieved_knowledge=retrieved_knowledge.combined_prompt
        )
    else:
        prompt = PLANNER_PROMPT.format(
            user_intent=state["raw_user_intent"],
            dataset_path=state.get("dataset_path", "Not provided"),
            target_device=state["target_device"],
            constraints=json.dumps(state.get("constraints", {})),
            modality=state.get("modality", "unknown"),
            task_type=state.get("task_type", "unknown"),
            tool_results=json.dumps(state.get("tool_results", [])),
            edge_feedback=json.dumps(state.get("edge_feedback", {}))
        )

    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt)
    ])

    try:
        content = response.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        plan = json.loads(content.strip())
        plan = plan if isinstance(plan, list) else [plan]

        # Validate the plan - if any invalid tools, use default
        if not validate_plan(plan):
            return get_default_plan(state, retrieved_knowledge)

        return plan
    except json.JSONDecodeError:
        # Return a default plan for vision tasks
        return get_default_plan(state, retrieved_knowledge)


def get_default_plan(
    state: AgentState,
    retrieved_knowledge: Optional[RetrievedKnowledge] = None
) -> list[Dict[str, Any]]:
    """Return a default plan based on modality, optionally informed by retrieved knowledge.

    Args:
        state: Current agent state.
        retrieved_knowledge: Optional knowledge from RAG/CBR that may influence model selection.

    Returns:
        List of plan steps.
    """
    modality = state.get("modality", Modality.VISION)
    task_type = state.get("task_type", TaskType.CLASSIFICATION)

    # Get dataset config path from dataset_info if available
    dataset_info = state.get("dataset_info", {})
    dataset_path = dataset_info.get("config_path") or state.get("dataset_path")

    if modality == Modality.VISION:
        # Choose trainer based on task type
        if task_type in [TaskType.OBJECT_DETECTION, TaskType.SEGMENTATION, TaskType.POSE_ESTIMATION]:
            trainer_name = "ultralytics_trainer"
            model_name = "yolo11n.pt"
        else:
            trainer_name = "ultralytics_trainer"  # Also works for classification
            model_name = "yolo11n-cls.pt"

        return [
            {
                "step": 1,
                "tool_name": trainer_name,
                "parameters": {
                    "dataset_path": dataset_path,
                    "task_type": task_type,
                    "model_name": model_name,
                    "epochs": 1  # Quick test - change to 50 for production
                },
                "expected_output": "Trained model weights"
            },
            {
                "step": 2,
                "tool_name": "tensorrt_exporter",
                "parameters": {
                    "model_path": "<from_previous_step>",
                    "format": "onnx"  # Export ONNX only - TensorRT compiled on edge device
                },
                "expected_output": "Exported ONNX model"
            },
            {
                "step": 3,
                "tool_name": "edge_runner",
                "parameters": {
                    "artifact_path": "<from_previous_step>",
                    "device_id": state["target_device"],
                    "device_ip": state.get("device_ip", ""),
                    "ssh_key": state.get("ssh_key", "")
                },
                "expected_output": "Edge deployment and benchmark results"
            }
        ]
    else:
        # Placeholder for other modalities
        return [
            {
                "step": 1,
                "tool_name": "not_implemented",
                "parameters": {"modality": modality.value},
                "expected_output": f"Tools for {modality.value} are not yet implemented"
            }
        ]


def retrieve_knowledge_for_planning(
    state: AgentState,
    enable_rag: bool = True,
    enable_cbr: bool = True
) -> Optional[RetrievedKnowledge]:
    """Retrieve knowledge from RAG and CBR for planning.

    Args:
        state: Current agent state.
        enable_rag: Whether to retrieve from RAG sources.
        enable_cbr: Whether to retrieve from CBR case store.

    Returns:
        Retrieved knowledge or None if retrieval fails.
    """
    # Build UserSpec from state (intent-side fields; full parse lives in preflight)
    modality = state.get("modality")
    task_type = state.get("task_type")
    constraints = state.get("constraints", {})

    user_spec = UserSpec(
        description=state.get("raw_user_intent", ""),
        task_type=task_type,
        input_type=modality,
        dataset_path=state.get("dataset_path"),
        constraints=[],  # TODO: parse from constraints dict
        preferences=[]
    )

    target_device = state.get("target_device", "")

    try:
        orchestrator = KnowledgeOrchestrator()
        knowledge = orchestrator.retrieve_knowledge(
            user_spec=user_spec,
            target_device=target_device,
            enable_rag=enable_rag,
            enable_cbr=enable_cbr
        )
        return knowledge
    except Exception as e:
        logger.warning(f"Knowledge retrieval failed: {e}")
        return None


def planner_node(state: AgentState) -> AgentState:
    """Planner node: Detect modality, retrieve knowledge, and create execution plan."""
    llm = create_chat_llm(temperature=0.1, purpose="planner")

    # Detect modality if not already set
    if not state.get("modality"):
        modality, task_type = detect_modality(state, llm)
        state["modality"] = modality
        state["task_type"] = task_type

    # Retrieve knowledge from RAG and CBR (only on first iteration)
    retrieved_knowledge = None
    if state.get("iteration", 0) == 0:
        logger.info("Retrieving knowledge for planning...")
        retrieved_knowledge = retrieve_knowledge_for_planning(
            state,
            enable_rag=True,
            enable_cbr=True
        )
        if retrieved_knowledge and retrieved_knowledge.has_knowledge():
            logger.info("Knowledge retrieved successfully")
            state["retrieved_knowledge"] = retrieved_knowledge.combined_prompt
            if retrieved_knowledge.similar_case:
                state["similar_case_id"] = retrieved_knowledge.similar_case.id
                state["retrieved_case_ids"] = [retrieved_knowledge.similar_case.id]
        else:
            logger.info("No relevant knowledge found")
            state["retrieved_knowledge"] = ""

    # Create or update plan
    plan = create_plan(state, llm, retrieved_knowledge)
    state["plan"] = plan
    state["current_step"] = 0
    state["status"] = "executing"

    # Add to messages for context
    knowledge_note = ""
    if retrieved_knowledge and retrieved_knowledge.similar_case:
        knowledge_note = f" (informed by past case: {retrieved_knowledge.similar_case.id})"

    state["messages"].append({
        "role": "assistant",
        "content": f"Created plan with {len(plan)} steps for {state['modality'].value} {state['task_type'].value} task{knowledge_note}."
    })

    return state
