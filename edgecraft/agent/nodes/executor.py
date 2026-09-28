"""Executor node for the EdgeCraft agent."""
from typing import Dict, Any, Optional
from edgecraft.agent.state import AgentState
from edgecraft.core.modality import Modality
from edgecraft.tools.base import ToolRegistry

# Import tools to trigger registration
# These imports ensure tools are registered when the executor module is loaded
def _ensure_tools_registered():
    """Ensure all tools are registered."""
    try:
        from edgecraft.tools.train.vision.ultralytics_adapter import UltralyticsTrainer
        from edgecraft.tools.train.vision.timm_adapter import TimmTrainer
        from edgecraft.tools.deploy.edge_runner import EdgeRunner
        from edgecraft.tools.deploy.vision.trt_exporter import TensorRTExporter
    except ImportError as e:
        print(f"Warning: Could not import some tools: {e}")

_ensure_tools_registered()


def resolve_parameters(parameters: Dict[str, Any], tool_results: list) -> Dict[str, Any]:
    """Resolve parameter placeholders with results from previous steps.

    Handles:
    - "<from_previous_step>" - gets the main output path from the last step
    - "<from_step_N>" - gets output from step N
    """
    resolved = {}

    for key, value in parameters.items():
        if isinstance(value, str) and value == "<from_previous_step>":
            # Get output from the last successful step
            if tool_results:
                last_result = tool_results[-1].get("result", {})
                # Try common output keys
                resolved[key] = (
                    last_result.get("model_path") or
                    last_result.get("export_path") or
                    last_result.get("artifact_path") or
                    last_result.get("output_path") or
                    value  # Keep placeholder if no result found
                )
            else:
                resolved[key] = value
        else:
            resolved[key] = value

    return resolved


def execute_step(state: AgentState, step: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a single step of the plan."""
    tool_name = step.get("tool_name")
    parameters = step.get("parameters", {})
    modality = state.get("modality", Modality.VISION)

    # Resolve parameter placeholders from previous step results
    tool_results = state.get("tool_results", [])
    resolved_params = resolve_parameters(parameters, tool_results)

    # Get the tool from the global ToolRegistry
    tool = ToolRegistry.get(tool_name, modality)

    if tool is None:
        # List available tools for debugging
        available = ToolRegistry.list_tools(modality)
        return {
            "status": "error",
            "tool_name": tool_name,
            "error": f"Tool '{tool_name}' not found for modality '{modality.value if modality else 'global'}'",
            "available_tools": available
        }

    try:
        # Call the tool's run method with resolved parameters
        result = tool.run(**resolved_params)
        return {
            "status": "success",
            "tool_name": tool_name,
            "result": result
        }
    except Exception as e:
        return {
            "status": "error",
            "tool_name": tool_name,
            "error": str(e)
        }


def executor_node(state: AgentState) -> AgentState:
    """Executor node: Execute the current step of the plan."""
    plan = state.get("plan", [])
    current_step = state.get("current_step", 0)

    if current_step >= len(plan):
        state["status"] = "reflecting"
        return state

    step = plan[current_step]
    result = execute_step(state, step)

    # Store result
    if "tool_results" not in state:
        state["tool_results"] = []
    state["tool_results"].append({
        "step": current_step,
        **result
    })

    # If this was an edge_runner step, extract edge_feedback for constraint checking
    tool_name = step.get("tool_name", "")
    if tool_name == "edge_runner" and result.get("status") == "success":
        result_data = result.get("result", {})
        # Extract metrics from edge_runner result
        metrics = result_data.get("metrics", {})
        state["edge_feedback"] = {
            "device_id": result_data.get("device_id"),
            "job_id": result_data.get("job_id"),
            "metrics": metrics,
            "stdout": result_data.get("stdout", "")[:500],  # Truncate for state
            "status": result_data.get("status")
        }

    # Update messages
    state["messages"].append({
        "role": "assistant",
        "content": f"Executed step {current_step + 1}: {step.get('tool_name')} - {result.get('status')}"
    })

    # Move to next step or reflect
    if current_step + 1 >= len(plan):
        state["status"] = "reflecting"
    else:
        state["current_step"] = current_step + 1

    return state
