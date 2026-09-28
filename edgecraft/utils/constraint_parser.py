"""Constraint parser using LLM to extract performance metrics and UserSpec from user intent."""
import json
import math
import re
from typing import Dict, Any, List, Optional, Tuple
from langchain_core.messages import HumanMessage

from edgecraft.agent.prompts.system import USER_SPEC_PARSING_PROMPT
from edgecraft.core.task import UserSpec
from edgecraft.utils.llm import create_chat_llm


def _drop_unbounded_constraints(data: Dict[str, Any]) -> Dict[str, Any]:
    """Remove only constraints that have no finite numeric boundary.

    "Measure latency" describes evidence to collect, not a hard constraint.
    Some LLM backends encode it as ``target=null``; dropping that one invalid
    entry preserves the rest of the parsed UserSpec.
    """
    constraints = data.get("constraints")
    if not isinstance(constraints, list):
        return data
    bounded: List[Dict[str, Any]] = []
    for item in constraints:
        if not isinstance(item, dict):
            bounded.append(item)
            continue
        try:
            target = float(item.get("target"))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(target):
            continue
        bounded.append({**item, "target": target})
    return {**data, "constraints": bounded}


_NUMERIC_LITERAL = re.compile(
    r"(?<![A-Za-z0-9])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?![A-Za-z0-9])"
)


def _drop_constraints_without_numeric_intent(
    data: Dict[str, Any], user_intent: str
) -> Dict[str, Any]:
    """Do not let an LLM turn an unbounded objective into a hard target.

    Constraint extraction has one simple grounding invariant: a numeric bound
    cannot exist when the user supplied no numeric literal. Metric names such
    as F1 and dataset names such as CIFAR10 do not count as literals.
    """
    if _NUMERIC_LITERAL.search(user_intent):
        return data
    return {**data, "constraints": []}


def format_constraints_for_display(constraints: List[Dict[str, Any]], preferences: Optional[List[Dict[str, Any]]] = None) -> str:
    """Format constraints and preferences list for human-readable display.

    Args:
        constraints: List of constraint dicts with metric, comparison, target, unit, scope.
        preferences: Optional list of preference dicts with metric, direction, unit, scope.

    Returns:
        Formatted string describing the constraints and preferences.
    """
    parts = []

    # Format constraints
    if constraints:
        for constraint in constraints:
            metric = constraint.get("metric", "unknown")
            target = constraint.get("target", "?")
            comparison = constraint.get("comparison", "gte")
            unit = constraint.get("unit")

            if comparison == "gte":
                symbol = "≥"
            elif comparison == "lte":
                symbol = "≤"
            elif comparison == "eq":
                symbol = "="
            else:
                symbol = "?"

            unit_str = f" {unit}" if unit else ""
            parts.append(f"{metric} {symbol} {target}{unit_str}")

    # Format preferences
    if preferences:
        for preference in preferences:
            metric = preference.get("metric", "unknown")
            direction = preference.get("direction", "minimize")
            unit = preference.get("unit")

            if direction == "minimize":
                direction_str = "minimize"
            elif direction == "maximize":
                direction_str = "maximize"
            else:
                direction_str = direction

            unit_str = f" ({unit})" if unit else ""
            parts.append(f"{metric}: {direction_str}{unit_str}")

    if not parts:
        return "No constraints or preferences specified"

    return ", ".join(parts)


def parse_user_spec_from_intent(user_intent: str) -> UserSpec:
    """Parse a full UserSpec from user's natural language intent.

    Args:
        user_intent: User's natural language description.

    Returns:
        UserSpec object validated by Pydantic.
    """
    llm = create_chat_llm(temperature=0.1, purpose="user_spec_parser")
    parsing_prompt = USER_SPEC_PARSING_PROMPT.format(user_intent=user_intent)

    try:
        response = llm.invoke([HumanMessage(content=parsing_prompt)])
        content = response.content.strip()

        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        data = _drop_unbounded_constraints(json.loads(content))
        data = _drop_constraints_without_numeric_intent(data, user_intent)

        # Ensure description is present for Pydantic
        if not data.get("description"):
            data["description"] = user_intent[:100] + "..." if len(user_intent) > 100 else user_intent

        return UserSpec(**data)

    except Exception as e:
        print(f"Error parsing UserSpec: {e}")
        return UserSpec(description=user_intent)


def parse_constraints_from_intent(user_intent: str) -> Dict[str, Any]:
    """Parse intent and return constraints/preferences as dict (backward-compat for tests).
    Prefer parse_user_spec_from_intent for full UserSpec."""
    spec = parse_user_spec_from_intent(user_intent)
    return {
        "constraints": [c.model_dump() for c in spec.constraints] if spec.constraints else [],
        "preferences": [p.model_dump() for p in spec.preferences] if spec.preferences else [],
    }


def get_user_spec_display_rows(spec: UserSpec) -> List[Tuple[str, str]]:
    """Return (field, value) rows for UserSpec for use in combined display."""
    rows: List[Tuple[str, str]] = []
    rows.append(("Description", spec.description))
    if spec.dataset_name:
        rows.append(("Dataset Name", spec.dataset_name))
    if spec.dataset_path:
        rows.append(("Dataset Path", spec.dataset_path))
    if spec.task_type:
        rows.append(("Task Type", spec.task_type.value))
    if spec.input_type:
        rows.append(("Input Type", spec.input_type.value))
    if spec.output_type:
        rows.append(("Output Type", spec.output_type))
    if spec.constraints:
        c_str = format_constraints_for_display([c.model_dump() for c in spec.constraints])
        rows.append(("Constraints", c_str))
    if spec.preferences:
        p_str = format_constraints_for_display([], [p.model_dump() for p in spec.preferences])
        rows.append(("Preferences", p_str))
    if spec.eval_metrics:
        rows.append(("Eval Metrics", ", ".join(spec.eval_metrics)))
    return rows


def format_user_spec_for_display(spec: UserSpec):
    """Format a UserSpec for human-readable display.

    Returns:
        A rich.table.Table object that can be printed by a Console.
    """
    from rich.table import Table
    from rich import box

    table = Table(title="User specification", box=box.ROUNDED)
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="white")
    for field, value in get_user_spec_display_rows(spec):
        table.add_row(field, value)
    return table
