"""SolutionVariant: the atomic unit of iterative tree search in EdgeCraft.

In the code-centric architecture, SolutionVariant stores:
- Structured metadata for search (model_name, quant_mode, export_format) for Surrogate/Scorer
- Executable code (train_code, infer_code) which IS the search space
- Natural language plan for LLM reasoning

The LLM generates code directly rather than tuning JSON parameters.
"""
from __future__ import annotations

import uuid
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

from edgecraft.core.modality import Modality, TaskType


class SolutionVariant(BaseModel):
    """Configuration node in the iterative search tree (code-centric version)."""

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------
    trial_id: str = Field(
        default_factory=lambda: f"trial_{uuid.uuid4().hex[:8]}",
        description="Unique identifier assigned at creation time.",
    )
    parent_trial_id: Optional[str] = Field(
        None,
        description="trial_id of the parent node in the iterative tree (None for root).",
    )

    # ------------------------------------------------------------------
    # Search metadata (structured, for Surrogate/Scorer/RuntimeConfig)
    # ------------------------------------------------------------------
    modality: Modality
    task_type: TaskType
    model_name: str = Field(
        "yolo11n",
        description="Model identifier for Surrogate latency/memory lookup.",
    )
    model_family: str = Field(
        "ultralytics",
        description="Model family for template lookup: 'ultralytics' | 'timm' | 'huggingface'.",
    )
    quant_mode: Literal["fp32", "fp16", "int8"] = Field(
        "fp16",
        description="Quantization mode for Surrogate memory estimation.",
    )
    export_format: Literal["onnx", "engine", "tflite", "pt"] = Field(
        "onnx",
        description="Export format for RuntimeConfig validation.",
    )

    # ------------------------------------------------------------------
    # Code (the actual search space)
    # ------------------------------------------------------------------
    plan: str = Field(
        "",
        description="Natural language reasoning about why this configuration.",
    )
    dataset_plan: str = Field(
        "",
        description=(
            "Evidence-first implementation plan describing the observed data files, "
            "label source, split strategy, input kind, target, and baseline choice "
            "before executable code is generated."
        ),
    )
    train_code: str = Field(
        "",
        description="Complete train.py content (train + eval + save model).",
    )
    loader_code: str = Field(
        "",
        description="Optional complete loader.py content for dataset loading and smoke checks.",
    )
    infer_code: str = Field(
        "",
        description="Complete infer.py content (export + benchmark on edge).",
    )

    # ------------------------------------------------------------------
    # Code-space lineage (free-form, not a fixed parameter schema)
    # ------------------------------------------------------------------
    search_dimension: str = Field(
        "initial",
        description=(
            "Natural-language label of the search dimension this variant explores. "
            "Examples: 'architecture: switch to lighter backbone', "
            "'quantization: try int8', 'training: more epochs + augmentation', "
            "'compression: channel pruning 30%', 'input: reduce resolution'. "
            "Used by iterative tree search for diversity tracking and prompt context."
        ),
    )
    mutation_type: str = Field(
        "initial",
        description=(
            "Free-form label for the code-space evolution represented by the "
            "edge from parent to this node, e.g. repair, specialize, simplify, "
            "change_model_family, change_runtime, change_adapter, change_training_recipe."
        ),
    )
    inherited_components: List[str] = Field(
        default_factory=list,
        description=(
            "Code or evidence components intentionally inherited from the parent "
            "variant, e.g. train.py preprocessing, dataset adapter, best.pt, "
            "runtime fallback, metric extraction."
        ),
    )
    changed_components: List[str] = Field(
        default_factory=list,
        description=(
            "Code components intentionally changed in this variant.  This keeps "
            "the iterative tree search centered on executable code evolution "
            "rather than a fixed searchable parameter list."
        ),
    )
    evidence_refs: List[str] = Field(
        default_factory=list,
        description="Visible Evidence IDs that justify this code-space mutation.",
    )

    # ------------------------------------------------------------------
    # Provenance
    # ------------------------------------------------------------------
    proposal_reasoning: str = Field(
        "",
        description="LLM explanation of why this variant was proposed.",
    )
    proposal_hypothesis: str = Field(
        "",
        description="LLM prediction of expected effect.",
    )
    prior_score: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description=(
            "LLM-supplied feasibility prior in [0, 1].  Higher = more likely "
            "to satisfy hard constraints on the target device.  Used as the "
            "scheduling priority hint inside the JobPool; never used to pick "
            "which node to expand."
        ),
    )
    contract_rewrite_reason: str = Field(
        "",
        description="Reason when EdgeCraft replaced or normalized generated scripts with a contract template.",
    )
    requested_modality: str = Field(
        "",
        description="Task-contract modality requested by user/preflight before code generation.",
    )
    requested_task_type: str = Field(
        "",
        description="Task-contract type requested by user/preflight before code generation.",
    )
    observed_modality: str = Field(
        "",
        description="DatasetAnalyzer-observed modality when available; evidence only, not an override.",
    )
    observed_task_type: str = Field(
        "",
        description="DatasetAnalyzer-observed task type when available; evidence only, not an override.",
    )
    modality_shift_reason: str = Field(
        "",
        description="Evidence-only note when proposal vocabulary suggests another modality.",
    )

    # ------------------------------------------------------------------
    # Code-space search axes (observability only, not a fixed grid)
    # ------------------------------------------------------------------
    solution_source: str = Field(
        "unknown",
        description=(
            "Where the executable solution primarily comes from: library_model, "
            "scratch_torch_model, hybrid, or unknown. Used for depth reporting only."
        ),
    )
    initialization_source: str = Field(
        "unknown",
        description=(
            "Parameter initialization strategy: random_init, pretrained_finetune, "
            "frozen_feature_extractor, distilled_or_compressed, or unknown."
        ),
    )
    representation_strategy: str = Field(
        "unknown",
        description="How raw data is transformed into model inputs, e.g. mel_spectrogram, tfidf, sliding_window_imu.",
    )
    representation_stage: str = Field(
        "unknown",
        description=(
            "Representation-first evolution stage: discover, stabilize, optimize, or unknown. "
            "Used for prompt/report lineage only; it is not a fixed grid parameter."
        ),
    )
    model_capacity_strategy: str = Field(
        "unknown",
        description="Capacity choice such as tiny, small, medium, adaptive, or unknown.",
    )
    training_recipe: str = Field(
        "unknown",
        description="Compact summary of loss, optimizer, augmentation, class balance, epochs, or early stopping.",
    )
    export_runtime_strategy: str = Field(
        "unknown",
        description="Compact summary of artifact/export/runtime route such as pt, onnxruntime_cpu, TensorRT candidate.",
    )
    search_granularity: str = Field(
        "unknown",
        description=(
            "LLM-selected mutation granularity for code-space evolution: coarse, mid, fine, or unknown. "
            "Used for depth reporting and prompt context only."
        ),
    )
    why_this_granularity: str = Field(
        "",
        description="Short rationale for why this mutation granularity fits the current evidence stage.",
    )
    evidence_support: str = Field(
        "",
        description="Measured artifact/runtime/metric evidence supporting this granularity choice.",
    )
    complexity_risk: str = Field(
        "",
        description="Known risk when this proposal introduces more complex code-space changes.",
    )
    search_intent: str = Field(
        "unknown",
        description=(
            "LLM-declared tree-search intent for this code variant: explore, exploit, repair, "
            "or unknown. This is an observability and tree-selection hint, not a hard rule."
        ),
    )
    explore_exploit_rationale: str = Field(
        "",
        description="Short rationale for why this proposal should explore a new branch or exploit known evidence.",
    )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def short_description(self) -> str:
        """Compact single-line summary for logging and Rich tables."""
        dim = self.search_dimension if self.search_dimension != "initial" else ""
        suffix = f" | dim={dim}" if dim else ""
        return f"{self.model_name} | {self.quant_mode} | export={self.export_format}{suffix}"

    def to_prompt_text(self) -> str:
        """Format for inclusion in LLM prompts (for improve/debug iterations)."""
        lines = [
            f"model_name: {self.model_name}",
            f"model_family: {self.model_family}",
            f"quant_mode: {self.quant_mode}",
            f"export_format: {self.export_format}",
            f"search_dimension: {self.search_dimension}",
            f"mutation_type: {self.mutation_type}",
        ]
        if self.requested_modality or self.observed_modality:
            lines.append(
                "task_truth: "
                f"requested={self.requested_modality or 'unknown'}/{self.requested_task_type or 'unknown'}, "
                f"observed={self.observed_modality or 'unknown'}/{self.observed_task_type or 'unknown'}, "
                f"executed={self.modality.value}/{self.task_type.value}"
            )
        if self.modality_shift_reason:
            lines.append(f"modality_shift_reason: {self.modality_shift_reason}")
        axes = {
            "solution_source": self.solution_source,
            "initialization_source": self.initialization_source,
            "representation_strategy": self.representation_strategy,
            "representation_stage": self.representation_stage,
            "model_capacity_strategy": self.model_capacity_strategy,
            "training_recipe": self.training_recipe,
            "export_runtime_strategy": self.export_runtime_strategy,
            "search_granularity": self.search_granularity,
            "search_intent": self.search_intent,
        }
        if any(value and value != "unknown" for value in axes.values()):
            lines.append("search_axes: " + "; ".join(f"{key}={value or 'unknown'}" for key, value in axes.items()))
        if self.why_this_granularity or self.evidence_support or self.complexity_risk:
            lines.append(
                "granularity_rationale: "
                f"why={self.why_this_granularity or 'n/a'}; "
                f"evidence={self.evidence_support or 'n/a'}; "
                f"risk={self.complexity_risk or 'n/a'}"
            )
        if self.explore_exploit_rationale:
            lines.append(f"explore_exploit_rationale: {self.explore_exploit_rationale}")
        if self.inherited_components:
            lines.append(f"inherited_components: {', '.join(self.inherited_components)}")
        if self.changed_components:
            lines.append(f"changed_components: {', '.join(self.changed_components)}")
        if self.plan:
            lines.append(f"\nPlan:\n{self.plan}")
        if self.dataset_plan:
            plan_lines = self.dataset_plan.strip().splitlines()
            preview = "\n".join(plan_lines[:18])
            if len(plan_lines) > 18:
                preview += "\n# ... (truncated)"
            lines.append(f"\nDataset implementation plan:\n{preview}")
        if self.loader_code:
            code_lines = self.loader_code.strip().splitlines()
            preview = "\n".join(code_lines[:25])
            if len(code_lines) > 25:
                preview += "\n# ... (truncated)"
            lines.append(f"\nloader.py (preview):\n```python\n{preview}\n```")
        if self.train_code:
            # Include first ~30 lines of train_code for context
            code_lines = self.train_code.strip().splitlines()
            preview = "\n".join(code_lines[:30])
            if len(code_lines) > 30:
                preview += "\n# ... (truncated)"
            lines.append(f"\ntrain.py (preview):\n```python\n{preview}\n```")
        if self.proposal_reasoning:
            lines.append(f"\nReasoning: {self.proposal_reasoning}")
        return "\n".join(lines)

    def novelty_signature(self) -> str:
        """Return a compact code-space signature for diversity-aware tree selection."""
        parts = [
            self.solution_source,
            self.initialization_source,
            self.representation_strategy,
            self.model_family,
            self.model_capacity_strategy,
            self.export_runtime_strategy,
        ]
        cleaned = [str(p or "unknown").strip().lower() for p in parts]
        return "|".join(cleaned)

    def get_imgsz(self) -> int:
        """Extract imgsz from train_code if present (for latency estimation)."""
        import re
        match = re.search(r"imgsz[=\s]*(\d+)", self.train_code)
        if match:
            return int(match.group(1))
        return 640  # default


# ---------------------------------------------------------------------------
# Default variant (used as search root when no CBR case is available)
# ---------------------------------------------------------------------------

def default_variant(
    modality: Modality,
    task_type: TaskType,
) -> SolutionVariant:
    """Return a metadata-only planning root.

    The root should not bias the first executable proposal toward a fixed model
    family/template. It is only the parent node from which the LLM plans how to
    load the observed dataset and then chooses an implementation.
    """
    return SolutionVariant(
        modality=modality,
        task_type=task_type,
        model_name="dataset_first_plan",
        model_family="planning",
        quant_mode="fp16",
        export_format="onnx",
        search_dimension="initial: dataset evidence planning",
        plan="Planning root: inspect dataset evidence before choosing model code.",
        dataset_plan="",
        train_code="",
        loader_code="",
        infer_code="",
        proposal_reasoning="Metadata-only planning root.",
    )
