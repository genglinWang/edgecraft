"""PromptSampler: builds the rich context block injected into LLM proposals.

Selection strategy (FunSearch-inspired)
----------------------------------------
1. Code lineage root → selected_node  (how the implementation evolved)
2. Top-3 trials by score              (best measured code branches)
3. 1-2 diverse failures               (negative evidence for code evolution)
4. 1 CBR local run-history case       (measured prior solution/failure evidence)
5. UserSpec + RuntimeConfig           (goal anchoring + feasibility constraints)
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Dict, List, Optional

from edgecraft.agent.search.refinement_tree import RefinementTree, TreeNode
from edgecraft.agent.search.trial_bank import TrialBank
from edgecraft.agent.search.trial_dossier import build_trial_dossier, trial_dossier_to_prompt_text
from edgecraft.agent.search.verification import (
    build_environment_fingerprint,
    runtime_package_version,
)
from edgecraft.core.task import RuntimeConfig, UserSpec


@dataclass
class PromptContext:
    """Structured context ready for insertion into a prompt template."""
    raw_intent_block: str
    task_block: str
    dataset_block: str
    constraints_block: str
    search_space_block: str
    tree_path_block: str
    parent_dossier_block: str
    top_trials_block: str
    run_evidence_block: str
    failure_block: str
    cbr_block: str
    runtime_block: str
    compatibility_memory_block: str
    search_dimension_block: str

    def to_text(self) -> str:
        sections = [
            ("RAW USER INTENT", self.raw_intent_block),
            ("TASK", self.task_block),
            ("DATASET EVIDENCE AND FORMAT CONTRACT", self.dataset_block),
            ("HARD CONSTRAINTS", self.constraints_block),
            ("CODE-SPACE DIMENSIONS EXPLORED", self.search_dimension_block),
            ("CODE EVOLUTION PATH (root → current node)", self.tree_path_block),
            ("SELECTED PARENT TRIAL DOSSIER", self.parent_dossier_block),
            ("TOP PAST CODE VARIANTS", self.top_trials_block),
            ("LONG-RUN EVIDENCE SUMMARY", self.run_evidence_block),
            ("NOTABLE FAILED CODE BRANCHES", self.failure_block),
            ("LOCAL RUN HISTORY (CBR)", self.cbr_block),
            ("RUNTIME ENVIRONMENT", self.runtime_block),
            ("VERIFIED DEVICE/RUNTIME COMPATIBILITY MEMORY", self.compatibility_memory_block),
        ]
        parts = []
        for title, body in sections:
            if body.strip():
                parts.append(f"=== {title} ===\n{body.strip()}")
        return "\n\n".join(parts)


class PromptSampler:
    """Assembles a PromptContext for the ProposalGenerator LLM call."""

    def sample(
        self,
        trial_bank: TrialBank,
        search_tree: RefinementTree,
        selected_node_id: str,
        user_intent: str,
        user_spec: Optional[UserSpec],
        runtime_config: Optional[RuntimeConfig],
        cbr_case_text: str = "",
        dataset_info: Optional[Dict[str, Any]] = None,
        state: Optional[Dict[str, Any]] = None,
        top_k: int = 3,
        failure_k: int = 2,
    ) -> PromptContext:
        dataset_sample = (
            dataset_info.get("sample_observation")
            if isinstance(dataset_info, dict)
            else None
        )
        return PromptContext(
            raw_intent_block=self._build_raw_intent_block(user_intent),
            task_block=self._build_task_block(user_spec) if user_spec else "No UserSpec provided.",
            dataset_block=self._build_dataset_block(dataset_info),
            constraints_block=self._build_constraints_block(user_spec) if user_spec else "No constraints.",
            search_space_block="",  # Now provided by ModelRegistry in ProposalGenerator
            tree_path_block=self._build_tree_path_block(
                search_tree, selected_node_id, trial_bank, dataset_sample=dataset_sample
            ),
            parent_dossier_block=self._build_parent_dossier_block(
                trial_bank, selected_node_id, state
            ),
            top_trials_block=self._build_top_trials_block(trial_bank, top_k, dataset_sample=dataset_sample),
            run_evidence_block=self._build_run_evidence_block(
                trial_bank, search_tree, selected_node_id, dataset_sample=dataset_sample
            ),
            failure_block=self._build_failure_block(trial_bank, failure_k, dataset_sample=dataset_sample),
            cbr_block=cbr_case_text or "No local run-history case retrieved.",
            runtime_block=self._build_runtime_block(runtime_config),
            compatibility_memory_block=self._build_compatibility_memory_block(runtime_config, state),
            search_dimension_block=self._build_search_dimension_block(search_tree),
        )

    def _build_compatibility_memory_block(
        self,
        runtime_config: Optional[RuntimeConfig],
        state: Optional[Dict[str, Any]],
    ) -> str:
        if state is not None:
            state["retrieved_rule_ids"] = []
        if not runtime_config or not state:
            return ""
        try:
            from edgecraft.knowledge.compatibility import get_compatibility_rule_store

            store = get_compatibility_rule_store()
            rules = []
            for runtime in runtime_config.available_runtimes:
                env_fp = build_environment_fingerprint(
                    device_id=str(state.get("target_device", "")),
                    runtime=str(runtime),
                    runtime_config=runtime_config,
                )
                rules.extend(
                    store.rules_for_environment(
                        environment_fingerprint=env_fp,
                        device=str(state.get("target_device", "")),
                        runtime=str(runtime),
                        version=runtime_package_version(runtime_config, str(runtime)),
                        limit=6,
                    )
                )
            unique = {rule.id: rule for rule in rules}
            state["retrieved_rule_ids"] = sorted(unique)[:8]
            if not unique:
                return "No verified compatibility rules matched this runtime environment."
            lines = [
                "These are verified physical facts. Use them as proposal evidence; exact artifact matching is handled by the verifier."
            ]
            lines.extend(json.dumps(rule.to_prompt_dict(), ensure_ascii=False) for rule in list(unique.values())[:8])
            return "\n".join(lines)
        except Exception:
            state["retrieved_rule_ids"] = []
            return ""

    def _build_raw_intent_block(self, user_intent: str) -> str:
        intent = (user_intent or "").strip()
        return intent or "No raw user intent provided."

    # ------------------------------------------------------------------
    # Block builders
    # ------------------------------------------------------------------

    def _build_task_block(self, user_spec: UserSpec) -> str:
        lines = [
            f"Intent: {user_spec.description}",
            f"Modality: {user_spec.input_type.value if user_spec.input_type else 'unknown'}",
            f"Task type: {user_spec.task_type.value if user_spec.task_type else 'unknown'}",
        ]
        if user_spec.preferences:
            pref = user_spec.preferences[0]
            lines.append(
                f"Primary optimisation goal: {pref.direction} {pref.metric}"
            )
        if user_spec.eval_metrics:
            lines.append(f"Tracked metrics: {', '.join(user_spec.eval_metrics)}")
        return "\n".join(lines)

    def _build_constraints_block(self, user_spec: UserSpec) -> str:
        if not user_spec.constraints:
            return "No hard constraints specified."
        lines = []
        for c in user_spec.constraints:
            unit = f" {c.unit}" if c.unit else ""
            op = {"lte": "≤", "gte": "≥", "eq": "="}.get(c.comparison, c.comparison)
            lines.append(f"  {c.metric} {op} {c.target}{unit}  [HARD CONSTRAINT]")
        return "\n".join(lines)

    def _build_parent_dossier_block(
        self,
        trial_bank: TrialBank,
        selected_node_id: str,
        state: Optional[Dict[str, Any]],
    ) -> str:
        """Render the selected parent using the same dossier evidence as BranchJudgment."""
        if not state:
            return ""
        trial = trial_bank.get(selected_node_id)
        if not trial:
            return ""
        dossier = build_trial_dossier(
            trial,
            state,
            score=float(getattr(trial, "score", 0.0) or 0.0),
            is_feasible=bool(getattr(trial, "is_feasible", False)),
            constraint_violations=list(getattr(trial, "constraint_violations", []) or []),
        )
        evidence_ids = [
            str(item.get("id"))
            for item in (dossier.verification.get("evidence") or [])
            if isinstance(item, dict) and item.get("id")
        ]
        evidence_index = (
            "CITABLE PARENT EVIDENCE IDS: " + ", ".join(evidence_ids) + "\n"
            if evidence_ids
            else ""
        )
        return evidence_index + trial_dossier_to_prompt_text(dossier, limit=24000)

    def _build_dataset_block(self, dataset_info: Optional[Dict[str, Any]]) -> str:
        if not dataset_info:
            return "No dataset evidence available."
        lines = []
        modality = dataset_info.get("modality") or "unknown"
        task_type = dataset_info.get("task_type") or "unknown"
        fmt = dataset_info.get("format") or "unknown"
        lines.append("AUTHORITATIVE DATASET FACTS (use these exact values in DATASET_PLAN and loader.py):")
        facts = self._build_authoritative_dataset_facts(dataset_info)
        if facts:
            lines.extend(f"  {line}" for line in facts)
        lines.append(f"Analyzer modality: {modality}")
        lines.append(f"Analyzer task_type: {task_type}")
        lines.append(f"Analyzer format: {fmt}")
        config_path = dataset_info.get("config_path")
        if config_path:
            lines.append(f"Resolved config/data source: {config_path}")
        description = str(dataset_info.get("description") or "").strip()
        if description:
            lines.append(f"Summary: {description[:700]}")
        structure = str(dataset_info.get("structure_summary") or "").strip()
        if structure:
            lines.append(f"Structure: {structure[:900]}")
        rec = dataset_info.get("recommended_config") or {}
        if isinstance(rec, dict):
            if rec.get("key_path"):
                lines.append(f"Key path/manifest: {rec.get('key_path')}")
            if rec.get("notes"):
                lines.append(f"Loader notes: {str(rec.get('notes'))[:900]}")
        warnings = dataset_info.get("preflight_warnings")
        if isinstance(warnings, list) and warnings:
            lines.append(f"Preflight warnings: {warnings[:5]}")
        provenance = dataset_info.get("provenance_notes")
        if isinstance(provenance, list) and provenance:
            lines.append(
                "Provenance notes (not allowed as replacement training data): "
                f"{[str(note)[:300] for note in provenance[:3]]}"
            )
        splits = dataset_info.get("splits")
        if splits:
            lines.append(f"Splits: {splits}")
        split_manifest = dataset_info.get("split_manifest")
        if isinstance(split_manifest, dict):
            lines.append(
                "Immutable split manifest (read this file; do not infer its schema): "
                + json.dumps(split_manifest, ensure_ascii=False)[:2400]
            )
            lines.append(
                "Split manifest IDs are evaluator-owned sample identifiers. Treat them as opaque: "
                "use the manifest examples, raw sample evidence, and label_join_key to map them to "
                "source rows; do not coerce or strip namespaces without evidence, and verify every "
                "requested split is non-empty."
            )
        sample_observation = dataset_info.get("sample_observation")
        raw_label_evidence = self._raw_label_evidence(sample_observation)
        if raw_label_evidence:
            lines.append(raw_label_evidence)
        column_value_evidence = self._column_value_evidence(sample_observation)
        if column_value_evidence:
            lines.append(column_value_evidence)
        classes = dataset_info.get("classes")
        if classes:
            lines.append(f"Classes/labels: {classes[:20]}")
        if isinstance(sample_observation, dict):
            excerpt = json.dumps(sample_observation, ensure_ascii=False)[:1800]
            lines.append(f"Sample-level observation: {excerpt}")
            lines.append(
                "When a prose summary conflicts with the deterministic sample observation, trust "
                "the sample observation. In particular, preserve observed shapes, delimiter counts, "
                "raw labels, and input types in loader.py."
            )
        path_facts = dataset_info.get("path_facts")
        if isinstance(path_facts, dict):
            excerpt = json.dumps(path_facts, ensure_ascii=False)[:2200]
            lines.append(f"Path/manifest facts: {excerpt}")
        report = str(dataset_info.get("exploration_report") or "").strip()
        if report:
            lower = report.lower()
            format_hints = []
            for token in (".jsonl", ".arrow", ".parquet", ".csv", ".tsv", ".arff", ".ts", ".wav", ".yaml"):
                if token in lower:
                    format_hints.append(token)
            if format_hints:
                lines.append(f"Observed file-format hints: {', '.join(sorted(set(format_hints)))}")
            lines.append(f"Exploration report excerpt: {report[:1200]}")
        lines.append(
            "Use this dataset evidence as authoritative for loader/code generation. "
            "Do not choose ImageFolder/YOLO/vision loaders unless the evidence explicitly "
            "shows image data with a compatible label contract."
        )
        return "\n".join(lines)

    def _raw_label_evidence(self, sample_observation: Any) -> str:
        if not isinstance(sample_observation, dict):
            return ""
        label_summary = sample_observation.get("label_summary")
        if not isinstance(label_summary, dict):
            return ""
        label_counts = label_summary.get("label_counts")
        if not isinstance(label_counts, dict) or not label_counts:
            return ""
        raw_values = [str(key) for key in list(label_counts.keys())[:12]]
        label_key = sample_observation.get("label_key") or "unknown"
        return (
            "Raw label evidence from sampled rows: "
            f"label_key={label_key}, observed_values={raw_values}. "
            "Use exact raw file values for code mappings; display class names may be descriptive."
        )

    def _column_value_evidence(self, sample_observation: Any) -> str:
        if not isinstance(sample_observation, dict):
            return ""
        samples = sample_observation.get("column_value_samples")
        if not isinstance(samples, dict) or not samples:
            return ""
        compact = {str(key): value[:5] for key, value in list(samples.items())[:8] if isinstance(value, list)}
        related = []
        for item in list(sample_observation.get("related_file_observations") or [])[:6]:
            if not isinstance(item, dict):
                continue
            values = item.get("column_value_samples")
            if not isinstance(values, dict):
                continue
            related.append(
                {
                    "source_file": item.get("source_file"),
                    "values": {str(k): v[:4] for k, v in list(values.items())[:6] if isinstance(v, list)},
                }
            )
        if not compact and not related:
            return ""
        return (
            "Observed raw column values from sampled files: "
            f"{json.dumps({'primary': compact, 'related_files': related}, ensure_ascii=False)[:1600]}. "
            "Use these values to avoid semantic guesses about metadata columns."
        )

    def _build_authoritative_dataset_facts(self, dataset_info: Dict[str, Any]) -> List[str]:
        facts: List[str] = []
        for key in ("data_root", "root", "dataset_path", "dataset_root", "config_path", "file_format"):
            value = dataset_info.get(key)
            if value:
                facts.append(f"{key}: {value}")
        if dataset_info.get("data_root") or dataset_info.get("root"):
            facts.append(
                "path semantics: data_root/root is the analyzer-resolved data directory; "
                "dataset_path is the original submission boundary"
            )
        schema = dataset_info.get("schema") if isinstance(dataset_info.get("schema"), dict) else {}
        files = dataset_info.get("files") or schema.get("files") or []
        if isinstance(files, (str, bytes)):
            files = [str(files)]
        if files:
            facts.append("schema.files exact paths:")
            for item in list(files)[:8]:
                facts.append(f"  - {item}")
            if len(files) > 8:
                facts.append(f"  ... {len(files) - 8} more")
        columns = schema.get("columns") or dataset_info.get("columns") or []
        if columns:
            facts.append("schema.columns: " + ", ".join(str(c) for c in list(columns)[:30]))
        for key in ("label_column", "text_column", "split_column", "label_join_key"):
            value = dataset_info.get(key) or schema.get(key)
            if value is not None:
                facts.append(f"{key}: {value}")
        label_files = dataset_info.get("label_files") or []
        if label_files:
            facts.append("label_files exact paths:")
            for item in list(label_files)[:5]:
                facts.append(f"  - {item}")
        if dataset_info.get("supervised_label_hint"):
            facts.append("supervised_label_hint: " + str(dataset_info["supervised_label_hint"]))
        path_facts = dataset_info.get("path_facts")
        if isinstance(path_facts, dict):
            config_files = path_facts.get("config_files") or []
            if config_files:
                facts.append("path_facts.config_files:")
                for item in list(config_files)[:4]:
                    if isinstance(item, dict):
                        facts.append(
                            "  - "
                            + json.dumps(
                                {
                                    "path": item.get("path"),
                                    "raw_path": item.get("raw_path"),
                                    "resolved_kind": item.get("resolved_kind"),
                                    "splits": item.get("splits"),
                                },
                                ensure_ascii=False,
                            )[:500]
                        )
            manifests = path_facts.get("manifest_files") or []
            if manifests:
                facts.append("path_facts.manifest_files:")
                for item in list(manifests)[:6]:
                    if isinstance(item, dict):
                        facts.append(
                            "  - "
                            + json.dumps(
                                {
                                    "path": item.get("path"),
                                    "kind": item.get("kind"),
                                    "sample_lines": item.get("sample_lines"),
                                    "sample_path_checks": item.get("sample_path_checks"),
                                },
                                ensure_ascii=False,
                            )[:500]
                        )
        return facts

    def _build_tree_path_block(
        self,
        search_tree: RefinementTree,
        selected_node_id: str,
        trial_bank: TrialBank,
        dataset_sample: Optional[Dict[str, Any]] = None,
    ) -> str:
        path: List[TreeNode] = search_tree.get_path_to_root(selected_node_id)
        if not path:
            return "Empty tree (first proposal)."

        lines = []
        for depth, node in enumerate(path):
            trial = trial_bank.get(node.node_id)
            prefix = "→ " * depth + ("[current]" if node.node_id == selected_node_id else "")
            variant_str = node.variant.short_description()
            if node.is_evaluated and trial:
                feasibility = "FEASIBLE" if trial.is_feasible else "INFEASIBLE"
                progress = getattr(trial, "crafting_progress", "")
                progress_str = f" progress={progress.value if hasattr(progress, 'value') else progress}" if progress else ""
                mutation_str = self._variant_lineage_summary(node.variant)
                lat_str = ""
                mem_str = ""
                if trial.edge_metrics:
                    if trial.edge_metrics.latency_ms is not None:
                        lat_str = f"  lat={trial.edge_metrics.latency_ms:.1f}ms"
                    if trial.edge_metrics.memory_mb is not None:
                        mem_str = f"  mem={trial.edge_metrics.memory_mb:.0f}MB"

                # Show metrics from all_metrics
                metric_str = ""
                if trial.local_metrics and trial.local_metrics.all_metrics:
                    # Show first 2 metrics
                    items = list(trial.local_metrics.all_metrics.items())[:2]
                    metric_str = "  " + ", ".join(f"{k}={v:.4f}" for k, v in items)

                lines.append(
                    f"depth={depth} {prefix} {variant_str}"
                    f"{mutation_str}{metric_str}{lat_str}{mem_str}"
                    f"{progress_str}  [{feasibility}]"
                )
                evidence = self._trial_evidence_summary(trial, dataset_sample=dataset_sample)
                if evidence:
                    lines.append(f"  evidence: {evidence}")
            else:
                mutation_str = self._variant_lineage_summary(node.variant)
                lines.append(f"depth={depth} {prefix} {variant_str}{mutation_str}  [not evaluated]")

        # Siblings of the selected node
        siblings = search_tree.get_siblings(selected_node_id)
        if siblings:
            lines.append("\nSiblings of current node:")
            for sib in siblings:
                sib_trial = trial_bank.get(sib.node_id)
                if sib_trial:
                    lines.append(f"  {self._trial_compact_evidence(sib_trial, dataset_sample=dataset_sample)}")

        return "\n".join(lines)

    def _build_top_trials_block(
        self,
        trial_bank: TrialBank,
        k: int,
        dataset_sample: Optional[Dict[str, Any]] = None,
    ) -> str:
        top = trial_bank.get_top_k(k=k)
        if not top:
            return "No trials completed yet."
        lines = [f"Top {len(top)} trials by score:"]
        for t in top:
            lines.append(f"  {self._trial_compact_evidence(t, dataset_sample=dataset_sample)}")
        return "\n".join(lines)

    def _build_run_evidence_block(
        self,
        trial_bank: TrialBank,
        search_tree: RefinementTree,
        selected_node_id: str,
        dataset_sample: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Summarize long-run evidence without creating a second control system."""
        trials = trial_bank.get_all()
        if not trials:
            return "No long-run evidence yet."

        lines = [f"Trials recorded: {len(trials)}"]
        selected = search_tree.get_node(selected_node_id)
        planning_root_selected = bool(
            selected
            and getattr(selected.variant, "model_family", "") == "planning"
        )
        if planning_root_selected and selected.children_ids:
            lines.append(
                "INITIAL BREADTH SLOT: the selected planning root has no executable "
                "code to repair. A repair of an existing executable sibling belongs "
                "under that sibling, not under the planning root."
            )
            lines.append("Existing root solution hypotheses already tested:")
            for child_id in selected.children_ids[:8]:
                child = search_tree.get_node(child_id)
                if child is None:
                    continue
                variant = child.variant
                lines.append(
                    "  - "
                    f"trial={child_id} model={variant.model_name} "
                    f"representation={variant.representation_strategy or 'unspecified'} "
                    f"initialization={variant.initialization_source or 'unspecified'} "
                    f"recipe={variant.training_recipe or 'unspecified'}"
                )
            lines.append(
                "Generate one complete solution hypothesis materially distinct from "
                "those root siblings; use their failures as evidence, not as a repair directive."
            )
        best = max(trials, key=lambda t: float(t.score or 0.0))
        lines.append("Best lineage by measured score:")
        for item in self._lineage_trials(best, trial_bank):
            lines.append(f"  - {self._trial_one_line(item)}")

        lines.append("Recent metric/latency trajectory:")
        for idx, trial in enumerate(trials[-8:], start=max(1, len(trials) - 7)):
            lines.append(f"  #{idx}: {self._trial_one_line(trial)}")

        siblings = []
        try:
            siblings = search_tree.get_siblings(selected_node_id)
        except Exception:  # noqa: BLE001
            siblings = []
        if siblings:
            lines.append("Sibling comparison for selected parent:")
            for node in siblings[:6]:
                trial = trial_bank.get(node.node_id)
                if trial:
                    lines.append(f"  - {self._trial_one_line(trial)}")

        weak = [
            t for t in trials
            if getattr(getattr(t, "branch_judgment", None), "judgment", "")
            in {"weak_branch", "dead_end"}
        ]
        if weak:
            lines.append("Branches judged weak/dead; keep as negative evidence, do not repeat blindly:")
            for trial in weak[-5:]:
                bj = getattr(trial, "branch_judgment", None)
                reason = (getattr(bj, "reasoning", "") or "")[:180]
                lines.append(f"  - {self._trial_one_line(trial)} reason={reason}")

        working = [t for t in trials if getattr(t, "edge_metrics", None) and getattr(t, "artifact_contract", None)]
        if working:
            lines.append("Working components worth preserving unless evidence says otherwise:")
            lines.append("Observed component inventory:")
            for trial in sorted(working, key=lambda t: float(t.score or 0.0), reverse=True)[:5]:
                lines.append(f"  - {self._component_inventory_line(trial, dataset_sample=dataset_sample)}")

        repeated = self._repeated_failure_motifs(trials)
        if repeated:
            lines.append("Repeated bad mutation/failure motifs observed as evidence:")
            for motif, count in repeated:
                lines.append(f"  - {motif}: {count} trial(s)")

        if planning_root_selected:
            lines.append(
                "The planning root contributes dataset and runtime evidence only; it has no "
                "working implementation to inherit."
            )
        else:
            lines.append(
                "Next child should cite this evidence: inherit working loader/artifact/export/infer "
                "routes from the selected parent by default, and explain any deliberate rewrite as "
                "a measured code-space mutation."
            )
        return "\n".join(lines)

    def _component_inventory_line(self, trial, dataset_sample: Optional[Dict[str, Any]] = None) -> str:
        variant = getattr(trial, "variant", None)
        changed = getattr(variant, "changed_components", []) or []
        inherited = getattr(variant, "inherited_components", []) or []
        note = "loader/train/export/infer reached edge"
        runtime_report = getattr(trial, "runtime_report", None)
        provider = (getattr(runtime_report, "runtime_provider", "") or "").lower()
        used = (getattr(runtime_report, "runtime_used", "") or "").lower()
        if used == "torch" or "fallback" in provider:
            note = "fallback runtime observed; next mutation should fix artifact/infer runtime before changing model capacity"
        elif getattr(trial, "is_feasible", False):
            note = "feasible component path; preserve loader/artifact/runtime unless replacing deliberately"
        sample_obs = self._trial_sample_observation(trial, dataset_sample=dataset_sample)
        sample_note = ""
        if sample_obs:
            label = sample_obs.get("label_summary") or {}
            input_summary = sample_obs.get("input_summary") or {}
            sample_note = (
                f" sample=n={sample_obs.get('num_observed_samples')},"
                f" input={input_summary}, labels={label}"
            )
        return (
            f"{self._trial_one_line(trial)} "
            f"inherits={', '.join(inherited[:3]) or 'n/a'} "
            f"changes={', '.join(changed[:3]) or 'n/a'} "
            f"note={note}{sample_note}"
        )

    def _lineage_trials(self, trial, trial_bank: TrialBank):
        by_id = {t.trial_id: t for t in trial_bank.get_all()}
        lineage = []
        current = trial
        seen = set()
        while current and current.trial_id not in seen:
            lineage.append(current)
            seen.add(current.trial_id)
            parent_id = getattr(getattr(current, "variant", None), "parent_trial_id", "")
            current = by_id.get(parent_id)
        return list(reversed(lineage))

    def _trial_one_line(self, trial) -> str:
        variant = getattr(trial, "variant", None)
        model = getattr(variant, "model_name", "unknown") if variant else "unknown"
        mutation = getattr(variant, "mutation_type", "") if variant else ""
        metric = self._metric_one_line(trial)
        latency = ""
        if getattr(trial, "edge_metrics", None) and trial.edge_metrics.latency_ms is not None:
            latency = f" lat={trial.edge_metrics.latency_ms:.3g}ms"
        runtime = self._runtime_one_line(trial)
        validity = self._validity_one_line(trial)
        judgment = getattr(getattr(trial, "branch_judgment", None), "judgment", "")
        progress = getattr(getattr(trial, "crafting_progress", None), "value", "")
        return (
            f"{trial.trial_id} model={model} mutation={mutation or 'n/a'} "
            f"score={float(trial.score or 0.0):.4f} {metric}{latency}{runtime}{validity} "
            f"progress={progress} judgment={judgment or 'n/a'}"
        )

    def _validity_one_line(self, trial) -> str:
        observations = getattr(trial, "observations", {}) or {}
        if not isinstance(observations, dict):
            return ""
        validity = observations.get("evaluation_validity")
        contract = observations.get("edge_eval_bundle_contract")
        parity = observations.get("edge_quality_parity")
        facts = [item for item in (validity, contract, parity) if isinstance(item, dict)]
        if not facts:
            return ""
        if isinstance(validity, dict):
            value = validity.get("evaluation_valid")
            status = "true" if value is True else "false" if value is False else "unknown"
            reasons = [str(item) for item in (validity.get("reasons") or []) if item]
        else:
            bundle_failed = isinstance(contract, dict) and contract.get("status") == "fail"
            parity_failed = isinstance(parity, dict) and parity.get("status") == "fail"
            status = "false" if bundle_failed or parity_failed else "unknown"
            reasons = []
            if bundle_failed:
                reasons.append("evaluation_bundle_invalid")
            if parity_failed:
                reasons.append("local_edge_parity_unproven")
        evidence_ids = []
        for item, keys in (
            (validity, ("contract_evidence_id", "parity_evidence_id")),
            (contract, ("evidence_id",)),
            (parity, ("evidence_id",)),
        ):
            if not isinstance(item, dict):
                continue
            evidence_ids.extend(str(item[key]) for key in keys if item.get(key))
        evidence_ids = list(dict.fromkeys(evidence_ids))
        suffix = f" eval_valid={status}"
        if reasons:
            suffix += " eval_reasons=" + ",".join(reasons[:3])
        if evidence_ids:
            suffix += " eval_evidence=" + ",".join(evidence_ids)
        return suffix

    def _runtime_one_line(self, trial) -> str:
        runtime_report = getattr(trial, "runtime_report", None)
        if not runtime_report or not getattr(runtime_report, "runtime_used", None):
            return ""
        provider = getattr(runtime_report, "runtime_provider", None) or "unknown_provider"
        artifact = getattr(runtime_report, "artifact_used", None) or ""
        text = f" rt={runtime_report.runtime_used}/{provider}"
        if artifact:
            text += f" artifact={artifact}"
        return text

    def _metric_one_line(self, trial) -> str:
        local = getattr(trial, "local_metrics", None)
        metrics = getattr(local, "all_metrics", None) or {}
        if not metrics:
            return "metric=n/a"
        items = list(metrics.items())[:2]
        return "metric=" + ",".join(
            f"{k}={v:.4g}" if isinstance(v, (int, float)) else f"{k}={v}"
            for k, v in items
        )

    def _repeated_failure_motifs(self, trials) -> List[tuple[str, int]]:
        counts: Dict[str, int] = {}
        for trial in trials:
            variant = getattr(trial, "variant", None)
            mutation = getattr(variant, "mutation_type", "") or "unknown_mutation"
            taxonomy = getattr(trial, "failure_taxonomy", None) or "unknown_failure"
            if taxonomy == "unknown_failure" and getattr(trial, "edge_metrics", None):
                continue
            key = f"{mutation}/{taxonomy}"
            counts[key] = counts.get(key, 0) + 1
            observations = getattr(trial, "observations", {}) or {}
            for hit in observations.get("compatibility_hits", []) or []:
                if not isinstance(hit, dict):
                    continue
                if hit.get("status") != "verified" or hit.get("visibility") != "tenant_public":
                    continue
                rule_id = str(hit.get("rule_id") or "").strip()
                if rule_id:
                    rule_key = f"verified_rule:{rule_id}"
                    counts[rule_key] = counts.get(rule_key, 0) + 1
        return sorted(
            [(k, v) for k, v in counts.items() if v >= 2],
            key=lambda item: (-item[1], item[0]),
        )[:6]

    def _build_failure_block(
        self,
        trial_bank: TrialBank,
        k: int,
        dataset_sample: Optional[Dict[str, Any]] = None,
    ) -> str:
        failures = trial_bank.get_failures(k=k)
        if not failures:
            return "No notable failures yet."
        lines = [f"Notable failures (avoid these directions):"]
        for t in failures:
            lines.append(f"  {self._trial_compact_evidence(t, dataset_sample=dataset_sample)}")
        return "\n".join(lines)

    def _build_runtime_block(self, runtime_config: Optional[RuntimeConfig]) -> str:
        if runtime_config is None:
            return "Runtime config not available."
        cloud = runtime_config.cloud_buildable_formats()
        targets = runtime_config.deploy_target_formats()
        lines = [
            f"Docker image (edge): {runtime_config.docker_image}",
            f"Edge GPU available: {runtime_config.has_gpu}",
            f"Cloud-buildable formats (what train.py may export): {', '.join(cloud)}",
            f"Deploy-target formats (what infer.py may load on edge): {', '.join(targets)}",
            "Engine policy: TRT engines are NEVER built on the cloud — when "
            "export_format=engine, the edge runner compiles outputs/best.onnx "
            "into outputs/best.engine on the target device before infer.py runs.",
            "Edge staging budget: all required model files, runtime sidecars, and "
            f"edge evaluation payloads combined must fit within {runtime_config.edge_artifact_bundle_max_mb} MiB. "
            "Keep the evaluation payload representative but bounded; do not copy "
            "the full evaluation dataset when a smaller same-provenance sample is sufficient.",
        ]
        if runtime_config.python_version:
            lines.append(f"Edge Python version: {runtime_config.python_version}")
        if runtime_config.trt_version:
            lines.append(f"Edge TensorRT version: {runtime_config.trt_version}")
        if runtime_config.onnxruntime_version:
            lines.append(f"Edge ONNXRuntime version: {runtime_config.onnxruntime_version}")
        providers = getattr(runtime_config, "onnxruntime_providers", []) or []
        if providers:
            lines.append("Edge ONNXRuntime providers: " + ", ".join(providers))
        cloud_packages = getattr(runtime_config, "cloud_python_packages", {}) or {}
        if cloud_packages:
            available = [
                f"{name}={status}"
                for name, status in sorted(cloud_packages.items())
                if not str(status).startswith("unavailable")
            ]
            unavailable = [
                name
                for name, status in sorted(cloud_packages.items())
                if str(status).startswith("unavailable")
            ]
            if available:
                lines.append("Cloud/server train.py Python packages available: " + ", ".join(available))
            if unavailable:
                lines.append("Cloud/server train.py Python packages unavailable: " + ", ".join(unavailable))
        packages = getattr(runtime_config, "python_packages", {}) or {}
        runtime_packages = getattr(runtime_config, "runtime_python_packages", {}) or {}
        for runtime_name, package_map in sorted(runtime_packages.items()):
            available = [
                name
                for name, status in sorted(package_map.items())
                if not str(status).startswith("unavailable")
            ]
            unavailable = [
                name
                for name, status in sorted(package_map.items())
                if str(status).startswith("unavailable")
            ]
            lines.append(
                f"Edge native {runtime_name} environment packages available: "
                + (", ".join(available) or "none probed")
            )
            if unavailable:
                lines.append(
                    f"Edge native {runtime_name} environment packages unavailable: "
                    + ", ".join(unavailable)
                )
        if packages:
            available = [
                f"{name}={status}"
                for name, status in sorted(packages.items())
                if not str(status).startswith("unavailable")
            ]
            unavailable = [
                name
                for name, status in sorted(packages.items())
                if str(status).startswith("unavailable")
            ]
            if available:
                lines.append("Edge Docker Python packages available: " + ", ".join(available))
            if unavailable:
                lines.append("Edge Docker Python packages unavailable: " + ", ".join(unavailable))
        if cloud_packages or packages or runtime_packages:
            lines.append(
                "Package evidence boundary: train.py runs in the cloud/server environment; "
                "infer.py runs in the selected edge Docker or native runtime environment. "
                "Import a third-party package only when it is positively listed as available "
                "in that exact runtime environment, not merely in another environment on "
                "the same device. Never install packages from generated infer.py; a managed "
                "environment change requires a new preflight snapshot."
            )
        return "\n".join(lines)

    def _build_search_dimension_block(
        self, search_tree: RefinementTree
    ) -> str:
        """Summarise which search dimensions have been explored so far.

        Helps the LLM balance diversity: if most variants fall under
        'architecture' but 'quantization' is under-explored, the LLM
        can be nudged toward it.
        """
        dist = search_tree.layer_distribution()
        if not dist:
            return "No search dimensions recorded yet."
        lines = ["Search dimension coverage (guide your proposal toward diversity):"]
        for dim, count in sorted(dist.items(), key=lambda x: -x[1]):
            lines.append(f"  {dim}: {count} variant(s)")
        lines.append(
            "\nConsider exploring under-represented code-space dimensions to diversify the iterative tree search."
        )
        return "\n".join(lines)

    def _variant_lineage_summary(self, variant) -> str:
        parts = []
        mutation = getattr(variant, "mutation_type", "") or ""
        if mutation and mutation != "initial":
            parts.append(f"mutation={mutation}")
        inherited = getattr(variant, "inherited_components", []) or []
        if inherited:
            parts.append(f"inherits={', '.join(inherited[:3])}")
        changed = getattr(variant, "changed_components", []) or []
        if changed:
            parts.append(f"changes={', '.join(changed[:3])}")
        hypothesis = getattr(variant, "proposal_hypothesis", "") or ""
        if hypothesis:
            parts.append(f"hypothesis={hypothesis[:160]}")
        return ("  " + " | ".join(parts)) if parts else ""

    def _trial_evidence_summary(
        self,
        trial,
        dataset_sample: Optional[Dict[str, Any]] = None,
    ) -> str:
        parts = []
        contract = getattr(trial, "artifact_contract", None)
        if contract and contract.primary_artifact:
            parts.append(f"primary_artifact={contract.primary_artifact}")
            artifacts = getattr(contract, "artifacts", {}) or {}
            artifact_bits = []
            for name, info in list(artifacts.items())[:3]:
                if isinstance(info, dict):
                    artifact_bits.append(
                        f"{name}:{info.get('kind', 'unknown')}:{info.get('size_bytes', 'unknown')}B"
                    )
            if artifact_bits:
                parts.append("artifacts=" + ", ".join(artifact_bits))
        progress = getattr(trial, "crafting_progress", None)
        if progress:
            parts.append(f"progress={progress.value if hasattr(progress, 'value') else progress}")
        taxonomy = getattr(trial, "failure_taxonomy", None)
        if taxonomy:
            parts.append(f"failure_taxonomy={taxonomy}")
        branch_judgment = getattr(trial, "branch_judgment", None)
        if branch_judgment:
            parts.append(branch_judgment.to_prompt_text())
        hint = getattr(trial, "next_search_hint", None)
        if hint:
            parts.append(f"next_code_mutation_hint={hint}")
        runtime_report = getattr(trial, "runtime_report", None)
        if runtime_report and runtime_report.runtime_used:
            provider = runtime_report.runtime_provider or "unknown_provider"
            parts.append(f"runtime={runtime_report.runtime_used}/{provider}")
            artifact_used = getattr(runtime_report, "artifact_used", "") or ""
            if artifact_used:
                parts.append(f"runtime_artifact={artifact_used}")
            if "torch_load_fallback" in str(provider).lower():
                contract = getattr(trial, "artifact_contract", None)
                primary = getattr(contract, "primary_artifact", "") if contract else ""
                if primary and str(primary).endswith(".onnx"):
                    parts.append(
                        "runtime_path_issue=torch_load_fallback used despite ONNX primary artifact; "
                        "next mutation should preserve loader/model evidence and repair infer/runtime path"
                    )
        failure_context = getattr(trial, "failure_context", {}) or {}
        observations = getattr(trial, "observations", {}) or {}
        if isinstance(observations, dict):
            scientific_validity = {
                key: observations[key]
                for key in (
                    "edge_eval_bundle_contract",
                    "edge_quality_parity",
                    "edge_quality_provenance",
                    "evaluation_validity",
                )
                if isinstance(observations.get(key), dict)
            }
            if scientific_validity:
                parts.append(f"scientific_validity={scientific_validity}")
        sample_obs = self._trial_sample_observation(trial, dataset_sample=dataset_sample)
        if isinstance(sample_obs, dict):
            label = sample_obs.get("label_summary") or {}
            split_counts = sample_obs.get("split_counts") or {}
            input_summary = sample_obs.get("input_summary") or {}
            parts.append(
                "sample_observation="
                f"n={sample_obs.get('num_observed_samples')}, "
                f"input={input_summary}, "
                f"labels={label}, "
                f"splits={split_counts}"
            )
        if isinstance(failure_context, dict):
            repair_diag = failure_context.get("repair_diagnosis")
            if isinstance(repair_diag, dict):
                compact = {
                    k: repair_diag.get(k)
                    for k in ("repair_action", "diagnosis_type", "failed_component", "next_search_hint")
                    if repair_diag.get(k)
                }
                if compact:
                    parts.append(f"repair_diagnosis={compact}")
            component_roundtrip = failure_context.get("component_roundtrip")
            failure = {
                "stage": getattr(trial, "error_stage", "") or "",
                "error": str(getattr(trial, "error", "") or "")[-600:],
            }
            if isinstance(component_roundtrip, dict):
                failure["component_roundtrip"] = {
                    key: component_roundtrip.get(key)
                    for key in ("boundary", "evidence_id", "error")
                    if component_roundtrip.get(key)
                }
            debug_history = list(getattr(trial, "debug_history", []) or [])[-2:]
            if debug_history:
                parts.append(
                    "debug_history="
                    + str([
                        {
                            "stage": getattr(item, "stage", ""),
                            "error_signature": getattr(item, "error_signature", ""),
                            "patch_summary": getattr(item, "patch_summary", ""),
                            "retry_passed": bool(getattr(item, "success", False)),
                        }
                        for item in debug_history
                    ])
                )
            if failure["error"] or failure.get("component_roundtrip"):
                # Keep raw execution facts last so the compact tail cannot hide
                # the exact component contract that the next child must consume.
                parts.append(f"execution_failure={failure}")
        return self._tail("; ".join(parts), 1200)

    def _trial_sample_observation(
        self,
        trial,
        dataset_sample: Optional[Dict[str, Any]] = None,
    ) -> dict:
        observations = getattr(trial, "observations", {}) or {}
        sample_obs = observations.get("sample_observation") if isinstance(observations, dict) else {}
        if not isinstance(sample_obs, dict) or not sample_obs:
            sample_obs = self._sample_observation_from_failure_context(
                getattr(trial, "failure_context", {}) or {}
            )
        return self._more_informative_sample(sample_obs, dataset_sample)

    def _more_informative_sample(self, first: Any, second: Any) -> dict:
        candidates = [c for c in (first, second) if isinstance(c, dict) and c]
        if not candidates:
            return {}
        return max(candidates, key=self._sample_observation_strength)

    def _sample_observation_strength(self, obs: Dict[str, Any]) -> tuple[int, int, int, int]:
        label = obs.get("label_summary") if isinstance(obs.get("label_summary"), dict) else {}
        input_summary = obs.get("input_summary") if isinstance(obs.get("input_summary"), dict) else {}
        try:
            labels = int(label.get("num_labels_observed") or 0)
        except (TypeError, ValueError):
            labels = 0
        try:
            unique_labels = int(label.get("unique_labels_observed") or 0)
        except (TypeError, ValueError):
            unique_labels = 0
        try:
            samples = int(obs.get("num_observed_samples") or 0)
        except (TypeError, ValueError):
            samples = 0
        return (labels, unique_labels, samples, 1 if input_summary else 0)

    def _sample_observation_from_failure_context(self, failure_context) -> dict:
        if not isinstance(failure_context, dict):
            return {}
        loader_smoke = failure_context.get("loader_smoke")
        if isinstance(loader_smoke, dict):
            sample_obs = loader_smoke.get("sample_observation")
            if isinstance(sample_obs, dict):
                return sample_obs
            probe = loader_smoke.get("loader_return_probe")
            if isinstance(probe, dict) and isinstance(probe.get("sample_observation"), dict):
                return probe["sample_observation"]
        diag = failure_context.get("diagnostics") or {}
        if isinstance(diag, dict) and isinstance(diag.get("sample_observation"), dict):
            return diag["sample_observation"]
        repair = failure_context.get("repair_diagnosis") or {}
        if isinstance(repair, dict):
            evidence = repair.get("evidence") or {}
            if isinstance(evidence, dict) and isinstance(evidence.get("sample_observation"), dict):
                return evidence["sample_observation"]
        return {}

    def _trial_compact_evidence(
        self,
        trial,
        dataset_sample: Optional[Dict[str, Any]] = None,
    ) -> str:
        evidence = self._trial_evidence_summary(trial, dataset_sample=dataset_sample)
        if evidence:
            return f"{self._trial_one_line(trial)} | evidence={self._tail(evidence, 700)}"
        return self._trial_one_line(trial)

    def _tail(self, text: str, limit: int) -> str:
        if not text or len(text) <= limit:
            return text
        return "..." + text[-limit:]
