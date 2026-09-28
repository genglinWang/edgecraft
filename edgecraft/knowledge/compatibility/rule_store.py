"""Device/runtime compatibility memory.

Raw failures and verified rules are deliberately different records.  A repeated
failure is useful evidence, but it never gains authority to stop a candidate
until a minimal reproduction fails in the same runtime environment.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from edgecraft.config.settings import settings


def _now() -> str:
    return datetime.now().isoformat()


def error_fingerprint(text: str) -> str:
    """Stable error identity independent of runner and JSON log wrappers."""
    raw = re.sub(r"\\+n", "\n", str(text or ""))
    raw = re.sub(r'\\+"', '"', raw)
    raw_lines = [line.strip() for line in raw.splitlines() if line.strip()]
    strong_markers = (
        "unsupported",
        "not supported",
        "not registered",
        "no checker",
        "no importer",
    )
    diagnostic = [
        line for line in raw_lines
        if any(marker in line.lower() for marker in strong_markers)
    ]
    minimal_markers = ("unsupported", "no checker", "no importer", "not registered")
    minimal_diagnostic = []
    for line in diagnostic:
        lowered = line.lower()
        positions = [lowered.index(marker) for marker in minimal_markers if marker in lowered]
        if positions:
            line = line[min(positions):]
        minimal_diagnostic.append(line)
    diagnostic = minimal_diagnostic
    if not diagnostic:
        diagnostic = [line for line in raw_lines if "error" in line.lower() or "failed" in line.lower()]
    normalized = "\n".join(diagnostic or raw_lines).lower()
    normalized = re.sub(r"^\[[^\]]+\]\s*", "", normalized, flags=re.M)
    normalized = re.sub(r"0x[0-9a-f]+", "<addr>", normalized)
    normalized = re.sub(r"(?:/[^\s:]+)+", "<path>", normalized)
    normalized = re.sub(
        r"\b(?:[\w.-]+/)*[\w.-]+\.(?:onnx|engine|pt|pth|tflite|torchscript)\b",
        "<artifact>",
        normalized,
    )
    normalized = re.sub(r"\d+(?:\.\d+)?", "<n>", normalized)
    normalized_lines = sorted({re.sub(r"\s+", " ", line).strip() for line in normalized.splitlines() if line.strip()})
    normalized = "\n".join(normalized_lines)[:1200]
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _signature_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _tenant_scope_key(tenant_id: str) -> str:
    """Opaque storage scope for tenant-private failure observations."""
    tenant = str(tenant_id or "default").strip() or "default"
    return "tenant_" + hashlib.sha256(tenant.encode("utf-8")).hexdigest()[:16]


def _runtime_key(value: Any) -> str:
    key = str(value or "").strip().lower()
    return {
        "onnxruntime": "onnx",
        "ort": "onnx",
        "tensorrt": "engine",
        "trt": "engine",
        "litert": "tflite",
        "ai_edge_litert": "tflite",
        "tflite_runtime": "tflite",
        "pytorch": "torch",
        "torchscript": "torch",
        "pt": "torch",
        "pth": "torch",
    }.get(key, key)


def _environment_scope_matches(
    rule: "CompatibilityRule",
    *,
    device: str,
    runtime: str,
    version: str,
    environment_fingerprint: str,
) -> bool:
    """Require the audited environment hash when a rule carries one.

    Verified rules are created with an environment fingerprint and may hard
    gate only in that exact environment. The structured identity fallback is
    retained solely for reading legacy, non-gating records that predate the
    fingerprint field.
    """
    if rule.environment_fingerprint:
        return bool(
            environment_fingerprint
            and rule.environment_fingerprint == environment_fingerprint
        )
    return bool(
        rule.device
        and device
        and rule.device == device
        and rule.runtime
        and runtime
        and _runtime_key(rule.runtime) == _runtime_key(runtime)
        and rule.version_range
        and version
        and rule.version_range == version
    )


_RUNTIME_DIAGNOSTIC_MARKERS = {
    "onnx": (
        "[onnxruntimeerror]",
        "onnxruntime_pybind11_state",
        "onnxruntime:",
        "onnx.version_converter",
        "onnx checker",
    ),
    "engine": ("[trt]", "tensorrt", "trtexec"),
    "tflite": ("ai_edge_litert", "tflite_runtime", "tensorflow.lite"),
    "torch": ("torch.jit", "pytorchstreamreader", "c10::", "aten::"),
}


def runtime_failure_providers(error_text: str) -> set[str]:
    """Return runtime backends explicitly present in a diagnostic log."""
    text = str(error_text or "").lower()
    return {
        runtime
        for runtime, markers in _RUNTIME_DIAGNOSTIC_MARKERS.items()
        if any(marker in text for marker in markers)
    }


def runtime_failure_is_observable(
    *,
    stage: str,
    runtime: str,
    error_text: str,
    artifact_fingerprint: Optional[Dict[str, Any]] = None,
) -> bool:
    """Admit only failures tied to the requested physical runtime.

    Edge wrappers also surface generated Python, payload, and transport errors.
    A physical stage name alone therefore has no observation authority.  An
    exact artifact-header rejection or a diagnostic owned by the requested
    runtime is required; fallback itself remains ordinary trial evidence.
    """
    normalized_stage = str(stage or "").strip().lower()
    if normalized_stage not in {
        "artifact_load",
        "compile",
        "engine_build",
        "inference",
        "model_load",
        "runtime",
    }:
        return False
    fingerprint = artifact_fingerprint or {}
    artifact_format = str(fingerprint.get("artifact_format") or "").strip().lower()
    runtime_key = _runtime_key(runtime)
    expected_formats = {
        "onnx": {"onnx"},
        "engine": {"onnx"} if normalized_stage in {"compile", "engine_build"} else {"engine"},
        "tflite": {"tflite"},
        "torch": {"pt", "pth", "torchscript", "ts"},
    }.get(runtime_key, set())
    if not artifact_format or artifact_format not in expected_formats:
        return False
    if locate_failure_artifact_predicate(error_text, fingerprint):
        return True
    return runtime_failure_providers(error_text) == {runtime_key}


def locate_failure_operator(error_text: str, artifact_fingerprint: Dict[str, Any]) -> str:
    """Return the artifact operator explicitly named by a runtime failure."""
    graph_ops = [str(item) for item in (artifact_fingerprint or {}).get("op_set") or []]
    op_match = re.search(r"(?:op|operator|node)[:= ]+([A-Za-z0-9_:.]+)", str(error_text or ""), re.I)
    parsed_op = op_match.group(1) if op_match else ""
    exact = next((item for item in graph_ops if item.lower() == parsed_op.lower()), "")
    if exact:
        return exact
    return next((item for item in graph_ops if item.lower() in str(error_text or "").lower()), "")


def locate_failure_artifact_predicate(
    error_text: str,
    artifact_fingerprint: Dict[str, Any],
) -> Dict[str, Any]:
    """Locate an exact artifact-header fact explicitly rejected by a runtime."""
    text = str(error_text or "")
    properties = dict((artifact_fingerprint or {}).get("artifact_properties") or {})
    ir_version = properties.get("ir_version")
    if (
        str((artifact_fingerprint or {}).get("artifact_format") or "").lower() == "onnx"
        and isinstance(ir_version, int)
        and re.search(
            r"(?:unsupported|support).*\bir\s+version|\bir\s+version.*(?:unsupported|support)",
            text,
            re.I,
        )
    ):
        return {
            "artifact_format": "onnx",
            "artifact_properties": {"ir_version": ir_version},
        }
    opset_imports = properties.get("opset_imports")
    onnx_opset = opset_imports.get("ai.onnx") if isinstance(opset_imports, dict) else None
    if (
        str((artifact_fingerprint or {}).get("artifact_format") or "").lower() == "onnx"
        and isinstance(onnx_opset, int)
        and re.search(
            r"(?:opset.*(?:unsupported|support|guarantee)|(?:unsupported|support).*opset)",
            text,
            re.I,
        )
    ):
        return {
            "artifact_format": "onnx",
            "artifact_properties": {"opset_imports": {"ai.onnx": onnx_opset}},
        }
    rejected_dtype = re.search(
        r"unsupported\s+onnx\s+data\s+type\s*:\s*([a-z0-9_]+)",
        text,
        re.I,
    )
    input_dtypes = {
        str(spec.get("dtype") or "").upper()
        for spec in (artifact_fingerprint or {}).get("input_specs") or []
        if isinstance(spec, dict)
    }
    if (
        str((artifact_fingerprint or {}).get("artifact_format") or "").lower() == "onnx"
        and rejected_dtype
        and rejected_dtype.group(1).upper() in input_dtypes
    ):
        return {
            "artifact_format": "onnx",
            "input_dtypes": [rejected_dtype.group(1).upper()],
        }
    return {}


def _artifact_predicate_matches(
    predicate: Dict[str, Any],
    fingerprint: Dict[str, Any],
) -> bool:
    """Match only facts explicitly recorded by a verified reproduction."""
    if not predicate:
        return False
    if predicate.get("artifact_format") and (
        str(predicate["artifact_format"]).lower()
        != str((fingerprint or {}).get("artifact_format") or "").lower()
    ):
        return False
    expected = dict(predicate.get("artifact_properties") or {})
    actual = dict((fingerprint or {}).get("artifact_properties") or {})
    if not all(actual.get(key) == value for key, value in expected.items()):
        return False
    expected_dtypes = {
        str(item).upper() for item in predicate.get("input_dtypes") or []
    }
    actual_dtypes = {
        str(spec.get("dtype") or "").upper()
        for spec in (fingerprint or {}).get("input_specs") or []
        if isinstance(spec, dict)
    }
    return not expected_dtypes or expected_dtypes.issubset(actual_dtypes)


def _pattern_has_verified_scope(pattern: Dict[str, Any]) -> bool:
    """A hard gate needs the scope proved by the minimal reproduction."""
    candidate = pattern or {}
    subgraph = candidate.get("subgraph_signature")
    if isinstance(subgraph, dict):
        has_structure = bool(
            subgraph.get("op_type")
            and (
                "inputs" in subgraph
                or "outputs" in subgraph
                or bool(subgraph.get("attributes_hash"))
                or bool(subgraph.get("schema"))
            )
        )
        if has_structure:
            return True
    artifact = candidate.get("artifact_predicate")
    return bool(
        isinstance(artifact, dict)
        and artifact.get("artifact_format")
        and (artifact.get("artifact_properties") or artifact.get("input_dtypes"))
    )


class FailureObservation(BaseModel):
    id: str
    tenant_id: str = ""
    dataset_id: str = ""
    run_id: str = ""
    trial_id: str = ""
    stage: str = ""
    device: str = ""
    runtime: str = ""
    version: str = ""
    precision: str = ""
    environment_fingerprint: str = ""
    artifact_fingerprint: Dict[str, Any] = Field(default_factory=dict)
    error_fingerprint: str
    error_tail: str = ""
    op_type: str = ""
    artifact_predicate: Dict[str, Any] = Field(default_factory=dict)
    artifact_path: str = ""
    visibility: str = "tenant_private"
    repro_status: str = "pending"
    repro_scope: str = ""
    repro_path: str = ""
    reproduced_error_fingerprint: str = ""
    repro_protocol: Dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=_now)

    def to_prompt_dict(self) -> Dict[str, Any]:
        return {
            "observation_id": self.id,
            "device": self.device,
            "runtime": self.runtime,
            "version": self.version,
            "precision": self.precision,
            "environment_fingerprint": self.environment_fingerprint,
            "error_fingerprint": self.error_fingerprint,
            "op_type": self.op_type,
            "artifact_predicate": self.artifact_predicate,
            "error_tail": self.error_tail[-500:],
            "status": "observed_only",
        }


def failure_observation_matches(
    observation: FailureObservation,
    reproduced_error_text: str,
) -> bool:
    """Match current and legacy persisted error identities without widening scope."""
    if not reproduced_error_text:
        return False
    expected = {observation.error_fingerprint}
    if observation.error_tail:
        expected.add(error_fingerprint(observation.error_tail))
    return error_fingerprint(reproduced_error_text) in expected


class CompatibilityRule(BaseModel):
    id: str
    device: str = ""
    runtime: str = ""
    version_range: str = ""
    precision: str = ""
    environment_fingerprint: str = ""
    pattern: Dict[str, Any] = Field(default_factory=dict)
    verdict: str = "unsupported"
    evidence: Dict[str, Any] = Field(default_factory=dict)
    verification_protocol: Dict[str, Any] = Field(default_factory=dict)
    reproduced_error_fingerprint: str = ""
    minimal_repro_path: str = ""
    repro_status: str = "pending"
    support_count: int = 0
    confidence: float = 0.5
    status: str = "observed"
    visibility: str = "tenant_public"
    created_at: str = Field(default_factory=_now)
    updated_at: str = Field(default_factory=_now)

    def to_prompt_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.id,
            "device": self.device,
            "runtime": self.runtime,
            "version_range": self.version_range,
            "precision": self.precision,
            "environment_fingerprint": self.environment_fingerprint,
            "pattern": self.pattern,
            "verdict": self.verdict,
            "support_count": self.support_count,
            "repro_status": self.repro_status,
            "status": self.status,
            "visibility": self.visibility,
        }

    def can_hard_gate(self, *, min_support: int = 0) -> bool:
        _ = min_support  # compatibility parameter; support never grants authority.
        return (
            self.status == "verified"
            and self.repro_status == "verified"
            and self.verdict == "unsupported"
            and bool(self.device)
            and bool(self.runtime)
            and bool(self.version_range)
            and bool(self.precision)
            and bool(self.environment_fingerprint)
            and bool(self.reproduced_error_fingerprint)
            and _pattern_has_verified_scope(self.pattern)
        )


def _verified_rule_scope(rule: CompatibilityRule) -> str:
    """Identity of the physical fact proved by a compatibility rule."""
    return _signature_key({
        "device": rule.device,
        "runtime": _runtime_key(rule.runtime),
        "version_range": rule.version_range,
        "precision": rule.precision,
        "environment_fingerprint": rule.environment_fingerprint,
        "pattern": rule.pattern,
        "verdict": rule.verdict,
        "visibility": rule.visibility,
    })


def _verified_rule_id(rule: CompatibilityRule) -> str:
    """Stable ID for one exact verified physical scope."""
    return "rule_" + hashlib.sha256(
        _verified_rule_scope(rule).encode()
    ).hexdigest()[:16]


def _load_public_rules(path: Path) -> List[CompatibilityRule]:
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(raw, dict) or not str(raw.get("sharing_policy") or "").startswith(
        "Only verified and sanitized hardware facts"
    ):
        return []
    public = raw.get("verified_compatibility_rules")
    if not isinstance(public, list):
        return []
    return [
        CompatibilityRule(
            id=str(item["rule_id"]),
            device=str(item["device"]),
            runtime=str(item["runtime"]),
            version_range=str(item["version_range"]),
            precision=str(item["precision"]),
            environment_fingerprint=str(item["environment_fingerprint"]),
            pattern=dict(item["pattern"]),
            verdict=str(item.get("verdict") or "unsupported"),
            verification_protocol=dict(item.get("verification_protocol") or {}),
            reproduced_error_fingerprint=str(item["reproduced_error_fingerprint"]),
            repro_status="verified",
            support_count=int(item.get("support_count") or 0),
            confidence=float(item.get("confidence") or 1.0),
            status="verified",
            visibility="tenant_public",
        )
        for item in public
        if isinstance(item, dict)
        and item.get("status") == "verified"
        and item.get("precision")
        and (item.get("verification_protocol") or {}).get("gate_eligible")
    ]


_PUBLIC_RULE_PROTOCOL_KEYS = (
    "gate_eligible",
    "reproduction_scope",
    "device",
    "runtime",
    "runtime_version",
    "precision",
    "environment_fingerprint",
    "artifact_format",
    "compiler",
    "compiler_version",
)


def public_rule_record(rule: CompatibilityRule) -> Optional[Dict[str, Any]]:
    """Return the allowlisted cross-tenant form of one verified rule."""
    if not rule.can_hard_gate() or rule.visibility != "tenant_public":
        return None
    pattern = {
        key: rule.pattern[key]
        for key in (
            "op_type",
            "op_signature",
            "subgraph_signature",
            "artifact_predicate",
        )
        if rule.pattern.get(key) not in (None, "", [], {})
    }
    protocol = {
        key: rule.verification_protocol[key]
        for key in _PUBLIC_RULE_PROTOCOL_KEYS
        if rule.verification_protocol.get(key) not in (None, "", [], {})
    }
    if not _pattern_has_verified_scope(pattern):
        return None
    return {
        "rule_id": rule.id,
        "device": rule.device,
        "runtime": rule.runtime,
        "version_range": rule.version_range,
        "precision": rule.precision,
        "environment_fingerprint": rule.environment_fingerprint,
        "pattern": pattern,
        "verdict": rule.verdict,
        "verification_protocol": protocol,
        "reproduced_error_fingerprint": rule.reproduced_error_fingerprint,
        "support_count": int(rule.support_count),
        "confidence": float(rule.confidence),
        "status": "verified",
    }


class CompatibilityRuleStore:
    """Exact compatibility memory backed by SQLite or a public read-only snapshot."""

    def __init__(self, path: Optional[Path] = None) -> None:
        env_path = str(getattr(settings, "COMPAT_RULE_STORE_PATH", "") or "").strip()
        self.path = Path(path or env_path or (Path(settings.DATA_DIR) / "compatibility_memory.sqlite3"))
        self.read_only = bool(settings.COMPAT_RULE_STORE_READ_ONLY)
        self.requested_path = str(self.path)
        self._snapshot_only = False
        self._snapshot_rules: List[CompatibilityRule] = []
        self._legacy_items: List[Dict[str, Any]] = []
        if self.path.exists():
            try:
                prefix = self.path.read_bytes()[:1]
                if prefix in {b"[", b"{"}:
                    raw = json.loads(self.path.read_text())
                    self._snapshot_rules = _load_public_rules(self.path)
                    if self._snapshot_rules:
                        self._snapshot_only = True
                        self.read_only = True
                        return
                    self._legacy_items = list(
                        raw if isinstance(raw, list) else raw.get("rules", [])
                    )
                    self.path = self.path.with_suffix(".sqlite3")
            except Exception:
                pass
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self._import_legacy()
        self._compact_verified_rules()
        seed_path = str(getattr(settings, "COMPAT_RULE_SEED_PATH", "") or "").strip()
        if seed_path:
            self._snapshot_rules = _load_public_rules(Path(seed_path))

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path), timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS observations (
                    id TEXT PRIMARY KEY,
                    tenant_scope TEXT NOT NULL DEFAULT '',
                    environment_fingerprint TEXT,
                    runtime TEXT,
                    error_fingerprint TEXT,
                    op_type TEXT,
                    payload TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_observation_exact
                    ON observations(environment_fingerprint, runtime, error_fingerprint, op_type);
                CREATE TABLE IF NOT EXISTS rules (
                    id TEXT PRIMARY KEY,
                    environment_fingerprint TEXT,
                    device TEXT,
                    runtime TEXT,
                    version_range TEXT,
                    op_type TEXT,
                    status TEXT,
                    payload TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_rule_exact
                    ON rules(environment_fingerprint, runtime, op_type, status);
                """
            )
            columns = {
                str(row[1]) for row in conn.execute("PRAGMA table_info(observations)")
            }
            if "tenant_scope" not in columns:
                conn.execute(
                    "ALTER TABLE observations ADD COLUMN tenant_scope TEXT NOT NULL DEFAULT ''"
                )
            # Upgrade any pre-boundary records into an opaque default or
            # payload-declared tenant scope. No raw tenant identifier remains
            # in the persisted observation after migration.
            rows = conn.execute(
                "SELECT id, payload FROM observations WHERE tenant_scope=''"
            ).fetchall()
            for observation_id, payload in rows:
                try:
                    observation = FailureObservation.model_validate_json(payload)
                except Exception:
                    continue
                scope = _tenant_scope_key(observation.tenant_id)
                observation.tenant_id = scope
                conn.execute(
                    "UPDATE observations SET tenant_scope=?, payload=? WHERE id=?",
                    (scope, observation.model_dump_json(), observation_id),
                )
            conn.execute(
                """CREATE INDEX IF NOT EXISTS idx_observation_tenant_exact
                   ON observations(
                       tenant_scope, environment_fingerprint, runtime,
                       error_fingerprint, op_type
                   )"""
            )

    def _import_legacy(self) -> None:
        for item in self._legacy_items:
            try:
                legacy = CompatibilityRule(**item)
                # Legacy confidence/support are evidence, not verification.
                legacy.status = "observed"
                legacy.repro_status = "pending"
                self.add_or_update(legacy)
            except Exception:
                continue
        self._legacy_items = []

    def _compact_verified_rules(self) -> None:
        """Canonicalize IDs and collapse proofs of the same exact scope."""
        if self.read_only:
            return
        groups: Dict[str, List[CompatibilityRule]] = {}
        for rule in self.list_rules():
            if rule.status == "verified" and rule.repro_status == "verified":
                groups.setdefault(_verified_rule_scope(rule), []).append(rule)
        migration_groups = [
            items
            for items in groups.values()
            if len(items) > 1
            or any(item.id != _verified_rule_id(item) for item in items)
        ]
        if not migration_groups:
            return
        with self._connect() as conn:
            for items in migration_groups:
                items.sort(key=lambda item: (item.created_at, item.id))
                canonical = items[0]
                observation_ids = {
                    str(value)
                    for item in items
                    for value in [
                        (item.evidence or {}).get("observation_id"),
                        *((item.evidence or {}).get("observation_ids") or []),
                    ]
                    if value
                }
                canonical.evidence = {
                    **(canonical.evidence or {}),
                    "observation_ids": sorted(observation_ids),
                }
                canonical.support_count = max(
                    len(observation_ids),
                    *(item.support_count for item in items),
                )
                canonical.updated_at = _now()
                canonical_id = _verified_rule_id(canonical)
                for item in items:
                    conn.execute("DELETE FROM rules WHERE id=?", (item.id,))
                canonical.id = canonical_id
                conn.execute(
                    """INSERT OR REPLACE INTO rules
                       (id, environment_fingerprint, device, runtime, version_range, op_type, status, payload)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        canonical.id,
                        canonical.environment_fingerprint,
                        canonical.device,
                        canonical.runtime,
                        canonical.version_range,
                        str(canonical.pattern.get("op_type") or ""),
                        canonical.status,
                        canonical.model_dump_json(),
                    ),
                )

    def add_or_update(self, rule: CompatibilityRule) -> CompatibilityRule:
        rule.updated_at = _now()
        if self.read_only:
            return rule
        payload = rule.model_dump_json()
        op_type = str(rule.pattern.get("op_type") or "")
        with self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO rules
                   (id, environment_fingerprint, device, runtime, version_range, op_type, status, payload)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    rule.id,
                    rule.environment_fingerprint,
                    rule.device,
                    rule.runtime,
                    rule.version_range,
                    op_type,
                    rule.status,
                    payload,
                ),
            )
        return rule

    def list_rules(self) -> List[CompatibilityRule]:
        if self._snapshot_only:
            return list(self._snapshot_rules)
        with self._connect() as conn:
            rows = conn.execute("SELECT payload FROM rules ORDER BY id").fetchall()
        values = [CompatibilityRule.model_validate_json(row[0]) for row in rows]
        return list({item.id: item for item in [*self._snapshot_rules, *values]}.values())

    def _list_observations_for_scope(self, tenant_scope: str) -> List[FailureObservation]:
        if self._snapshot_only:
            return []
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT payload FROM observations WHERE tenant_scope=? ORDER BY rowid",
                (tenant_scope,),
            ).fetchall()
        return [FailureObservation.model_validate_json(row[0]) for row in rows]

    def list_observations(self, *, tenant_id: str = "default") -> List[FailureObservation]:
        """List only one tenant's private observations."""
        return self._list_observations_for_scope(_tenant_scope_key(tenant_id))

    def export_public_snapshot(
        self,
        output_path: str | Path,
        *,
        overwrite: bool = False,
    ) -> Path:
        """Atomically export only sanitized, gate-eligible verified rules."""
        destination = Path(output_path).expanduser()
        if destination.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite existing snapshot: {destination}")
        records = [public_rule_record(rule) for rule in self.list_rules()]
        payload = {
            "schema_version": "edgecraft_public_rules_v1",
            "sharing_policy": (
                "Only verified and sanitized hardware facts are shared across tenants; "
                "raw failures, paths, tenant IDs, datasets, code, and artifacts are excluded."
            ),
            "verified_compatibility_rules": [item for item in records if item is not None],
        }
        destination.parent.mkdir(parents=True, exist_ok=True)
        pending = destination.with_suffix(destination.suffix + ".tmp")
        pending.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        pending.replace(destination)
        return destination

    def exact_match(
        self,
        *,
        device: str,
        runtime: str,
        version: str = "",
        precision: str = "",
        environment_fingerprint: str = "",
        op_types: Optional[List[str]] = None,
        op_signatures: Optional[List[str]] = None,
        subgraph_signatures: Optional[List[Dict[str, Any]]] = None,
        artifact_fingerprint: Optional[Dict[str, Any]] = None,
        min_confidence: float = 0.0,
    ) -> List[CompatibilityRule]:
        _ = min_confidence
        ops = {str(op).lower() for op in (op_types or [])}
        signatures = {str(sig) for sig in (op_signatures or [])}
        subgraphs = {_signature_key(sig) for sig in (subgraph_signatures or [])}
        hits: List[CompatibilityRule] = []
        for rule in self.list_rules():
            if rule.status != "verified" or rule.visibility != "tenant_public":
                continue
            if rule.device and rule.device != device:
                continue
            if rule.runtime and _runtime_key(rule.runtime) != _runtime_key(runtime):
                continue
            if rule.version_range and rule.version_range != version:
                continue
            if rule.precision and rule.precision != precision:
                continue
            if not _environment_scope_matches(
                rule,
                device=device,
                runtime=runtime,
                version=version,
                environment_fingerprint=environment_fingerprint,
            ):
                continue
            rule_op = str(rule.pattern.get("op_type") or "").lower()
            rule_signature = str(rule.pattern.get("op_signature") or "")
            rule_subgraph = rule.pattern.get("subgraph_signature")
            rule_artifact = rule.pattern.get("artifact_predicate")
            if rule_subgraph and _signature_key(rule_subgraph) not in subgraphs:
                continue
            if rule_signature and rule_signature not in signatures:
                continue
            if rule_op and rule_op not in ops:
                continue
            if rule_artifact and not _artifact_predicate_matches(
                dict(rule_artifact), artifact_fingerprint or {}
            ):
                continue
            if rule_op or rule_signature or rule_subgraph or rule_artifact:
                hits.append(rule)
        return hits

    def observe_failure(
        self,
        *,
        device: str,
        runtime: str,
        version: str = "",
        precision: str = "",
        error_text: str,
        artifact_path: str = "",
        tenant_id: str = "",
        dataset_id: str = "",
        run_id: str = "",
        trial_id: str = "",
        stage: str = "edge_benchmark",
        environment_fingerprint: str = "",
        artifact_fingerprint: Optional[Dict[str, Any]] = None,
    ) -> Optional[FailureObservation]:
        if self.read_only:
            return None
        text = str(error_text or "").strip()
        if not text:
            return None
        op_type = locate_failure_operator(text, artifact_fingerprint or {})
        artifact_predicate = locate_failure_artifact_predicate(
            text, artifact_fingerprint or {}
        )
        err_fp = error_fingerprint(text)
        tenant_scope = _tenant_scope_key(tenant_id)
        identity = {
            "tenant_scope": tenant_scope,
            "run_id": run_id,
            "trial_id": trial_id,
            "environment": environment_fingerprint,
            "runtime": runtime,
            "precision": precision,
            "error": err_fp,
            "op": op_type,
            "artifact_predicate": artifact_predicate,
        }
        observation = FailureObservation(
            id=f"obs_{hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]}",
            tenant_id=tenant_scope,
            dataset_id=dataset_id,
            run_id=run_id,
            trial_id=trial_id,
            stage=stage,
            device=device,
            runtime=runtime,
            version=version,
            precision=precision,
            environment_fingerprint=environment_fingerprint,
            artifact_fingerprint=artifact_fingerprint or {},
            error_fingerprint=err_fp,
            error_tail=text[-1600:],
            op_type=op_type,
            artifact_predicate=artifact_predicate,
            artifact_path=artifact_path,
        )
        with self._connect() as conn:
            conn.execute(
                """INSERT OR IGNORE INTO observations
                   (id, tenant_scope, environment_fingerprint, runtime,
                    error_fingerprint, op_type, payload)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    observation.id,
                    tenant_scope,
                    environment_fingerprint,
                    runtime,
                    err_fp,
                    op_type,
                    observation.model_dump_json(),
                ),
            )
        return observation

    def record_reproduction(
        self,
        observation: FailureObservation,
        *,
        status: str,
        scope: str,
        path: str = "",
        reproduced_error_text: str = "",
        protocol: Optional[Dict[str, Any]] = None,
    ) -> FailureObservation:
        """Attach replay evidence to the private observation record."""
        observation.repro_status = status
        observation.repro_scope = scope
        observation.repro_path = path
        observation.reproduced_error_fingerprint = (
            error_fingerprint(reproduced_error_text) if reproduced_error_text else ""
        )
        observation.repro_protocol = dict(protocol or {})
        if self.read_only:
            return observation
        with self._connect() as conn:
            conn.execute(
                "UPDATE observations SET payload=? WHERE id=?",
                (observation.model_dump_json(), observation.id),
            )
        return observation

    def observation_support(self, observation: FailureObservation) -> int:
        return sum(
            1
            for item in self._list_observations_for_scope(
                observation.tenant_id or _tenant_scope_key("default")
            )
            if item.environment_fingerprint == observation.environment_fingerprint
            and _runtime_key(item.runtime) == _runtime_key(observation.runtime)
            and item.version == observation.version
            and item.precision == observation.precision
            and item.error_fingerprint == observation.error_fingerprint
            and item.op_type == observation.op_type
        )

    def verify_observation(
        self,
        observation: FailureObservation,
        *,
        reproduced_error_text: str,
        minimal_repro_path: str,
        pattern: Dict[str, Any],
        protocol: Dict[str, Any],
    ) -> Optional[CompatibilityRule]:
        if not _pattern_has_verified_scope(pattern):
            return None
        if not bool(protocol.get("gate_eligible")):
            return None
        if str(protocol.get("reproduction_scope") or "") not in {
            "artifact_header",
            "minimal_subgraph",
        }:
            return None
        if _runtime_key(protocol.get("runtime")) != _runtime_key(observation.runtime):
            return None
        if observation.device and str(protocol.get("device") or "") != observation.device:
            return None
        if observation.version and str(protocol.get("runtime_version") or "") != observation.version:
            return None
        if observation.precision and str(protocol.get("precision") or "") != observation.precision:
            return None
        if (
            observation.environment_fingerprint
            and str(protocol.get("environment_fingerprint") or "")
            != observation.environment_fingerprint
        ):
            return None
        reproduced = error_fingerprint(reproduced_error_text)
        if not failure_observation_matches(observation, reproduced_error_text):
            return None
        scope_probe = CompatibilityRule(
            id="scope_probe",
            device=observation.device,
            runtime=observation.runtime,
            version_range=observation.version,
            precision=observation.precision,
            environment_fingerprint=observation.environment_fingerprint,
            pattern=pattern,
            verdict="unsupported",
        )
        existing = next(
            (
                rule
                for rule in self.list_rules()
                if rule.status == "verified"
                and rule.repro_status == "verified"
                and _verified_rule_scope(rule) == _verified_rule_scope(scope_probe)
            ),
            None,
        )
        rule_id = existing.id if existing else _verified_rule_id(scope_probe)
        previous_evidence = (existing.evidence or {}) if existing else {}
        observation_ids = sorted({
            observation.id,
            *(
                str(value)
                for value in (previous_evidence.get("observation_ids") or [])
                if value
            ),
            *(
                [str(previous_evidence.get("observation_id"))]
                if previous_evidence.get("observation_id")
                else []
            ),
        })
        rule = CompatibilityRule(
            id=rule_id,
            device=observation.device,
            runtime=observation.runtime,
            version_range=observation.version,
            precision=observation.precision,
            environment_fingerprint=observation.environment_fingerprint,
            pattern=pattern,
            verdict="unsupported",
            evidence={
                "observation_id": observation.id,
                "observation_ids": observation_ids,
            },
            verification_protocol=protocol,
            reproduced_error_fingerprint=reproduced,
            minimal_repro_path=minimal_repro_path,
            repro_status="verified",
            support_count=max(
                self.observation_support(observation),
                len(observation_ids),
                existing.support_count if existing else 0,
            ),
            confidence=1.0,
            status="verified",
        )
        stored = self.add_or_update(rule)
        self.record_reproduction(
            observation,
            status="verified",
            scope=str(protocol.get("reproduction_scope") or ""),
            path=minimal_repro_path,
            reproduced_error_text=reproduced_error_text,
            protocol=protocol,
        )
        return stored

    def log_hints(
        self,
        *,
        device: str,
        runtime: str,
        error_text: str,
        limit: int = 5,
    ) -> List[CompatibilityRule]:
        blob = str(error_text or "").lower()
        hits = []
        for rule in self.list_rules():
            if rule.status != "verified" or rule.visibility != "tenant_public":
                continue
            if rule.device and rule.device != device:
                continue
            if rule.runtime and _runtime_key(rule.runtime) != _runtime_key(runtime):
                continue
            op = str(rule.pattern.get("op_type") or "").lower()
            needle = str(rule.pattern.get("error_substring") or "").lower()
            if (op and op in blob) or (needle and needle in blob):
                hits.append(rule)
        return hits[:limit]

    def rules_for_environment(
        self,
        *,
        environment_fingerprint: str,
        device: str = "",
        runtime: str = "",
        version: str = "",
        precision: str = "",
        limit: int = 8,
    ) -> List[CompatibilityRule]:
        return [
            rule
            for rule in self.list_rules()
            if rule.status == "verified"
            and rule.visibility == "tenant_public"
            and (not precision or rule.precision == precision)
            and _environment_scope_matches(
                rule,
                device=device,
                runtime=runtime,
                version=version,
                environment_fingerprint=environment_fingerprint,
            )
        ][:limit]


_STORE: Optional[CompatibilityRuleStore] = None


def get_compatibility_rule_store() -> CompatibilityRuleStore:
    global _STORE
    env_path = str(getattr(settings, "COMPAT_RULE_STORE_PATH", "") or "").strip()
    desired = str(Path(env_path) if env_path else Path(settings.DATA_DIR) / "compatibility_memory.sqlite3")
    if _STORE is None or _STORE.requested_path != desired:
        _STORE = CompatibilityRuleStore()
    return _STORE
