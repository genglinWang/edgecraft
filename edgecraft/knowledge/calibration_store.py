"""Append-only calibration evidence for the multi-fidelity verifier."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


def _stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


_PUBLIC_PROTOCOL_KEYS = (
    "runtime",
    "runtime_provider",
    "precision",
    "warmup",
    "warmup_unit",
    "min_warmup_seconds",
    "repetitions",
    "measurement_policy",
    "min_measure_seconds",
    "dvfs_state",
    "thermal_state",
    "measurement_contract_version",
    "energy_source",
    "energy_scope",
    "trt_timing_cache_id",
    "trt_timing_cache_hit",
    "actual_warmup",
    "actual_warmup_seconds",
    "actual_repetitions",
    "measurement_sessions",
    "latency_statistic",
    "latency_uncertainty",
    "benchmark_scope",
    "energy_trusted",
)

_CALIBRATION_PROTOCOL_KEYS = (
    "runtime",
    "runtime_provider",
    "precision",
    "warmup",
    "warmup_unit",
    "min_warmup_seconds",
    "repetitions",
    "measurement_policy",
    "min_measure_seconds",
    "dvfs_state",
    "thermal_state",
    "measurement_contract_version",
    "energy_source",
    "energy_scope",
    "trt_timing_cache_id",
    "trt_timing_cache_hit",
    "actual_warmup",
    "actual_warmup_seconds",
    "actual_repetitions",
    "measurement_sessions",
    "latency_statistic",
    "latency_uncertainty",
    "benchmark_scope",
    "energy_trusted",
)


def sanitize_protocol(protocol: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Keep only hardware-measurement facts safe to share across tenants."""
    source = protocol or {}
    return {
        key: source.get(key)
        for key in _PUBLIC_PROTOCOL_KEYS
        if source.get(key) not in (None, "", [])
    }


def calibration_protocol_fingerprint(protocol: Optional[Dict[str, Any]]) -> str:
    source = protocol or {}
    public = sanitize_protocol(protocol)
    input_profile = source.get("input_profile")
    if input_profile not in (None, "", []):
        # Keep model-specific shape details private while still isolating
        # calibration evidence acquired under different input profiles.
        public["input_profile_hash"] = _stable_hash(input_profile)
    return _stable_hash(
        {
            **{key: public[key] for key in _CALIBRATION_PROTOCOL_KEYS if key in public},
            **(
                {"input_profile_hash": public["input_profile_hash"]}
                if "input_profile_hash" in public
                else {}
            ),
        }
    )


class CalibrationPair(BaseModel):
    pair_id: str = ""
    environment_fingerprint: str
    protocol_fingerprint: str
    quantity: str
    cheap_value: float
    full_value: float
    absolute_error: float = 0.0
    cheap_sigma: Optional[float] = None
    full_sigma: Optional[float] = None
    graph_hash: str = ""
    cheap_evidence_id: str
    full_evidence_id: str
    cheap_protocol: Dict[str, Any] = Field(default_factory=dict)
    full_protocol: Dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def model_post_init(self, __context: Any) -> None:
        if not self.pair_id:
            self.pair_id = "cal_" + _stable_hash(
                {
                    "environment": self.environment_fingerprint,
                    "protocol": self.protocol_fingerprint,
                    "quantity": self.quantity,
                    "cheap_evidence": self.cheap_evidence_id,
                    "full_evidence": self.full_evidence_id,
                }
            )[:16]
        self.absolute_error = abs(float(self.full_value) - float(self.cheap_value))
        self.cheap_protocol = sanitize_protocol(self.cheap_protocol)
        self.full_protocol = sanitize_protocol(self.full_protocol)


class CalibrationSnapshot(BaseModel):
    snapshot_id: str
    environment_fingerprint: str
    protocol_fingerprint: str
    graph_hash: str = ""
    pair_ids: List[str] = Field(default_factory=list)
    errors: Dict[str, float] = Field(default_factory=dict)
    pair_count: int = 0


class CalibrationStore:
    """SQLite ledger whose only mutation is insertion of measured pairs."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 30000")
        conn.execute("PRAGMA journal_mode = WAL")
        return conn

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS calibration_pairs (
                    pair_id TEXT PRIMARY KEY,
                    environment_fingerprint TEXT NOT NULL,
                    protocol_fingerprint TEXT NOT NULL,
                    quantity TEXT NOT NULL,
                    cheap_value REAL NOT NULL,
                    full_value REAL NOT NULL,
                    absolute_error REAL NOT NULL,
                    cheap_sigma REAL,
                    full_sigma REAL,
                    graph_hash TEXT NOT NULL,
                    cheap_evidence_id TEXT NOT NULL,
                    full_evidence_id TEXT NOT NULL,
                    cheap_protocol_json TEXT NOT NULL,
                    full_protocol_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(cheap_evidence_id, full_evidence_id)
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_calibration_environment_protocol
                ON calibration_pairs(environment_fingerprint, protocol_fingerprint, pair_id)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_calibration_environment_protocol_graph
                ON calibration_pairs(
                    environment_fingerprint, protocol_fingerprint, graph_hash, pair_id
                )
                """
            )

    def append(self, pair: CalibrationPair) -> str:
        if not all(
            (
                pair.environment_fingerprint,
                pair.protocol_fingerprint,
                pair.graph_hash,
                pair.cheap_evidence_id,
                pair.full_evidence_id,
            )
        ):
            raise ValueError(
                "calibration pairs require environment, protocol, graph, and both evidence IDs"
            )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO calibration_pairs (
                    pair_id, environment_fingerprint, protocol_fingerprint,
                    quantity, cheap_value, full_value, absolute_error,
                    cheap_sigma, full_sigma, graph_hash, cheap_evidence_id,
                    full_evidence_id, cheap_protocol_json, full_protocol_json,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    pair.pair_id,
                    pair.environment_fingerprint,
                    pair.protocol_fingerprint,
                    pair.quantity,
                    pair.cheap_value,
                    pair.full_value,
                    pair.absolute_error,
                    pair.cheap_sigma,
                    pair.full_sigma,
                    pair.graph_hash,
                    pair.cheap_evidence_id,
                    pair.full_evidence_id,
                    json.dumps(pair.cheap_protocol, sort_keys=True),
                    json.dumps(pair.full_protocol, sort_keys=True),
                    pair.created_at,
                ),
            )
        return pair.pair_id

    def list_pairs(self) -> List[CalibrationPair]:
        """Return measured pairs for reporting; insertion remains the only mutation."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM calibration_pairs ORDER BY created_at, pair_id"
            ).fetchall()
        return [
            CalibrationPair(
                pair_id=str(row["pair_id"]),
                environment_fingerprint=str(row["environment_fingerprint"]),
                protocol_fingerprint=str(row["protocol_fingerprint"]),
                quantity=str(row["quantity"]),
                cheap_value=float(row["cheap_value"]),
                full_value=float(row["full_value"]),
                absolute_error=float(row["absolute_error"]),
                cheap_sigma=row["cheap_sigma"],
                full_sigma=row["full_sigma"],
                graph_hash=str(row["graph_hash"]),
                cheap_evidence_id=str(row["cheap_evidence_id"]),
                full_evidence_id=str(row["full_evidence_id"]),
                cheap_protocol=json.loads(str(row["cheap_protocol_json"])),
                full_protocol=json.loads(str(row["full_protocol_json"])),
                created_at=str(row["created_at"]),
            )
            for row in rows
        ]

    def export_public_snapshot(
        self,
        output_path: str | Path,
        *,
        overwrite: bool = False,
    ) -> Path:
        """Export pair-backed calibration without tenant or workspace provenance."""
        destination = Path(output_path).expanduser()
        if destination.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite existing snapshot: {destination}")
        pairs = [
            {
                "pair_id": pair.pair_id,
                "environment_fingerprint": pair.environment_fingerprint,
                "protocol_fingerprint": pair.protocol_fingerprint,
                "graph_hash": pair.graph_hash,
                "quantity": pair.quantity,
                "cheap_value": pair.cheap_value,
                "full_value": pair.full_value,
                "absolute_error": pair.absolute_error,
                "cheap_sigma": pair.cheap_sigma,
                "full_sigma": pair.full_sigma,
                "cheap_protocol": sanitize_protocol(pair.cheap_protocol),
                "full_protocol": sanitize_protocol(pair.full_protocol),
            }
            for pair in self.list_pairs()
            if pair.graph_hash
            and pair.environment_fingerprint
            and pair.protocol_fingerprint
            and pair.pair_id
        ]
        payload = {
            "schema_version": "edgecraft_public_calibration_v1",
            "sharing_policy": (
                "Pair-backed physical metric deltas only; tenant IDs, dataset IDs, "
                "workspace paths, model files, evidence transcripts, and timestamps are excluded."
            ),
            "pairs": pairs,
        }
        destination.parent.mkdir(parents=True, exist_ok=True)
        pending = destination.with_suffix(destination.suffix + ".tmp")
        pending.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        pending.replace(destination)
        return destination

    def snapshot(
        self,
        *,
        environment_fingerprint: str,
        protocol_fingerprint: str,
        graph_hash: str = "",
    ) -> CalibrationSnapshot:
        if not environment_fingerprint or not protocol_fingerprint or not graph_hash:
            raise ValueError(
                "calibration snapshots require exact environment, protocol, and graph identities"
            )
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT pair_id, quantity, absolute_error
                FROM calibration_pairs
                WHERE environment_fingerprint = ?
                  AND protocol_fingerprint = ?
                  AND graph_hash = ?
                ORDER BY pair_id
                """,
                (environment_fingerprint, protocol_fingerprint, graph_hash),
            ).fetchall()
        errors: Dict[str, float] = {}
        pair_ids: List[str] = []
        for row in rows:
            pair_ids.append(str(row["pair_id"]))
            quantity = str(row["quantity"])
            errors[quantity] = max(errors.get(quantity, 0.0), float(row["absolute_error"]))
        snapshot_id = "cs_" + _stable_hash(
            {
                "environment": environment_fingerprint,
                "protocol": protocol_fingerprint,
                "graph": graph_hash,
                "pairs": pair_ids,
            }
        )[:16]
        return CalibrationSnapshot(
            snapshot_id=snapshot_id,
            environment_fingerprint=environment_fingerprint,
            protocol_fingerprint=protocol_fingerprint,
            graph_hash=graph_hash,
            pair_ids=pair_ids,
            errors=errors,
            pair_count=len(pair_ids),
        )
