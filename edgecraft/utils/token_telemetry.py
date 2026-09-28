"""Append provider-reported LLM usage without storing prompts or responses."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from langchain_core.callbacks.base import BaseCallbackHandler


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return str(value)


def _nonnegative_int(*values: Any) -> Optional[int]:
    for value in values:
        if value is None:
            continue
        try:
            number = int(value)
        except (TypeError, ValueError):
            continue
        if number >= 0:
            return number
    return None


def _first_generation(response: Any) -> Any:
    generations = getattr(response, "generations", None) or []
    if generations and generations[0]:
        return generations[0][0]
    return None


class TokenTelemetryCallback(BaseCallbackHandler):
    """Write one compact JSONL record for every provider response."""

    raise_error = True

    def __init__(
        self,
        path: str,
        *,
        provider: str,
        model: str,
        purpose: str = "",
        experiment_run_id: str = "",
        required: bool = False,
    ) -> None:
        self.path = Path(path).expanduser()
        self.provider = provider
        self.model = model
        self.purpose = purpose
        self.experiment_run_id = experiment_run_id
        self.required = bool(required)
        self._starts: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def _remember_start(self, run_id: Any, payload: Any) -> None:
        encoded = json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":")).encode()
        with self._lock:
            self._starts[str(run_id)] = {
                "request_started_at": _utc_now(),
                "started_monotonic_s": time.monotonic(),
                "request_payload_sha256": hashlib.sha256(encoded).hexdigest(),
            }

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: Any,
        *,
        run_id: Any,
        **kwargs: Any,
    ) -> None:
        self._remember_start(run_id, messages)

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: Any,
        *,
        run_id: Any,
        **kwargs: Any,
    ) -> None:
        self._remember_start(run_id, prompts)

    def _pop_start(self, run_id: Any) -> Dict[str, Any]:
        with self._lock:
            return self._starts.pop(str(run_id), {})

    def _append(self, record: Dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        with self.path.open("a", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def on_llm_end(self, response: Any, *, run_id: Any, **kwargs: Any) -> None:
        start = self._pop_start(run_id)
        generation = _first_generation(response)
        message = getattr(generation, "message", None)
        usage_metadata = _jsonable(getattr(message, "usage_metadata", None) or {})
        response_metadata = _jsonable(getattr(message, "response_metadata", None) or {})
        llm_output = _jsonable(getattr(response, "llm_output", None) or {})
        raw_usage = llm_output.get("token_usage") or response_metadata.get("token_usage") or {}

        input_tokens = _nonnegative_int(
            usage_metadata.get("input_tokens"),
            raw_usage.get("input_tokens"),
            raw_usage.get("prompt_tokens"),
        )
        output_tokens = _nonnegative_int(
            usage_metadata.get("output_tokens"),
            raw_usage.get("output_tokens"),
            raw_usage.get("completion_tokens"),
        )
        total_tokens = _nonnegative_int(
            usage_metadata.get("total_tokens"),
            raw_usage.get("total_tokens"),
        )
        if total_tokens is None and input_tokens is not None and output_tokens is not None:
            total_tokens = input_tokens + output_tokens

        headers = response_metadata.get("headers") or {}
        provider_request_id = (
            response_metadata.get("request_id")
            or response_metadata.get("id")
            or headers.get("x-request-id")
            or headers.get("request-id")
            or llm_output.get("request_id")
            or llm_output.get("id")
            or getattr(message, "id", None)
        )
        elapsed_s = None
        if start.get("started_monotonic_s") is not None:
            elapsed_s = max(0.0, time.monotonic() - start["started_monotonic_s"])

        record = {
                "schema_version": "llm_token_telemetry_v1",
                "status": "completed",
                "experiment_run_id": self.experiment_run_id or None,
                "langchain_run_id": str(run_id),
                "provider": self.provider,
                "model": self.model,
                "requested_model": self.model,
                "reported_model": llm_output.get("model_name") or response_metadata.get("model_name"),
                "purpose": self.purpose or None,
                "provider_request_id": provider_request_id,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "raw_provider_usage": raw_usage,
                "request_started_at": start.get("request_started_at"),
                "response_received_at": _utc_now(),
                "elapsed_s": elapsed_s,
                "request_payload_sha256": start.get("request_payload_sha256"),
            }
        self._append(record)
        if self.required:
            missing = [
                key
                for key in (
                    "provider_request_id",
                    "input_tokens",
                    "output_tokens",
                    "total_tokens",
                    "request_started_at",
                    "response_received_at",
                )
                if record.get(key) is None
            ]
            if missing or not isinstance(record.get("raw_provider_usage"), dict):
                raise RuntimeError(
                    "required token telemetry is incomplete: "
                    + ", ".join(missing or ["raw_provider_usage"])
                )

    def on_llm_error(self, error: BaseException, *, run_id: Any, **kwargs: Any) -> None:
        start = self._pop_start(run_id)
        elapsed_s = None
        if start.get("started_monotonic_s") is not None:
            elapsed_s = max(0.0, time.monotonic() - start["started_monotonic_s"])
        self._append(
            {
                "schema_version": "llm_token_telemetry_v1",
                "status": "failed",
                "experiment_run_id": self.experiment_run_id or None,
                "langchain_run_id": str(run_id),
                "provider": self.provider,
                "model": self.model,
                "requested_model": self.model,
                "purpose": self.purpose or None,
                "provider_request_id": None,
                "input_tokens": None,
                "output_tokens": None,
                "total_tokens": None,
                "token_count_complete": False,
                "request_started_at": start.get("request_started_at"),
                "response_received_at": _utc_now(),
                "elapsed_s": elapsed_s,
                "request_payload_sha256": start.get("request_payload_sha256"),
                "error_type": type(error).__name__,
                "error_message": str(error)[:500],
            }
        )
