"""LLM backend factory.

This module keeps provider plumbing in one place.  The agent still consumes a
normal LangChain chat model; provider choice is configuration, not search
logic.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, Optional
from uuid import uuid4

from langchain_openai import ChatOpenAI

from edgecraft.config.settings import settings
from edgecraft.utils.token_telemetry import TokenTelemetryCallback


_TOOL_CALLING_PURPOSES = {"dataset_analyzer"}


def active_llm_provider() -> str:
    provider = (settings.LLM_PROVIDER or "openai").strip().lower()
    aliases = {
        "deepseek-v4": "deepseek",
        "deepseek_v4": "deepseek",
        "ds": "deepseek",
        "poe-openai": "poe",
        "openai-compatible": "openai",
    }
    return aliases.get(provider, provider)


def llm_credentials_available(provider: Optional[str] = None) -> bool:
    if settings.DISABLE_LLM:
        return False
    provider = provider or active_llm_provider()
    if provider == "deepseek":
        return bool(settings.DEEPSEEK_API_KEY)
    if provider == "poe":
        return bool(settings.POE_API_KEY or settings.OPENAI_API_KEY)
    return bool(settings.OPENAI_API_KEY)


def _telemetry_callbacks(*, provider: str, model: str, purpose: str) -> list[Any]:
    path = settings.TOKEN_TELEMETRY_PATH
    if settings.REQUIRE_TOKEN_TELEMETRY and not path:
        raise RuntimeError(
            "EDGECRAFT_REQUIRE_TOKEN_TELEMETRY=1 requires EDGECRAFT_TOKEN_TELEMETRY_PATH"
        )
    if not path:
        return []
    return [
        TokenTelemetryCallback(
            path,
            provider=provider,
            model=model,
            purpose=purpose,
            experiment_run_id=settings.EXPERIMENT_RUN_ID,
            required=settings.REQUIRE_TOKEN_TELEMETRY,
        )
    ]


class _TelemetryChatModel:
    """Delegate to a chat runnable while recording every synchronous invocation."""

    def __init__(self, client: Any, callback: TokenTelemetryCallback) -> None:
        self._client = client
        self._callback = callback

    def invoke(self, input_value: Any, config: Any = None, **kwargs: Any) -> Any:
        run_id = uuid4()
        self._callback.on_chat_model_start({}, [input_value], run_id=run_id)
        try:
            message = self._client.invoke(input_value, config=config, **kwargs)
        except BaseException as exc:
            self._callback.on_llm_error(exc, run_id=run_id)
            raise
        response = SimpleNamespace(
            generations=[[SimpleNamespace(message=message)]],
            llm_output={
                "token_usage": (
                    getattr(message, "response_metadata", None) or {}
                ).get("token_usage", {}),
                "request_id": getattr(message, "id", None),
            },
        )
        self._callback.on_llm_end(response, run_id=run_id)
        return message

    def bind_tools(self, *args: Any, **kwargs: Any) -> "_TelemetryChatModel":
        return _TelemetryChatModel(
            self._client.bind_tools(*args, **kwargs), self._callback
        )

    def with_config(self, **kwargs: Any) -> "_TelemetryChatModel":
        return _TelemetryChatModel(self._client.with_config(**kwargs), self._callback)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


def _with_telemetry(client: ChatOpenAI, callbacks: list[Any]) -> Any:
    """Attach deterministic invocation telemetry without callback dispatch assumptions."""
    if not callbacks:
        return client
    if len(callbacks) != 1 or not isinstance(callbacks[0], TokenTelemetryCallback):
        raise ValueError("exactly one token telemetry callback is required")
    return _TelemetryChatModel(client, callbacks[0])


def create_chat_llm(
    *,
    temperature: float = 0.1,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    purpose: str = "",
) -> Any:
    """Create the configured chat backend.

    ``purpose`` does not choose the provider.  It only captures provider
    capability needs that must be expressed at construction time.  DeepSeek
    thinking mode currently rejects OpenAI tool_choice/tool calls, so the
    dataset analyzer uses non-thinking chat while ordinary proposal/debugger
    calls can still use the configured reasoning mode.
    """
    if settings.DISABLE_LLM:
        raise RuntimeError("LLM requests are disabled by EDGECRAFT_DISABLE_LLM=1")
    purpose = (purpose or "").strip().lower()
    provider = provider or active_llm_provider()
    kwargs: Dict[str, Any] = {
        "temperature": temperature,
        "timeout": settings.LLM_REQUEST_TIMEOUT_S,
        "max_retries": settings.LLM_MAX_RETRIES,
    }
    if settings.LLM_MAX_OUTPUT_TOKENS > 0:
        kwargs["max_tokens"] = settings.LLM_MAX_OUTPUT_TOKENS

    if provider == "deepseek":
        resolved_model = model or settings.DEEPSEEK_MODEL
        callbacks = _telemetry_callbacks(
            provider=provider, model=resolved_model, purpose=purpose
        )
        kwargs.update(
            {
                "model": resolved_model,
                "api_key": settings.DEEPSEEK_API_KEY,
                "base_url": settings.DEEPSEEK_API_BASE,
            }
        )
        if callbacks:
            kwargs["include_response_headers"] = True
        supports_thinking = purpose not in _TOOL_CALLING_PURPOSES
        thinking_enabled = supports_thinking and settings.DEEPSEEK_THINKING_ENABLED
        if thinking_enabled and settings.DEEPSEEK_REASONING_EFFORT:
            kwargs["reasoning_effort"] = settings.DEEPSEEK_REASONING_EFFORT
        if thinking_enabled:
            kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
        else:
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        return _with_telemetry(ChatOpenAI(**kwargs), callbacks)

    if provider == "poe":
        resolved_model = model or settings.POE_MODEL
        callbacks = _telemetry_callbacks(
            provider=provider, model=resolved_model, purpose=purpose
        )
        kwargs.update(
            {
                "model": resolved_model,
                "api_key": settings.POE_API_KEY or settings.OPENAI_API_KEY,
                "base_url": settings.POE_API_BASE,
            }
        )
        if resolved_model.startswith("deepseek-v4-"):
            kwargs["extra_body"] = {
                "enable_thinking": settings.POE_THINKING_ENABLED,
            }
        if callbacks:
            kwargs["include_response_headers"] = True
        return _with_telemetry(ChatOpenAI(**kwargs), callbacks)

    resolved_model = model or settings.OPENAI_MODEL
    callbacks = _telemetry_callbacks(
        provider=provider, model=resolved_model, purpose=purpose
    )
    kwargs.update(
        {
            "model": resolved_model,
            "api_key": settings.OPENAI_API_KEY,
        }
    )
    if settings.OPENAI_API_BASE:
        kwargs["base_url"] = settings.OPENAI_API_BASE
    if callbacks:
        kwargs["include_response_headers"] = True
    return _with_telemetry(ChatOpenAI(**kwargs), callbacks)
