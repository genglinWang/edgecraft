import pytest


def test_api_off_switch_blocks_client_creation(monkeypatch):
    pytest.importorskip("langchain_openai")
    from edgecraft.config.settings import settings
    from edgecraft.utils.llm import create_chat_llm, llm_credentials_available

    monkeypatch.setattr(settings, "DISABLE_LLM", True)
    assert not llm_credentials_available()
    with pytest.raises(RuntimeError, match="LLM requests are disabled"):
        create_chat_llm()
