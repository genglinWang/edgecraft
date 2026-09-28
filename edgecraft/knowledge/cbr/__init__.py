"""Case-Based Reasoning (CBR) module with lazy optional dependencies."""
from __future__ import annotations

from typing import Any

__all__ = ["CaseStore", "Case", "CaseRetriever", "CaseCrawler"]


def __getattr__(name: str) -> Any:
    if name in {"CaseStore", "Case"}:
        from .case_store import Case, CaseStore

        return {"CaseStore": CaseStore, "Case": Case}[name]
    if name == "CaseRetriever":
        from .retriever import CaseRetriever

        return CaseRetriever
    if name == "CaseCrawler":
        from .case_crawler import CaseCrawler

        return CaseCrawler
    raise AttributeError(name)
