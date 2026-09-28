"""Knowledge module for EdgeCraft: RAG, CBR, and device facts.

RAG/CBR backends pull optional heavy dependencies such as vector stores.  Keep
package import cheap so independent knowledge stores can be used in lightweight
tests and scheduler/verifier paths.
"""

try:
    from .rag import RAGRetriever, MultiSourceRetriever
except ImportError:  # optional vector-store dependencies may be absent
    RAGRetriever = None
    MultiSourceRetriever = None

try:
    from .cbr import CaseStore, CaseRetriever, Case, CaseCrawler
except ImportError:
    CaseStore = None
    CaseRetriever = None
    Case = None
    CaseCrawler = None

try:
    from .orchestrator import KnowledgeOrchestrator, RetrievedKnowledge, get_knowledge_for_planning
except ImportError:
    KnowledgeOrchestrator = None
    RetrievedKnowledge = None
    get_knowledge_for_planning = None

__all__ = [
    "RAGRetriever",
    "MultiSourceRetriever",
    "CaseStore",
    "Case",
    "CaseRetriever",
    "CaseCrawler",
    "KnowledgeOrchestrator",
    "RetrievedKnowledge",
    "get_knowledge_for_planning",
]
