"""Multi-source RAG retriever for EdgeCraft.

This module implements retrieval from multiple knowledge sources:
- Web search (Google/SerpAPI)
- arXiv papers
- Kaggle notebooks

Inspired by automl-agent's retriever.py implementation.
"""
import json
from typing import List, Dict, Any, Optional
from loguru import logger

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document

from edgecraft.config.settings import settings
from edgecraft.core.task import UserSpec
from edgecraft.utils.llm import create_chat_llm


CURATED_EDGECRAFT_KNOWLEDGE: List[Dict[str, Any]] = [
    {
        "id": "vision_classification_public_apis",
        "tasks": {"classification", "image_classification"},
        "modalities": {"vision"},
        "content": (
            "For vision classification with MobileNet/EfficientNet/ResNet model names, "
            "use timm or torchvision public APIs and ImageFolder-style datasets. "
            "Do not pass timm model names such as mobilenetv3_large_100 to Ultralytics YOLO()."
        ),
    },
    {
        "id": "whisper_current_transformers_api",
        "tasks": {"speech_recognition", "automatic_speech_recognition"},
        "modalities": {"audio"},
        "content": (
            "Current Transformers WhisperProcessor no longer supports as_target_processor. "
            "Create labels with processor.tokenizer(...) or text_target-style tokenization, "
            "and collate input_features/labels with processor feature_extractor/tokenizer padding."
        ),
    },
    {
        "id": "jetson_onnxruntime_provider_policy",
        "tasks": {"object_detection", "segmentation", "classification", "text_classification", "anomaly_detection"},
        "modalities": {"vision", "text", "structured"},
        "content": (
            "On Jetson edge jobs, do not pip install onnxruntime-gpu at runtime. "
            "Use the Docker image's preflight-reported onnxruntime providers, selecting "
            "TensorrtExecutionProvider, CUDAExecutionProvider, then CPUExecutionProvider when available."
        ),
    },
    {
        "id": "onnx_external_data_staging",
        "tasks": {"text_classification", "classification", "anomaly_detection"},
        "modalities": {"text", "structured", "vision"},
        "content": (
            "ONNX models may be exported with external tensor data. Stage best.onnx together "
            "with same-stem sidecars such as best.onnx.data or best.onnx.data files in the same outputs directory."
        ),
    },
    {
        "id": "stable_segmentation_family",
        "tasks": {"segmentation"},
        "modalities": {"vision"},
        "content": (
            "For stable development runs, prefer standard YOLO segmentation models such as "
            "yolo11n-seg/yolo11s-seg over YOLOe variants that may auto-install CLIP during validation."
        ),
    },
    {
        "id": "structured_log_anomaly_schema",
        "tasks": {"anomaly_detection", "text_classification"},
        "modalities": {"structured", "text"},
        "content": (
            "Structured/log anomaly datasets in parquet/csv/json need schema inference. "
            "Identify text/log/message/event columns and label/anomaly columns before choosing a model. "
            "Do not route parquet log anomaly datasets through generic news classification templates without an adapter."
        ),
    },
    {
        "id": "schema_files_contract",
        "tasks": {"classification", "text_classification", "regression", "anomaly_detection"},
        "modalities": {"structured", "text"},
        "content": (
            "When config/data.yaml contains schema.files or files, training code should load those exact files "
            "instead of guessing dataset/train.csv or benchmark-specific names. Support csv, parquet, jsonl, json, "
            "and glob patterns; then infer text/label or numeric target columns from the schema contract."
        ),
    },
    {
        "id": "huggingface_tabular_layout",
        "tasks": {"classification", "text_classification", "regression"},
        "modalities": {"text", "structured"},
        "content": (
            "Many HuggingFace-style datasets are stored as data/train-*.parquet, validation-*.parquet, "
            "test-*.parquet, or dataset_dict.json plus Arrow/Parquet shards. Artifact-first baselines should "
            "read shard files through the materialized schema contract rather than assuming a hand-written CSV."
        ),
    },
    {
        "id": "metric_conventions",
        "tasks": {"object_detection", "segmentation", "anomaly_detection", "speech_recognition"},
        "modalities": {"vision", "audio", "structured", "text"},
        "content": (
            "Metric conventions: mAP means mAP50-95 unless the intent explicitly says mAP50; "
            "AUROC canonicalizes to AUC; F1 canonicalizes to F1Score; WER is lower-is-better; "
            "segmentation metrics should report canonical mIoU or mask mAP aliases consistently."
        ),
    },
]


def _curated_knowledge_for(user_spec: UserSpec, target_device: str = "") -> str:
    """Return local, curated EdgeCraft knowledge relevant to the task."""
    task = user_spec.task_type.value if user_spec.task_type else ""
    modality = user_spec.input_type.value if user_spec.input_type else ""
    selected: List[str] = []
    for card in CURATED_EDGECRAFT_KNOWLEDGE:
        task_match = not task or task in card["tasks"]
        modality_match = not modality or modality in card["modalities"]
        device_match = True
        if "jetson" in card["id"] and target_device:
            device_match = "jetson" in target_device.lower()
        if task_match and modality_match and device_match:
            selected.append(f"- ({card['id']}) {card['content']}")
    if not selected:
        return ""
    return "### Curated EdgeCraft Knowledge\n" + "\n".join(selected)


def get_curated_edgecraft_knowledge(user_spec: UserSpec, target_device: str = "") -> str:
    """Public helper for prompt-time local knowledge injection."""
    return _curated_knowledge_for(user_spec, target_device)


class MultiSourceRetriever:
    """Retriever that aggregates knowledge from multiple sources."""

    def __init__(self, llm = None):
        """Initialize the multi-source retriever.

        Args:
            llm: Optional LLM for summarization. Uses default if not provided.
        """
        self.llm = llm or create_chat_llm(temperature=0.3, purpose="rag_summary")

    def retrieve_web(
        self,
        query: str,
        top_k: int = 10
    ) -> str:
        """Retrieve knowledge from web search.

        Args:
            query: Search query.
            top_k: Number of results to fetch.

        Returns:
            Summarized knowledge from web search.
        """
        try:
            from serpapi import GoogleSearch
        except ImportError:
            logger.warning("serpapi not installed. Skipping web search.")
            return ""

        serpapi_key = getattr(settings, 'SERPAPI_KEY', None)
        if not serpapi_key:
            logger.warning("SERPAPI_KEY not configured. Skipping web search.")
            return ""

        try:
            # Generate search query using LLM
            query_prompt = f"""Generate a specific Google search query (max 10 words) for finding
edge AI deployment solutions related to: {query}

Focus on practical implementation, model optimization, and edge device deployment.
Return only the search query, nothing else."""

            response = self.llm.invoke([HumanMessage(content=query_prompt)])
            search_query = response.content.strip().replace('"', '').replace('.', '')

            logger.info(f"Web search query: {search_query}")

            # Perform search
            params = {
                "engine": "google",
                "q": search_query,
                "api_key": serpapi_key,
                "num": top_k
            }
            search = GoogleSearch(params)
            results = search.get_dict().get("organic_results", [])

            if not results:
                return ""

            # Filter and fetch content
            domain_blocklist = ["youtube.com", "twitter.com", "x.com", "facebook.com"]
            filtered_results = [
                r for r in results
                if not any(d in r.get("link", "") for d in domain_blocklist)
            ][:top_k]

            # Build context from snippets
            context_parts = []
            for r in filtered_results:
                title = r.get("title", "")
                snippet = r.get("snippet", "")
                link = r.get("link", "")
                context_parts.append(f"**{title}**\n{snippet}\nSource: {link}\n")

            context = "\n".join(context_parts)

            # Summarize
            return self._summarize_source(
                source_name="Web Search",
                context=context,
                query=query
            )

        except Exception as e:
            logger.warning(f"Web search failed: {e}")
            return ""

    def retrieve_arxiv(
        self,
        task_type: str,
        domain: str = "edge AI",
        top_k: int = 5
    ) -> str:
        """Retrieve knowledge from arXiv papers.

        Args:
            task_type: Task type (e.g., "object detection").
            domain: Application domain.
            top_k: Number of papers to fetch.

        Returns:
            Summarized knowledge from arXiv.
        """
        try:
            import arxivloader
        except ImportError:
            logger.warning("arxivloader not installed. Skipping arXiv search.")
            return ""

        try:
            # Build query
            task_kw = task_type.replace("-", " ").replace("_", " ")
            query = f'search_query=all:"{task_kw}" AND all:"{domain}" AND (cat:cs.AI OR cat:cs.CV OR cat:cs.LG)'

            logger.info(f"arXiv query: {query}")

            df = arxivloader.load(query, num=top_k, sortBy="submittedDate", verbosity=0)

            if df.empty:
                return ""

            # Build context from abstracts
            context_parts = []
            for _, row in df.iterrows():
                title = row.get("title", "")
                abstract = row.get("summary", "")[:500]
                context_parts.append(f"**{title}**\n{abstract}\n")

            context = "\n".join(context_parts)

            return self._summarize_source(
                source_name="arXiv Papers",
                context=context,
                query=f"{task_type} {domain}"
            )

        except Exception as e:
            logger.warning(f"arXiv search failed: {e}")
            return ""

    def retrieve_kaggle(
        self,
        task_type: str,
        domain: str = "edge",
        top_k: int = 5
    ) -> str:
        """Retrieve knowledge from Kaggle notebooks.

        Args:
            task_type: Task type.
            domain: Application domain.
            top_k: Number of notebooks to fetch.

        Returns:
            Summarized knowledge from Kaggle.
        """
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
        except ImportError:
            logger.warning("Kaggle API not installed. Skipping Kaggle search.")
            return ""

        try:
            api = KaggleApi()
            api.authenticate()

            search_term = f"{task_type} {domain}"
            logger.info(f"Kaggle search: {search_term}")

            # Kaggle API v2 uses kernels_list() (no kernels_list_with_http_info)
            notebooks = api.kernels_list(
                search=search_term,
                sort_by="relevance",
                language="python",
                page_size=top_k,
            ) or []
            notebooks = [nb for nb in notebooks if nb is not None]

            if not notebooks:
                return ""

            # ApiKernelMetadata has .ref and .title attributes
            def _ref(nb):
                return getattr(nb, "ref", None) or (nb.get("ref", "") if isinstance(nb, dict) else "")
            def _title(nb):
                return getattr(nb, "title", None) or (nb.get("title", "") if isinstance(nb, dict) else "")

            context_parts = []
            for nb in notebooks:
                title = _title(nb) or ""
                ref = _ref(nb) or ""
                context_parts.append(f"**{title}**\nNotebook: {ref}\n")

            context = "\n".join(context_parts)

            return self._summarize_source(
                source_name="Kaggle Notebooks",
                context=context,
                query=search_term
            )

        except Exception as e:
            logger.warning(f"Kaggle search failed: {e}")
            return ""

    def _summarize_source(
        self,
        source_name: str,
        context: str,
        query: str
    ) -> str:
        """Summarize content from a single source.

        Args:
            source_name: Name of the source.
            context: Raw context to summarize.
            query: Original query for context.

        Returns:
            Summarized knowledge.
        """
        if not context.strip():
            return ""

        prompt = f"""Summarize the following content from {source_name} into actionable insights
for edge AI development. Focus on:
- Model selection recommendations
- Optimization techniques (quantization, pruning, TensorRT)
- Deployment best practices
- Performance benchmarks

Query: {query}

Content:
{context[:4000]}

Provide a concise summary (3-5 bullet points) with practical recommendations."""

        try:
            response = self.llm.invoke([
                SystemMessage(content="You are an expert in edge AI deployment and optimization."),
                HumanMessage(content=prompt)
            ])
            return response.content.strip()
        except Exception as e:
            logger.warning(f"Summarization failed for {source_name}: {e}")
            return ""

    def retrieve_all(
        self,
        user_spec: UserSpec,
        target_device: str = "",
        enable_web: bool = True,
        enable_arxiv: bool = True,
        enable_kaggle: bool = True
    ) -> str:
        """Retrieve and combine knowledge from all sources.

        Args:
            user_spec: Edge specification with task details.
            target_device: Target deployment device.
            enable_web: Whether to search web.
            enable_arxiv: Whether to search arXiv.
            enable_kaggle: Whether to search Kaggle.

        Returns:
            Combined knowledge summary.
        """
        # Build query from user_spec
        task_type = user_spec.task_type.value if user_spec.task_type else "classification"
        modality = user_spec.input_type.value if user_spec.input_type else "vision"
        description = user_spec.description

        query = f"{modality} {task_type} edge deployment"
        if target_device:
            query += f" {target_device}"

        summaries = []
        curated = _curated_knowledge_for(user_spec, target_device)
        if curated:
            summaries.append(curated)

        # Retrieve from each source
        if enable_web:
            web_summary = self.retrieve_web(query)
            if web_summary:
                summaries.append(f"### Web Search Insights\n{web_summary}")

        if enable_arxiv:
            arxiv_summary = self.retrieve_arxiv(task_type, domain=f"edge {modality}")
            if arxiv_summary:
                summaries.append(f"### Research Papers Insights\n{arxiv_summary}")

        if enable_kaggle:
            kaggle_summary = self.retrieve_kaggle(task_type, domain="edge deployment")
            if kaggle_summary:
                summaries.append(f"### Kaggle Community Insights\n{kaggle_summary}")

        if not summaries:
            return ""

        # Combine all summaries
        combined = "\n\n".join(summaries)

        # Final synthesis
        synthesis_prompt = f"""Based on the following knowledge gathered from multiple sources,
provide a unified set of recommendations for this edge AI task:

Task: {description}
Target Device: {target_device or 'General edge device'}
Modality: {modality}
Task Type: {task_type}

Gathered Knowledge:
{combined}

Synthesize into 3-5 key recommendations that are most relevant to this specific task.
Focus on actionable advice for model selection, training, and deployment."""

        try:
            response = self.llm.invoke([
                SystemMessage(content="You are a senior edge AI consultant providing practical recommendations."),
                HumanMessage(content=synthesis_prompt)
            ])
            return response.content.strip()
        except Exception as e:
            logger.warning(f"Final synthesis failed: {e}")
            return combined  # Return raw combined summaries as fallback


def get_rag_knowledge(
    user_spec: UserSpec,
    target_device: str = ""
) -> str:
    """Convenience function to retrieve RAG knowledge.

    Args:
        user_spec: Edge specification.
        target_device: Target device.

    Returns:
        Retrieved knowledge string.
    """
    retriever = MultiSourceRetriever()
    return retriever.retrieve_all(user_spec, target_device)
