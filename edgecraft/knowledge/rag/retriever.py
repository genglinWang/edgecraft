"""RAG Retriever for the EdgeCraft knowledge base."""
import re
from typing import List, Dict, Any, Optional
from .vectorstore import VectorStore


class RAGRetriever:
    """Retriever for domain knowledge using RAG."""

    def __init__(self, collection_name: str = "edgecraft_knowledge"):
        """Initialize the RAG retriever.

        Args:
            collection_name: Name of the ChromaDB collection.
        """
        self.vectorstore = VectorStore(collection_name=collection_name)
        self._initialized = False

    def initialize_with_docs(self, docs: List[Dict[str, str]]) -> None:
        """Initialize the knowledge base with documents.

        Args:
            docs: List of dicts with 'content', 'metadata', 'id' keys.
        """
        if self.vectorstore.count() == 0:
            documents = [d["content"] for d in docs]
            metadatas = [dict(d.get("metadata", {}), id=d.get("id", f"doc_{i}")) for i, d in enumerate(docs)]
            ids = [d.get("id", f"doc_{i}") for i, d in enumerate(docs)]

            self.vectorstore.add_documents(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
        else:
            # Keep the seed corpus append-only.  Older local Chroma stores may
            # contain the first few documents only; add new seed docs without
            # resetting user knowledge.
            for i, doc in enumerate(docs):
                try:
                    self.vectorstore.add_documents(
                        documents=[doc["content"]],
                        metadatas=[dict(doc.get("metadata", {}), id=doc.get("id", f"doc_{i}"))],
                        ids=[doc.get("id", f"doc_{i}")],
                    )
                except Exception:
                    pass

        self._initialized = True

    def retrieve(
        self,
        query: str,
        n_results: int = 3,
        filter_by: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for a query.

        Args:
            query: Query string.
            n_results: Number of results to return.
            filter_by: Optional metadata filter.

        Returns:
            List of relevant documents with metadata.
        """
        results = self.vectorstore.query(
            query_text=query,
            n_results=n_results,
            where=filter_by
        )

        return [
            {
                "content": doc,
                "metadata": meta,
                "relevance": 1 - dist  # Convert distance to relevance score
            }
            for doc, meta, dist in zip(
                results["documents"],
                results["metadatas"],
                results["distances"]
            )
        ]

    def add_knowledge(
        self,
        content: str,
        metadata: Dict[str, Any] = None,
        doc_id: str = None
    ) -> None:
        """Add a single piece of knowledge to the store.

        Args:
            content: Document content.
            metadata: Optional metadata.
            doc_id: Optional document ID.
        """
        self.vectorstore.add_documents(
            documents=[content],
            metadatas=[metadata or {}],
            ids=[doc_id] if doc_id else None
        )


# Pre-defined local knowledge about edge devices, runtimes, and model choices.
# These are retrieved as evidence.  They must not become scheduler/proposal rules.
EDGE_KNOWLEDGE = [
    {
        "id": "device_jetson_agx_orin",
        "content": """Jetson AGX Orin edge profile:
- Strongest Jetson target in this project: Ampere GPU, large memory, TensorRT and ONNXRuntime are practical.
- Good fit for heavier vision detection/segmentation, visual anomaly, and higher-throughput audio/sensing models.
- Prefer ONNX as a portable artifact; TensorRT engine should be built on the target device when used.""",
        "metadata": {"category": "device_specs", "device": "jetson_agx_orin"}
    },
    {
        "id": "device_jetson_xavier_nx",
        "content": """Jetson Xavier NX edge profile:
- Mid-range Jetson target with Volta GPU and limited memory compared with AGX Orin.
- Good fit for tiny/small CNNs, lightweight detection, audio keyword/spoken-command models, and sensing models.
- ONNXRuntime CUDA or TensorRT may work, but smaller inputs and simpler operators are safer.""",
        "metadata": {"category": "device_specs", "device": "jetson_xavier_nx"}
    },
    {
        "id": "device_jetson_tx2",
        "content": """Jetson TX2 edge profile:
- Older Pascal-generation Jetson with much tighter memory and compute budget.
- Prefer small input sizes, shallow CNN/MLP/1D CNN models, and conservative ONNX operators.
- Treat heavy segmentation/detection and large transformer-style models as high-risk unless evidence shows they fit.""",
        "metadata": {"category": "device_specs", "device": "jetson_tx2"}
    },
    {
        "id": "device_raspberry_pi_5b",
        "content": """Raspberry Pi 5B edge profile:
- CPU-oriented target in this project; do not assume CUDA or TensorRT.
- Prefer ONNXRuntime CPU, TFLite/LiteRT, small TorchScript-free inference, and compact classical or tiny neural models.
- Good fit for tabular, text hashing, sensing/time-series, and small audio/image classifiers.""",
        "metadata": {"category": "device_specs", "device": "raspberry_pi_5b"}
    },
    {
        "id": "runtime_onnxruntime_provider_policy",
        "content": """ONNXRuntime deployment policy:
- Export ONNX in train.py when possible and benchmark with providers reported by device preflight.
- On Jetson prefer TensorRTExecutionProvider, then CUDAExecutionProvider, then CPUExecutionProvider when available.
- On Raspberry Pi expect CPUExecutionProvider unless a local accelerator/runtime is explicitly available.
- Do not install onnxruntime-gpu inside edge inference jobs.""",
        "metadata": {"category": "runtime", "runtime": "onnxruntime"}
    },
    {
        "id": "runtime_tensorrt_target_build",
        "content": """TensorRT deployment evidence:
- TensorRT engines are hardware-specific and should be built on the target edge device, not on the cloud training server.
- ONNX is the safer exchange artifact between cloud training and edge benchmarking.
- FP16 is usually a lower-risk first optimization than INT8; INT8 needs calibration and should be justified by latency evidence.""",
        "metadata": {"category": "runtime", "runtime": "tensorrt"}
    },
    {
        "id": "runtime_tflite_litert_cpu",
        "content": """TFLite/LiteRT deployment evidence:
- TFLite/LiteRT is a practical CPU-first runtime candidate for Raspberry Pi and small edge classifiers.
- It is most useful for compact CNN/MLP/audio/sensing models with supported operators.
- Prefer it as an additional export/runtime option when ONNXRuntime CPU is slow or unavailable, not as a replacement for ONNX evidence.""",
        "metadata": {"category": "runtime", "runtime": "tflite_litert"}
    },
    {
        "id": "runtime_edge_docker_dependency_evidence",
        "content": """Edge Docker dependency evidence:
- Treat preflight-reported Python package availability as the source of truth for edge infer.py dependencies.
- If a package is reported unavailable in the edge Docker image, do not import it in the edge hot path; choose installed packages or standard-library code instead.
- Common audio/data pitfalls include librosa, soundfile, datasets, pyarrow, pandas, and sklearn being absent or import-broken on edge even when training-server code can import them.""",
        "metadata": {"category": "runtime", "runtime": "edge_docker_dependencies"}
    },
    {
        "id": "model_detection_segmentation_edge",
        "content": """Vision detection/segmentation model evidence:
- Nano/small YOLO-style models are a practical starting point for edge detection and segmentation.
- Industrial crack/defect segmentation often needs correct mask/label representation before backbone tuning matters.
- Lower image size and simpler heads are often better first mutations than complex augmentation or quantization.""",
        "metadata": {"category": "model_selection", "task": "object_detection"}
    },
    {
        "id": "model_image_classification_edge",
        "content": """Image classification model evidence:
- MobileNetV3, EfficientNet-B0, ShuffleNetV2, and small ResNet variants are reasonable edge baselines.
- Pretrained ImageNet initialization is a useful freedom for small visual datasets when download/runtime constraints permit it.
- For unknown datasets, prove the loader and label mapping first, then tune model capacity.""",
        "metadata": {"category": "model_selection", "task": "classification"}
    },
    {
        "id": "model_audio_tiny_edge",
        "content": """Audio edge model evidence:
- For keyword/spoken-command tasks, tiny waveform CNN, mel-spectrogram CNN, MFCC+MLP, and small CRNN/TCN are practical code-space branches.
- Large ASR or SpeechBrain-style pretrained systems may be useful baselines but are often too heavy for artifact-first tiny edge goals.
- Label extraction and representation stability should be measured before residual blocks, mixup, INT8, or complex fine-tuning.""",
        "metadata": {"category": "model_selection", "task": "audio_classification"}
    },
    {
        "id": "model_sensing_timeseries_edge",
        "content": """Sensing/time-series edge model evidence:
- Important representation branches are row-tabular features, sliding windows, statistical features, and 1D sequence models.
- HHAR/MotionSense-style IMU tasks often need window construction and sensor-axis grouping before model capacity tuning.
- Tiny MLP, 1D CNN, TCN, and small GRU are reasonable edge candidates depending on sequence length.""",
        "metadata": {"category": "model_selection", "task": "time_series_classification"}
    },
    {
        "id": "format_ucr_uea_ts_timeseries_v2",
        "content": """UCR/UEA .ts time-series dataset format evidence:
- Header lines start with @problemName, @dimensions, @classLabel, and data begins after @data.
- @seriesLength gives the expected sequence length when present; otherwise infer length from parsed rows and pad/truncate to the observed max or a chosen fixed length.
- Each data row stores dimensions separated by ':'; the final field is the class label when @classLabel is true.
- Parse labels by splitting each row from the right once on ':', then parse the remaining dimension strings as comma-separated floats and stack to (channels, length).
- Do not hard-code 300 or another length unless the dataset evidence says so; Heartbeat-style files may use lengths such as 405.
- Do not parse UCR/UEA .ts as CSV or ImageFolder. Missing values may be '?'; convert or impute them explicitly before training a tiny 1D CNN/TCN/MLP.""",
        "metadata": {
            "category": "dataset_format",
            "format": "ucr_uea_ts",
            "task": "time_series_classification",
        },
    },
    {
        "id": "model_text_log_edge",
        "content": """Text/log edge model evidence:
- Hashing vectorizer, TF-IDF, small linear/MLP heads, and session/block aggregation are useful lightweight branches.
- Large transformers are not required for artifact-first edge baselines and may hurt latency.
- Log anomaly tasks should preserve session/block identity when labels are session-level rather than line-level.""",
        "metadata": {"category": "model_selection", "task": "text_classification"}
    },
    {
        "id": "model_visual_anomaly_edge",
        "content": """Visual anomaly edge model evidence:
- Normal-only reconstruction, patch embeddings, one-class scoring, and lightweight autoencoders are practical industrial anomaly branches.
- Do not treat category-structured visual anomaly datasets as ordinary multi-class classification unless labels support it.
- Artifact evidence should include the scoring threshold or calibration sidecar when the model outputs anomaly scores.""",
        "metadata": {"category": "model_selection", "task": "anomaly_detection"}
    },
]


def get_default_retriever() -> RAGRetriever:
    """Get a pre-initialized RAG retriever with default knowledge."""
    retriever = RAGRetriever()
    retriever.initialize_with_docs(EDGE_KNOWLEDGE)
    return retriever


def _rag_doc_id(doc: Dict[str, Any]) -> str:
    meta = doc.get("metadata", {}) or {}
    return str(meta.get("id") or meta.get("category") or "local_doc")


def _rag_semantic_key(doc: Dict[str, Any]) -> str:
    """Stable key for suppressing stale duplicate seed cards in prompt evidence."""
    meta = doc.get("metadata", {}) or {}
    category = str(meta.get("category") or "")
    for field in ("device", "runtime", "task", "format"):
        value = meta.get(field)
        if value:
            return f"{category}:{field}:{value}"
    doc_id = _rag_doc_id(doc)
    return re.sub(r"_v\d+$", "", doc_id)


def _target_match_score(doc: Dict[str, Any], target_device: str) -> int:
    """Count target-device tokens present in a retrieved doc."""
    tokens = {
        token
        for token in re.split(r"[^a-z0-9]+", (target_device or "").lower())
        if len(token) > 2
    }
    if not tokens:
        return 0
    meta = doc.get("metadata", {}) or {}
    haystack = " ".join([
        _rag_doc_id(doc),
        str(meta.get("device") or ""),
    ]).lower()
    return sum(1 for token in tokens if token in haystack)


def _select_rag_evidence_docs(
    docs: List[Dict[str, Any]],
    target_device: str,
    n_results: int,
) -> List[Dict[str, Any]]:
    """Return compact, device-aware, duplicate-free evidence docs.

    This is only a prompt evidence view.  It does not alter the vector store or
    impose synthesis decisions.
    """
    ranked = sorted(
        docs,
        key=lambda doc: (
            -(
                float(doc.get("relevance", 0.0) or 0.0)
                + 0.10 * _target_match_score(doc, target_device)
            ),
        ),
    )
    selected: List[Dict[str, Any]] = []
    seen = set()
    for doc in ranked:
        key = _rag_semantic_key(doc)
        if key in seen:
            continue
        seen.add(key)
        selected.append(doc)
        if len(selected) >= n_results:
            break
    return selected


def get_local_rag_evidence(
    user_spec: Any,
    target_device: str = "",
    n_results: int = 4,
) -> str:
    """Retrieve compact local RAG evidence for proposal generation.

    This is intentionally a prompt evidence view, not a rule engine.
    """
    if not user_spec:
        return ""
    task = user_spec.task_type.value if getattr(user_spec, "task_type", None) else ""
    modality = user_spec.input_type.value if getattr(user_spec, "input_type", None) else ""
    query = " ".join(
        x for x in [
            getattr(user_spec, "description", ""),
            modality,
            task,
            target_device,
            "dataset format loader edge runtime model hardware docker python package dependency",
        ] if x
    )
    if not query.strip():
        return ""

    retriever = get_default_retriever()
    docs = retriever.retrieve(query, n_results=max(n_results * 3, n_results))
    docs = _select_rag_evidence_docs(docs, target_device, n_results)
    if not docs:
        return ""
    lines = ["### Local RAG Evidence", "Retrieved local hardware/runtime/model notes. Use as evidence, not rules."]
    for doc in docs:
        meta = doc.get("metadata", {}) or {}
        doc_id = _rag_doc_id(doc)
        content = " ".join(str(doc.get("content", "")).split())
        if len(content) > 700:
            content = content[:697] + "..."
        lines.append(f"- ({doc_id}) {content}")
    return "\n".join(lines)
