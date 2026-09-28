import os
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, ConfigDict
from dotenv import load_dotenv

# Force ONNX Runtime to skip TensorRT when libnvinfer is not installed (e.g. dev machines).
# ChromaDB's default embedding uses ONNX; without this, ORT tries TensorRT first and logs an error
# before falling back to CPU. Set before any import that loads onnxruntime.
os.environ.setdefault("ORT_TENSORRT_DISABLED", "1")

# Load only the repository-local environment file. Evaluation launchers accept
# explicit manifests and environment variables; they never discover host files.
_package_root = Path(__file__).resolve().parents[2]
workspace_root = Path(os.getenv("EDGECRAFT_ROOT", str(Path.cwd()))).expanduser().resolve()
load_dotenv(workspace_root / ".env")


def _default_runtime_tmp_dir() -> str:
    if os.name == "posix":
        return os.path.join("/tmp", f"edgecraft-{os.getuid()}")
    return str(workspace_root / "tmp")


def edgecraft_env(name: str, default: str = "") -> str:
    """Read an EdgeCraft setting or its default."""
    return os.getenv(f"EDGECRAFT_{name}", default)


_edgecraft_env = edgecraft_env


class Settings(BaseModel):
    """EdgeCraft configuration settings."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # App
    APP_NAME: str = "EdgeCraft"
    DEBUG: bool = False
    PROFILE: str = _edgecraft_env("PROFILE", "paper").strip().lower()

    # LLM backend.
    DISABLE_LLM: bool = _edgecraft_env("DISABLE_LLM", "0").lower() in {"1", "true", "yes", "on"}
    LLM_PROVIDER: str = _edgecraft_env(
        "LLM_PROVIDER", os.getenv("LLM_PROVIDER", "openai")
    ).strip().lower()

    # OpenAI / generic OpenAI-compatible backend
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_API_BASE: str = os.getenv("OPENAI_API_BASE", "")  # For proxy support
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-5.4")

    # POE OpenAI-compatible backend
    POE_API_KEY: str = os.getenv("POE_API_KEY", "")
    POE_API_BASE: str = os.getenv("POE_API_BASE", os.getenv("OPENAI_API_BASE", "https://api.poe.com/v1"))
    POE_MODEL: str = os.getenv("POE_MODEL", os.getenv("OPENAI_MODEL", "gpt-5-chat"))
    POE_THINKING_ENABLED: bool = os.getenv("POE_THINKING_ENABLED", "1").lower() in (
        "1", "true", "yes", "on"
    )

    # DeepSeek OpenAI-compatible backend
    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
    DEEPSEEK_API_BASE: str = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")
    DEEPSEEK_MODEL: str = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro")
    DEEPSEEK_REASONING_EFFORT: str = os.getenv("DEEPSEEK_REASONING_EFFORT", "high")
    DEEPSEEK_THINKING_ENABLED: bool = os.getenv("DEEPSEEK_THINKING_ENABLED", "1").lower() in ("1", "true", "yes", "on")
    LLM_REQUEST_TIMEOUT_S: int = int(_edgecraft_env("LLM_REQUEST_TIMEOUT_S", "180"))
    LLM_MAX_RETRIES: int = int(_edgecraft_env("LLM_MAX_RETRIES", "2"))
    LLM_MAX_OUTPUT_TOKENS: int = int(_edgecraft_env("LLM_MAX_OUTPUT_TOKENS", "0"))
    TOKEN_TELEMETRY_PATH: str = _edgecraft_env("TOKEN_TELEMETRY_PATH").strip()
    REQUIRE_TOKEN_TELEMETRY: bool = _edgecraft_env(
        "REQUIRE_TOKEN_TELEMETRY", "1"
    ).lower() in ("1", "true", "yes", "on")
    EXPERIMENT_RUN_ID: str = _edgecraft_env("EXPERIMENT_RUN_ID").strip()
    DEBUGGER_ENABLED: bool = _edgecraft_env("DEBUGGER_ENABLED", "1").lower() in ("1", "true", "yes", "on")
    DEBUGGER_MAX_RETRIES: int = int(_edgecraft_env("DEBUGGER_MAX_RETRIES", "2"))
    # The paper profile permits the candidate to choose a substantive recipe;
    # the explicit offline profile lowers this cap for smoke checks.
    MAX_TRAIN_EPOCHS: int = int(_edgecraft_env("MAX_TRAIN_EPOCHS", "100"))
    REQUIRE_SPLIT_MANIFEST_CONSUMPTION: bool = _edgecraft_env(
        "REQUIRE_SPLIT_MANIFEST_CONSUMPTION", "1"
    ).lower() in ("1", "true", "yes", "on")
    # Edge benchmark defaults to dataset-driven evaluation when dataset is available.
    EDGE_EVAL_WITH_DATASET: bool = _edgecraft_env("EDGE_EVAL_WITH_DATASET", "1").lower() in ("1", "true", "yes", "on")
    # Long-running edge jobs (TRT build + eval) need a relaxed default timeout.
    EDGE_RUN_TIMEOUT_S: int = int(_edgecraft_env("EDGE_RUN_TIMEOUT_S", "2400"))
    LOADER_STAGE_TIMEOUT_S: int = int(_edgecraft_env("LOADER_STAGE_TIMEOUT_S", "300"))
    FULL_TRAIN_EXPORT_TIMEOUT_S: int = int(_edgecraft_env("FULL_TRAIN_EXPORT_TIMEOUT_S", "7200"))
    HOST_INFER_TIMEOUT_S: int = int(_edgecraft_env("HOST_INFER_TIMEOUT_S", "180"))
    REQUIRE_EDGE_QUALITY_PARITY: bool = _edgecraft_env(
        "REQUIRE_EDGE_QUALITY_PARITY", "1"
    ).strip().lower() in {"1", "true", "yes", "on"}
    EDGE_QUALITY_PARITY_ABS_TOL: float = float(
        _edgecraft_env("EDGE_QUALITY_PARITY_ABS_TOL", "0.001")
    )
    EDGE_QUALITY_PARITY_REL_TOL: float = float(
        _edgecraft_env("EDGE_QUALITY_PARITY_REL_TOL", "0.02")
    )
    TREE_BRANCHING_FACTOR: int = int(_edgecraft_env("TREE_BRANCHING_FACTOR", "3"))
    # Read-only compatibility field for older internal imports. The public
    # configuration name and active controller use TREE_BRANCHING_FACTOR.
    MCTS_BRANCHING_FACTOR: int = TREE_BRANCHING_FACTOR
    TEMPLATE_MODE: str = _edgecraft_env("TEMPLATE_MODE", "guidance").strip().lower()
    SOLUTION_ZOO_MODE: str = _edgecraft_env("SOLUTION_ZOO_MODE", "off").strip().lower()
    MULTIFIDELITY_MODE: str = _edgecraft_env("MULTIFIDELITY_MODE", "ladder").strip().lower()
    VERIFIER_MODE: str = _edgecraft_env(
        "VERIFIER_MODE",
        _edgecraft_env("MULTIFIDELITY_MODE", "ladder"),
    ).strip().lower()
    VERIFIER_PROBES: str = _edgecraft_env("VERIFIER_PROBES", "static_artifact,efficiency,full")
    VERIFIER_CALIBRATION_ERROR: str = _edgecraft_env("VERIFIER_CALIBRATION_ERROR", "")
    VERIFIER_CALIBRATION_PATH: str = _edgecraft_env("VERIFIER_CALIBRATION_PATH", "")
    VERIFIER_CALIBRATION_SEED_PATH: str = _edgecraft_env(
        "VERIFIER_CALIBRATION_SEED_PATH", ""
    )
    VERIFIER_KAPPA: float = float(_edgecraft_env("VERIFIER_KAPPA", "2.0"))
    VERIFIER_QUALITY_STEPS: int = int(_edgecraft_env("VERIFIER_QUALITY_STEPS", "100"))
    COMPAT_RULES: str = _edgecraft_env("COMPAT_RULES", "enforce").strip().lower()
    EXPANSION_MODE: str = _edgecraft_env("EXPANSION_MODE", "constraint_directed").strip().lower()
    COMPAT_RULE_STORE_PATH: str = _edgecraft_env("COMPAT_RULE_STORE_PATH", "")
    COMPAT_RULE_SEED_PATH: str = _edgecraft_env("COMPAT_RULE_SEED_PATH", "")
    RULE_MIN_SUPPORT: int = int(_edgecraft_env("RULE_MIN_SUPPORT", "2"))
    L1_PRUNE_MARGIN: float = float(_edgecraft_env("L1_PRUNE_MARGIN", "0.2"))
    L1_PROBE_TIMEOUT_S: int = int(_edgecraft_env("L1_PROBE_TIMEOUT_S", "300"))
    L1_EDGE_TIMEOUT_S: int = int(_edgecraft_env("L1_EDGE_TIMEOUT_S", "600"))
    L1_FALSE_PRUNE_SAMPLE: bool = _edgecraft_env("L1_FALSE_PRUNE_SAMPLE", "0").lower() in ("1", "true", "yes", "on")
    LLM_TRACE_ENABLED: bool = _edgecraft_env("LLM_TRACE", "0").lower() in ("1", "true", "yes", "on")
    NORMALIZE_GENERATED_CODE: bool = _edgecraft_env("NORMALIZE_GENERATED_CODE", "1").lower() in ("1", "true", "yes", "on")
    BRANCH_SELECTION_MODE: str = _edgecraft_env("BRANCH_SELECTION_MODE", "llm_strict").strip().lower()
    BRANCH_JUDGMENT_MODE: str = _edgecraft_env("BRANCH_JUDGMENT_MODE", "llm_strict").strip().lower()
    EXECUTION_BACKEND: str = _edgecraft_env("EXECUTION_BACKEND", "scheduler").strip().lower()
    SCHEDULER_POLICY: str = _edgecraft_env("SCHEDULER_POLICY", "cost_aware").strip().lower()
    CBR_SCOPE: str = _edgecraft_env("CBR_SCOPE", "tenant").strip().lower()
    CBR_READ_ONLY: bool = _edgecraft_env("CBR_READ_ONLY", "0").lower() in (
        "1", "true", "yes", "on"
    )
    CBR_CASE_PATH: str = _edgecraft_env("CBR_CASE_PATH", "").strip()
    COMPAT_RULE_STORE_READ_ONLY: bool = _edgecraft_env(
        "COMPAT_RULE_STORE_READ_ONLY", "0"
    ).lower() in ("1", "true", "yes", "on")
    TASK_MAX_WORKERS: int = int(_edgecraft_env("TASK_MAX_WORKERS", "2"))
    API_TENANT_TOKENS: str = _edgecraft_env("API_TENANT_TOKENS", "").strip()
    TENANT_DATA_ROOT: str = _edgecraft_env(
        "TENANT_DATA_ROOT", str(workspace_root / "tenant-data")
    ).strip()
    TENANT_CREDENTIAL_ROOT: str = _edgecraft_env(
        "TENANT_CREDENTIAL_ROOT", str(workspace_root / ".edgecraft" / "tenant-credentials")
    ).strip()
    SCHEDULER_JOB_TIMEOUT_S: float = float(
        _edgecraft_env("SCHEDULER_JOB_TIMEOUT_S", "0") or "0"
    )
    PENDING_EDGE_ARCHIVE_DIR: str = _edgecraft_env(
        "PENDING_EDGE_ARCHIVE_DIR", ""
    ).strip()
    RUNTIME_TMP_DIR: str = _edgecraft_env(
        "RUNTIME_TMP_DIR",
        _default_runtime_tmp_dir(),
    )
    KEEP_EDGE_JOB_DIR: bool = _edgecraft_env("KEEP_EDGE_JOB_DIR", "0").lower() in ("1", "true", "yes", "on")

    # Optional: RAG / knowledge
    SERPAPI_KEY: str = os.getenv("SERPAPI_KEY", "")

    # Paths
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    EDGECRAFT_ROOT: str = str(workspace_root)
    DATA_ROOT: str = os.getenv("DATA_ROOT", str(workspace_root / "data"))
    OUTPUT_DIR: str = os.getenv("OUTPUT_DIR", str(workspace_root / "outputs"))

    @property
    def DATA_DIR(self) -> str:
        return str(Path(self.DATA_ROOT).expanduser())

    @property
    def ARTIFACTS_DIR(self) -> str:
        return str(Path(self.OUTPUT_DIR).expanduser() / "artifacts")

    @property
    def CHROMA_PERSIST_DIRECTORY(self) -> str:
        return os.path.join(self.DATA_DIR, "chroma")

settings = Settings()

os.makedirs(settings.RUNTIME_TMP_DIR, exist_ok=True)
os.environ.setdefault("TMPDIR", settings.RUNTIME_TMP_DIR)
os.environ.setdefault("TMP", settings.RUNTIME_TMP_DIR)
os.environ.setdefault("TEMP", settings.RUNTIME_TMP_DIR)
os.environ.setdefault("HF_HOME", str(workspace_root / ".cache" / "huggingface"))
os.environ.setdefault("TORCH_HOME", str(workspace_root / ".cache" / "torch"))
os.environ.setdefault("ULTRALYTICS_CONFIG_DIR", str(workspace_root / ".cache" / "ultralytics"))
