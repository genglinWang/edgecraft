#!/usr/bin/env python3
"""Check source releases for credentials, private infrastructure, and run data."""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve()
SKIP_PARTS = {".git", "__pycache__", ".pytest_cache", ".venv", "venv", "build", "dist"}
FORBIDDEN_RELEASE_SUFFIXES = {
    ".7z",
    ".bin",
    ".ckpt",
    ".csv",
    ".db",
    ".engine",
    ".h5",
    ".hdf5",
    ".jsonl",
    ".log",
    ".npy",
    ".npz",
    ".onnx",
    ".parquet",
    ".pb",
    ".pickle",
    ".pkl",
    ".pt",
    ".pth",
    ".safetensors",
    ".sqlite",
    ".sqlite3",
    ".tflite",
    ".tar",
    ".tgz",
    ".tsv",
    ".zip",
}
FORBIDDEN_RELEASE_ROOTS = {
    "artifacts",
    "checkpoints",
    "collected_results",
    "data",
    "datasets",
    "models",
    "outputs",
    "reviewer_inputs",
    "workspaces",
}
PATTERNS = {
    "provider or repository access token": re.compile(
        r"\b(?:sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{24,}|"
        r"gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|"
        r"AKIA[A-Z0-9]{16})\b"
    ),
    "personal home path": re.compile(r"/(?:Users)/[^/\s]+/"),
    "private network address": re.compile(
        r"(?<!\d)(?:10(?:\.\d{1,3}){3}|192\.168(?:\.\d{1,3}){2}|"
        r"172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2})(?!\d)"
    ),
    "private key material": re.compile(r"BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY"),
    "named SSH identity file": re.compile(r"\bid_(?:rsa|dsa|ecdsa|ed25519)\b", re.I),
    "credential backup name": re.compile(
        r"(?:credential|secret|token|api)[_-]?(?:backup|dump)|\.env\.(?:bak|backup|old)",
        re.I,
    ),
    "machine-specific storage path": re.compile(
        r"/(?:nvme|raid|scratch)[^\s'\"]*/", re.I
    ),
    "personal mailbox": re.compile(
        r"[A-Z0-9._%+-]+@(?:gmail|outlook|hotmail|icloud|qq|163)\.[A-Z]{2,}", re.I
    ),
}


def release_files() -> list[Path]:
    return sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and path.resolve() != SELF
        and not any(part in SKIP_PARTS for part in path.relative_to(ROOT).parts)
        and not any(part.endswith(".egg-info") for part in path.relative_to(ROOT).parts)
    )


def text_files() -> list[Path]:
    return [path for path in release_files() if path.stat().st_size <= 2_000_000]


def main() -> int:
    findings: list[tuple[str, int, str]] = []
    files = release_files()
    for path in files:
        relative = path.relative_to(ROOT)
        suffixes = {suffix.lower() for suffix in path.suffixes}
        if path.name == ".env" or path.name.startswith(".env.") and path.name != ".env.example":
            findings.append((str(relative), 0, "local environment file"))
        if relative.parts and relative.parts[0] in FORBIDDEN_RELEASE_ROOTS:
            findings.append((str(relative), 0, "experiment/artifact directory"))
        if suffixes & FORBIDDEN_RELEASE_SUFFIXES:
            findings.append((str(relative), 0, "model or experiment artifact"))
        if path.stat().st_size > 2_000_000:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for line_number, line in enumerate(text.splitlines(), 1):
            for label, pattern in PATTERNS.items():
                if pattern.search(line):
                    findings.append((str(path.relative_to(ROOT)), line_number, label))
    if findings:
        for path, line, label in findings:
            print(f"{path}:{line}: {label}")
        print(f"release scan failed with {len(findings)} finding(s)", file=sys.stderr)
        return 1
    print(f"release scan passed ({len(files)} files checked)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
