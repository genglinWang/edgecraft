"""Shared Docker CLI fragments for edge jobs.

Keep in sync with:
- ``EdgeRunner._generate_infer_run_script`` (synth / infer benchmark)
- ``OfflineProfiler._generate_run_script_docker`` (skip_pip_install / pre-built image)
"""

from __future__ import annotations


def gpu_opts_shell(has_gpu: bool) -> str:
    """Same GPU flags as production ``run.sh`` generators."""
    # Docker drops the host user's supplementary groups when --user is set.
    # Jetson GPU device nodes are owned by the video group.
    return "--runtime=nvidia --gpus all --group-add video" if has_gpu else ""


def edge_runner_infer_docker_opts_bash(has_gpu: bool) -> str:
    """``DOCKER_OPTS=...`` value for infer / synth edge jobs (includes ``--network=host``)."""
    gpu_opts = gpu_opts_shell(has_gpu)
    return (
        "--rm --user $(id -u):$(id -g) -e HOME=/workspace -e XDG_CACHE_HOME=/workspace/.cache "
        "-e HF_HOME=/workspace/.cache/huggingface -e TRANSFORMERS_CACHE=/workspace/.cache/huggingface "
        f"-e TORCH_HOME=/workspace/.cache/torch -v $(pwd):/workspace -w /workspace --network=host {gpu_opts}"
    )


def offline_profiler_prebuilt_docker_args_bash(has_gpu: bool) -> str:
    """Arguments between ``docker run`` and the image ref for pre-built profile jobs (no ``--network=host``)."""
    gpu_opts = gpu_opts_shell(has_gpu)
    # Mount host tegrastats for Jetson power measurement (no-op if binary absent)
    tegra = "-v /usr/bin/tegrastats:/usr/bin/tegrastats:ro" if has_gpu else ""
    return (
        f"--rm --privileged --user $(id -u):$(id -g) -e HOME=/workspace -e XDG_CACHE_HOME=/workspace/.cache "
        "-e HF_HOME=/workspace/.cache/huggingface -e TRANSFORMERS_CACHE=/workspace/.cache/huggingface "
        f"-e TORCH_HOME=/workspace/.cache/torch -v $(pwd):/workspace -w /workspace {tegra} {gpu_opts}"
    )
