"""OfflineProfiler: Run random-weight benchmarks for offline profiling.

This module provides:
- Random-weight benchmark templates (ULTRALYTICS_PYTORCH_RANDOM,
  ULTRALYTICS_TENSORRT_RANDOM, TIMM_PYTORCH_RANDOM) with tegrastats energy capture
- profile_local(): Run benchmark in subprocess (local machine)
- profile_remote(): Deploy to edge device with Docker
- profile_registered_models(): Batch profile all registered vision models

Execution modes for profile_remote:
- Docker (default image): pip install ultralytics timm psutil, then benchmark
- Docker (--docker-image): pre-built image, skip pip install
  directly - NO docker, NO pip install. Conda env is pre-built on edge device.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
import inspect
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from edgecraft.config.settings import settings
from edgecraft.core.modality import TaskType
from edgecraft.models.family_registry import DeviceRegistry, FamilyRegistry
from edgecraft.models.profile_store import ProfileRequest, ProfileResult, ProfileStore, get_profile_store
from edgecraft.models.specs import (
    DeviceClass,
    ExportFormat,
    QuantMode,
    RuntimeId,
)
from edgecraft.tools.deploy.docker_opts import offline_profiler_prebuilt_docker_args_bash
from edgecraft.utils.trash import move_to_trash


# ---------------------------------------------------------------------------
# Random-weight benchmark templates (with tegrastats energy capture)
# ---------------------------------------------------------------------------

ULTRALYTICS_PYTORCH_RANDOM = '''#!/usr/bin/env python3
"""Benchmark ultralytics YOLO model with random weights (PyTorch runtime)."""
import json
import time
import sys
import subprocess
import threading
import re

try:
    from ultralytics import YOLO
    import torch
    import numpy as np
    import psutil
except ImportError as e:
    print(json.dumps({{"status": "error", "error_type": "import", "error_message": str(e)}}))
    sys.exit(1)

MODEL_NAME = "{model_name}"
IMGSZ = {imgsz}
WARMUP = {warmup}
ITERATIONS = {iterations}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CAPTURE_ENERGY = True

def _collect_tegrastats(interval_ms, stop_event, power_samples):
    import glob as _g
    use_sysfs = False
    try:
        proc = subprocess.Popen(
            ["tegrastats", "--interval", str(interval_ms)],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True
        )
        nopwr = 0
        for line in proc.stdout:
            if stop_event.is_set():
                break
            val = None
            for pat in [r'VIN_SYS_5V0\\s+(\\d+)mW', r'VDD_IN\\s+(\\d+)mW', r'POM_5V_IN\\s+(\\d+)']:
                m = re.search(pat, line)
                if m:
                    val = float(m.group(1)) / 1000.0
                    break
            if val is None:
                m = re.search(r'(\\d+\\.?\\d*)\\s*W\\b', line)
                if m:
                    val = float(m.group(1))
            if val is not None:
                power_samples.append(val)
            else:
                nopwr += 1
                if nopwr >= 3:
                    use_sysfs = True
                    break
        proc.terminate()
    except (FileNotFoundError, Exception):
        use_sysfs = True
    if use_sysfs and not stop_event.is_set():
        vp = cp = None
        for hw in _g.glob('/sys/class/hwmon/hwmon*'):
            try:
                with open(hw + '/name') as f:
                    if f.read().strip() != 'ina3221':
                        continue
                for i in range(1, 4):
                    try:
                        with open(hw + '/in%d_label' % i) as f:
                            if 'VDD_IN' in f.read():
                                vp = hw + '/in%d_input' % i
                                cp = hw + '/curr%d_input' % i
                                break
                    except (IOError, OSError):
                        continue
            except (IOError, OSError):
                continue
            if vp:
                break
        if vp:
            while not stop_event.is_set():
                try:
                    with open(vp) as f:
                        v = float(f.read().strip())
                    with open(cp) as f:
                        c = float(f.read().strip())
                    pw = v * c / 1e6
                    if pw > 0:
                        power_samples.append(pw)
                except (IOError, OSError, ValueError):
                    pass
                stop_event.wait(interval_ms / 1000.0)

def benchmark():
    device = torch.device(DEVICE)
    power_samples = []
    stop_event = threading.Event()
    tegrastats_thread = None
    if CAPTURE_ENERGY:
        tegrastats_thread = threading.Thread(target=_collect_tegrastats, args=(100, stop_event, power_samples))
        tegrastats_thread.daemon = True
        tegrastats_thread.start()
        time.sleep(0.5)

    load_start = time.perf_counter()
    try:
        # Load from yaml for random init (no pretrained weights)
        model = YOLO(MODEL_NAME + ".yaml")
    except Exception as e:
        return {{"status": "error", "error_type": "load", "error_message": str(e)}}
    load_time = (time.perf_counter() - load_start) * 1000

    dummy = np.random.randint(0, 255, (IMGSZ, IMGSZ, 3), dtype=np.uint8)

    for _ in range(WARMUP):
        _ = model.predict(dummy, verbose=False, device=DEVICE)

    latencies = []
    for _ in range(ITERATIONS):
        start = time.perf_counter()
        _ = model.predict(dummy, verbose=False, device=DEVICE)
        latencies.append((time.perf_counter() - start) * 1000)

    if CAPTURE_ENERGY:
        stop_event.set()
        if tegrastats_thread:
            tegrastats_thread.join(timeout=2)

    latencies_arr = np.array(latencies)
    process = psutil.Process()
    peak_memory_mb = process.memory_info().rss / (1024 * 1024)
    gpu_memory_mb = torch.cuda.memory_allocated() / (1024 * 1024) if device.type == "cuda" else None

    avg_power_w = float(np.mean(power_samples)) if power_samples else None
    energy_mj = (avg_power_w * np.mean(latencies_arr) / 1000) if avg_power_w else None

    return {{
        "status": "success",
        "latency_avg_ms": float(np.mean(latencies_arr)),
        "latency_p50_ms": float(np.percentile(latencies_arr, 50)),
        "latency_p95_ms": float(np.percentile(latencies_arr, 95)),
        "latency_p99_ms": float(np.percentile(latencies_arr, 99)),
        "latency_min_ms": float(np.min(latencies_arr)),
        "latency_max_ms": float(np.max(latencies_arr)),
        "latency_std_ms": float(np.std(latencies_arr)),
        "throughput_fps": 1000.0 / float(np.mean(latencies_arr)),
        "peak_memory_mb": peak_memory_mb,
        "gpu_memory_mb": gpu_memory_mb,
        "load_time_ms": load_time,
        "power_w": avg_power_w,
        "energy_mj_per_inference": energy_mj,
        "raw_latencies": latencies_arr.tolist(),
    }}

if __name__ == "__main__":
    print(json.dumps(benchmark()), flush=True)
'''

ULTRALYTICS_TENSORRT_RANDOM = '''#!/usr/bin/env python3
"""Benchmark ultralytics YOLO model with random weights (TensorRT runtime)."""
import contextlib
import io
import json
import time
import sys
import subprocess
import threading
import re

try:
    from ultralytics import YOLO
    import torch
    import numpy as np
    import psutil
except ImportError as e:
    print(json.dumps({{"status": "error", "error_type": "import", "error_message": str(e)}}))
    sys.exit(1)

MODEL_NAME = "{model_name}"
IMGSZ = {imgsz}
WARMUP = {warmup}
ITERATIONS = {iterations}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CAPTURE_ENERGY = True

def _collect_tegrastats(interval_ms, stop_event, power_samples):
    import glob as _g
    use_sysfs = False
    try:
        proc = subprocess.Popen(
            ["tegrastats", "--interval", str(interval_ms)],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True
        )
        nopwr = 0
        for line in proc.stdout:
            if stop_event.is_set():
                break
            val = None
            for pat in [r'VIN_SYS_5V0\\s+(\\d+)mW', r'VDD_IN\\s+(\\d+)mW', r'POM_5V_IN\\s+(\\d+)']:
                m = re.search(pat, line)
                if m:
                    val = float(m.group(1)) / 1000.0
                    break
            if val is None:
                m = re.search(r'(\\d+\\.?\\d*)\\s*W\\b', line)
                if m:
                    val = float(m.group(1))
            if val is not None:
                power_samples.append(val)
            else:
                nopwr += 1
                if nopwr >= 3:
                    use_sysfs = True
                    break
        proc.terminate()
    except (FileNotFoundError, Exception):
        use_sysfs = True
    if use_sysfs and not stop_event.is_set():
        vp = cp = None
        for hw in _g.glob('/sys/class/hwmon/hwmon*'):
            try:
                with open(hw + '/name') as f:
                    if f.read().strip() != 'ina3221':
                        continue
                for i in range(1, 4):
                    try:
                        with open(hw + '/in%d_label' % i) as f:
                            if 'VDD_IN' in f.read():
                                vp = hw + '/in%d_input' % i
                                cp = hw + '/curr%d_input' % i
                                break
                    except (IOError, OSError):
                        continue
            except (IOError, OSError):
                continue
            if vp:
                break
        if vp:
            while not stop_event.is_set():
                try:
                    with open(vp) as f:
                        v = float(f.read().strip())
                    with open(cp) as f:
                        c = float(f.read().strip())
                    pw = v * c / 1e6
                    if pw > 0:
                        power_samples.append(pw)
                except (IOError, OSError, ValueError):
                    pass
                stop_event.wait(interval_ms / 1000.0)

def benchmark():
    device = torch.device(DEVICE)
    power_samples = []
    stop_event = threading.Event()
    tegrastats_thread = None
    if CAPTURE_ENERGY:
        tegrastats_thread = threading.Thread(target=_collect_tegrastats, args=(100, stop_event, power_samples))
        tegrastats_thread.daemon = True
        tegrastats_thread.start()
        time.sleep(0.5)

    load_start = time.perf_counter()
    compile_start = load_start
    try:
        model = YOLO(MODEL_NAME + ".yaml")
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            engine_path = model.export(format="engine", imgsz=IMGSZ, half=True, workspace=4, simplify=True, verbose=False)
        compile_time = (time.perf_counter() - compile_start) * 1000
        model = YOLO(engine_path)
    except Exception as e:
        return {{"status": "error", "error_type": "export", "error_message": str(e)}}
    load_time = (time.perf_counter() - load_start) * 1000

    dummy = np.random.randint(0, 255, (IMGSZ, IMGSZ, 3), dtype=np.uint8)

    for _ in range(WARMUP):
        _ = model.predict(dummy, verbose=False, device=DEVICE)

    latencies = []
    for _ in range(ITERATIONS):
        start = time.perf_counter()
        _ = model.predict(dummy, verbose=False, device=DEVICE)
        latencies.append((time.perf_counter() - start) * 1000)

    if CAPTURE_ENERGY:
        stop_event.set()
        if tegrastats_thread:
            tegrastats_thread.join(timeout=2)

    latencies_arr = np.array(latencies)
    process = psutil.Process()
    peak_memory_mb = process.memory_info().rss / (1024 * 1024)
    gpu_memory_mb = torch.cuda.memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else None

    avg_power_w = float(np.mean(power_samples)) if power_samples else None
    energy_mj = (avg_power_w * np.mean(latencies_arr) / 1000) if avg_power_w else None

    return {{
        "status": "success",
        "latency_avg_ms": float(np.mean(latencies_arr)),
        "latency_p50_ms": float(np.percentile(latencies_arr, 50)),
        "latency_p95_ms": float(np.percentile(latencies_arr, 95)),
        "latency_p99_ms": float(np.percentile(latencies_arr, 99)),
        "latency_min_ms": float(np.min(latencies_arr)),
        "latency_max_ms": float(np.max(latencies_arr)),
        "latency_std_ms": float(np.std(latencies_arr)),
        "throughput_fps": 1000.0 / float(np.mean(latencies_arr)),
        "peak_memory_mb": peak_memory_mb,
        "gpu_memory_mb": gpu_memory_mb,
        "load_time_ms": load_time,
        "compile_time_ms": compile_time,
        "power_w": avg_power_w,
        "energy_mj_per_inference": energy_mj,
        "raw_latencies": latencies_arr.tolist(),
    }}

if __name__ == "__main__":
    print(json.dumps(benchmark()), flush=True)
'''

TIMM_PYTORCH_RANDOM = '''#!/usr/bin/env python3
"""Benchmark timm model with random weights (PyTorch runtime)."""
import json
import time
import sys
import subprocess
import threading
import re

try:
    import timm
    import torch
    import numpy as np
    import psutil
except ImportError as e:
    print(json.dumps({{"status": "error", "error_type": "import", "error_message": str(e)}}))
    sys.exit(1)

MODEL_NAME = "{model_name}"
IMGSZ = {imgsz}
WARMUP = {warmup}
ITERATIONS = {iterations}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CAPTURE_ENERGY = True

def _collect_tegrastats(interval_ms, stop_event, power_samples):
    import glob as _g
    use_sysfs = False
    try:
        proc = subprocess.Popen(
            ["tegrastats", "--interval", str(interval_ms)],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True
        )
        nopwr = 0
        for line in proc.stdout:
            if stop_event.is_set():
                break
            val = None
            for pat in [r'VIN_SYS_5V0\\s+(\\d+)mW', r'VDD_IN\\s+(\\d+)mW', r'POM_5V_IN\\s+(\\d+)']:
                m = re.search(pat, line)
                if m:
                    val = float(m.group(1)) / 1000.0
                    break
            if val is None:
                m = re.search(r'(\\d+\\.?\\d*)\\s*W\\b', line)
                if m:
                    val = float(m.group(1))
            if val is not None:
                power_samples.append(val)
            else:
                nopwr += 1
                if nopwr >= 3:
                    use_sysfs = True
                    break
        proc.terminate()
    except (FileNotFoundError, Exception):
        use_sysfs = True
    if use_sysfs and not stop_event.is_set():
        vp = cp = None
        for hw in _g.glob('/sys/class/hwmon/hwmon*'):
            try:
                with open(hw + '/name') as f:
                    if f.read().strip() != 'ina3221':
                        continue
                for i in range(1, 4):
                    try:
                        with open(hw + '/in%d_label' % i) as f:
                            if 'VDD_IN' in f.read():
                                vp = hw + '/in%d_input' % i
                                cp = hw + '/curr%d_input' % i
                                break
                    except (IOError, OSError):
                        continue
            except (IOError, OSError):
                continue
            if vp:
                break
        if vp:
            while not stop_event.is_set():
                try:
                    with open(vp) as f:
                        v = float(f.read().strip())
                    with open(cp) as f:
                        c = float(f.read().strip())
                    pw = v * c / 1e6
                    if pw > 0:
                        power_samples.append(pw)
                except (IOError, OSError, ValueError):
                    pass
                stop_event.wait(interval_ms / 1000.0)

def benchmark():
    device = torch.device(DEVICE)
    power_samples = []
    stop_event = threading.Event()
    tegrastats_thread = None
    if CAPTURE_ENERGY:
        tegrastats_thread = threading.Thread(target=_collect_tegrastats, args=(100, stop_event, power_samples))
        tegrastats_thread.daemon = True
        tegrastats_thread.start()
        time.sleep(0.5)

    load_start = time.perf_counter()
    try:
        model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=1000)
        model = model.to(device)
        model.eval()
    except Exception as e:
        return {{"status": "error", "error_type": "load", "error_message": str(e)}}
    load_time = (time.perf_counter() - load_start) * 1000

    dummy = torch.randn(1, 3, IMGSZ, IMGSZ).to(device)

    with torch.no_grad():
        for _ in range(WARMUP):
            _ = model(dummy)

    if device.type == "cuda":
        torch.cuda.synchronize()

    latencies = []
    with torch.no_grad():
        for _ in range(ITERATIONS):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - start) * 1000)

    if CAPTURE_ENERGY:
        stop_event.set()
        if tegrastats_thread:
            tegrastats_thread.join(timeout=2)

    latencies_arr = np.array(latencies)
    process = psutil.Process()
    peak_memory_mb = process.memory_info().rss / (1024 * 1024)
    gpu_memory_mb = torch.cuda.memory_allocated() / (1024 * 1024) if device.type == "cuda" else None

    avg_power_w = float(np.mean(power_samples)) if power_samples else None
    energy_mj = (avg_power_w * np.mean(latencies_arr) / 1000) if avg_power_w else None

    return {{
        "status": "success",
        "latency_avg_ms": float(np.mean(latencies_arr)),
        "latency_p50_ms": float(np.percentile(latencies_arr, 50)),
        "latency_p95_ms": float(np.percentile(latencies_arr, 95)),
        "latency_p99_ms": float(np.percentile(latencies_arr, 99)),
        "latency_min_ms": float(np.min(latencies_arr)),
        "latency_max_ms": float(np.max(latencies_arr)),
        "latency_std_ms": float(np.std(latencies_arr)),
        "throughput_fps": 1000.0 / float(np.mean(latencies_arr)),
        "peak_memory_mb": peak_memory_mb,
        "gpu_memory_mb": gpu_memory_mb,
        "load_time_ms": load_time,
        "power_w": avg_power_w,
        "energy_mj_per_inference": energy_mj,
        "raw_latencies": latencies_arr.tolist(),
    }}

if __name__ == "__main__":
    print(json.dumps(benchmark()), flush=True)
'''


# ---------------------------------------------------------------------------
# OfflineProfiler
# ---------------------------------------------------------------------------

class OfflineProfiler:
    """Run random-weight benchmarks for offline profiling on local or remote devices."""

    def __init__(self, edge_runner_path: Optional[str] = None):
        """Initialize OfflineProfiler.

        Args:
            edge_runner_path: Path to edge-runner/server. Auto-detected if None.
        """
        from edgecraft.models import ensure_registries_initialized
        ensure_registries_initialized()
        self.edge_runner_path = edge_runner_path or self._find_edge_runner()
        self.results_dir = Path(settings.EDGECRAFT_ROOT) / "outputs" / "profiling"

    def _find_edge_runner(self) -> Optional[str]:
        """Find edge-runner directory."""
        edge_runner = Path(settings.BASE_DIR) / "tools" / "deploy" / "runner" / "server"
        if edge_runner.exists():
            return str(edge_runner)
        return None

    def _log_profile_stage(
        self,
        request: ProfileRequest,
        stage: str,
        detail: str = "",
    ) -> None:
        """Compact per-model progress log for remote profiling."""
        msg = f"[profile:{request.runtime_id.value}] {request.model_id} :: {stage}"
        if detail:
            msg = f"{msg} | {detail}"
        logger.info(msg)

    def _get_device_config(self, device_id: str) -> Dict[str, Any]:
        """Get configuration for a specific device via DeviceRegistry."""
        spec = DeviceRegistry.get(device_id)
        if spec is None:
            logger.warning(f"Unknown device '{device_id}'. Assuming CPU-only device.")
            return {"has_gpu": False}
        return {"has_gpu": spec.has_gpu}

    def _generate_random_weight_benchmark_script(
        self,
        family_id: str,
        model_name: str,
        runtime_id: RuntimeId,
        imgsz: int = 640,
        warmup: int = 10,
        iterations: int = 100,
    ) -> str:
        """Generate benchmark.py content for random-weight profiling.

        Args:
            family_id: Family identifier (ultralytics, timm).
            model_name: Short model name (yolo11n, shufflenet_v2_x1_0, etc.).
            runtime_id: PYTORCH or TENSORRT.
            imgsz: Input image size.
            warmup: Warmup iterations.
            iterations: Benchmark iterations.

        Returns:
            Python script content.
        """
        spec = FamilyRegistry.get_model_by_name(model_name, family_id)
        if spec is None:
            raise ValueError(f"Unknown model: {family_id}:{model_name}")
        plugin = FamilyRegistry.get_plugin(family_id)
        if plugin is None:
            raise ValueError(f"No family plugin for '{family_id}'")
        script = plugin.render_profile_benchmark_script(
            model_spec=spec,
            runtime_id=runtime_id,
            imgsz=imgsz,
            warmup=warmup,
            iterations=iterations,
        )
        if not script:
            raise ValueError(f"No benchmark provider for {family_id}/{runtime_id.value}")
        return script

    def _parse_benchmark_json(self, output: str) -> Optional[Dict[str, Any]]:
        """Parse JSON result from benchmark stdout.

        Benchmarks print a single JSON line. This finds the last valid JSON line.

        Args:
            output: Combined stdout/stderr from benchmark.

        Returns:
            Parsed dict or None if parse failed.
        """
        for line in reversed(output.strip().splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
        return None

    def _read_collected_stdout(self, job_id: str) -> str:
        """Best-effort read of collected stdout log for a finished job."""
        if not self.edge_runner_path:
            return ""
        stdout_log = os.path.join(
            self.results_dir, job_id, "stdout.log"
        )
        if not os.path.exists(stdout_log):
            return ""
        try:
            with open(stdout_log, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception:
            return ""

    def _read_collected_stderr(self, job_id: str) -> str:
        """Best-effort read of collected stderr log for a finished job."""
        if not self.edge_runner_path:
            return ""
        stderr_log = os.path.join(
            self.results_dir, job_id, "stderr.log"
        )
        if not os.path.exists(stderr_log):
            return ""
        try:
            with open(stderr_log, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception:
            return ""

    def _read_collected_result_json(self, job_id: str) -> Dict[str, Any]:
        """Best-effort read of collected result.json metadata for a finished job."""
        if not self.edge_runner_path:
            return {}
        result_json = os.path.join(
            self.results_dir, job_id, "result.json"
        )
        if not os.path.exists(result_json):
            return {}
        try:
            with open(result_json, "r", encoding="utf-8", errors="ignore") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _result_from_benchmark_data(
        self,
        data: Dict[str, Any],
        request: ProfileRequest,
    ) -> ProfileResult:
        """Convert benchmark JSON output to ProfileResult.

        Args:
            data: Parsed benchmark output dict.
            request: Original ProfileRequest.

        Returns:
            ProfileResult with metrics or error status.
        """
        profile_key = request.get_profile_key()
        status = data.get("status", "error")
        error_type = data.get("error_type")
        error_message = data.get("error_message")

        result_kw: Dict[str, Any] = {
            "profile_key": profile_key,
            "model_id": request.model_id,
            "device_id": request.device_id,
            "runtime_id": request.runtime_id,
            "quant_mode": request.quant_mode,
            "export_format": request.export_format,
            "input_signature": dict(request.input_signature),
            "status": status,
            "error_type": error_type,
            "error_message": error_message,
            "environment_versions": dict(request.environment_versions),
            "environment_fingerprint": request.environment_fingerprint,
        }

        if status == "success":
            result_kw["latency_avg_ms"] = float(data.get("latency_avg_ms", 0))
            result_kw["latency_p50_ms"] = float(data.get("latency_p50_ms", 0))
            result_kw["latency_p95_ms"] = float(data.get("latency_p95_ms", 0))
            result_kw["latency_p99_ms"] = float(data.get("latency_p99_ms", 0))
            result_kw["latency_min_ms"] = float(data.get("latency_min_ms", 0))
            result_kw["latency_max_ms"] = float(data.get("latency_max_ms", 0))
            result_kw["latency_std_ms"] = float(data.get("latency_std_ms", 0))
            result_kw["throughput_fps"] = float(data.get("throughput_fps", 0))
            result_kw["peak_memory_mb"] = float(data.get("peak_memory_mb", 0))
            result_kw["gpu_memory_mb"] = data.get("gpu_memory_mb")
            result_kw["load_time_ms"] = float(data.get("load_time_ms", 0))
            result_kw["compile_time_ms"] = float(data.get("compile_time_ms", 0))
            result_kw["power_w"] = data.get("power_w")
            result_kw["energy_mj_per_inference"] = data.get("energy_mj_per_inference")
            result_kw["raw_latencies"] = data.get("raw_latencies", [])

        return ProfileResult(**result_kw)

    def profile_local(self, request: ProfileRequest, timeout: int = 600) -> ProfileResult:
        """Run benchmark in subprocess on local machine.

        Args:
            request: ProfileRequest specifying model, device, runtime, etc.
            timeout: Subprocess timeout in seconds.

        Returns:
            ProfileResult.
        """
        family_id = request.family_id
        model_name = request.model_id.split(":")[-1] if ":" in request.model_id else request.model_id
        imgsz = request.input_signature.get("imgsz", 640)

        try:
            script = self._generate_random_weight_benchmark_script(
                family_id=family_id,
                model_name=model_name,
                runtime_id=request.runtime_id,
                imgsz=imgsz,
                warmup=request.warmup_iterations,
                iterations=request.benchmark_iterations,
            )
        except ValueError as e:
            logger.warning(f"Cannot profile {request.model_id} {request.runtime_id}: {e}")
            return self._result_from_benchmark_data(
                {"status": "error", "error_type": "unsupported", "error_message": str(e)},
                request,
            )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write(script)
            script_path = f.name

        try:
            proc = subprocess.run(
                ["python3", script_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=os.path.dirname(script_path),
            )
            output = proc.stdout or ""
            if proc.stderr:
                output += "\n" + proc.stderr

            data = self._parse_benchmark_json(output)
            if data is None:
                return self._result_from_benchmark_data(
                    {
                        "status": "error",
                        "error_type": "parse",
                        "error_message": "No valid JSON in output",
                    },
                    request,
                )
            return self._result_from_benchmark_data(data, request)
        finally:
            move_to_trash(script_path)

    def profile_remote(
        self,
        request: ProfileRequest,
        device_ip: str,
        ssh_key: str,
        docker_image: str,
        timeout: int = 600,
    ) -> ProfileResult:
        """Run benchmark on remote edge device.

        Args:
            request: ProfileRequest.
            device_ip: SSH target (user@host).
            ssh_key: Path to SSH private key.
            docker_image: User-provided pre-built image.
            timeout: Job timeout in seconds.

        Returns:
            ProfileResult.
        """
        family_id = request.family_id
        model_name = request.model_id.split(":")[-1] if ":" in request.model_id else request.model_id
        imgsz = request.input_signature.get("imgsz", 640)
        device_id = request.device_id

        if not self.edge_runner_path:
            return self._result_from_benchmark_data(
                {"status": "error", "error_type": "config", "error_message": "edge-runner path not found"},
                request,
            )

        if not docker_image:
            return self._result_from_benchmark_data(
                {"status": "error", "error_type": "config", "error_message": "docker_image is required"},
                request,
            )
        device_config = self._get_device_config(device_id)
        image = docker_image
        has_gpu = device_config.get("has_gpu", False)
        skip_pip = True

        job_id = f"edgecraft_profile_{family_id}_{model_name}_{request.runtime_id.value}_{int(time.time())}"
        job_dir = tempfile.mkdtemp(prefix=f"edgecraft_job_{job_id}_")

        try:
            self._log_profile_stage(
                request, "prepare", f"device={device_id} timeout={timeout}s"
            )
            payload_dir = os.path.join(job_dir, "payload")
            os.makedirs(payload_dir, exist_ok=True)
            benchmark_path = os.path.join(payload_dir, "benchmark.py")

            # TRT path: export ONNX on server first, then benchmark TRT on edge
            # with trtexec only (no family imports on edge).
            if request.runtime_id == RuntimeId.TENSORRT:
                onnx_path = os.path.join(payload_dir, "model.onnx")
                try:
                    t0 = time.perf_counter()
                    self._log_profile_stage(request, "export", "onnx (server-side)")
                    shape_flags = self._export_onnx_for_trt_request(
                        request=request,
                        out_onnx_path=onnx_path,
                        imgsz=imgsz,
                    )
                    self._log_profile_stage(
                        request,
                        "export_done",
                        f"onnx ready in {(time.perf_counter() - t0):.1f}s",
                    )
                except Exception as e:
                    return self._result_from_benchmark_data(
                        {"status": "error", "error_type": "export", "error_message": str(e)},
                        request,
                    )
                script = self._generate_trtexec_benchmark_script(
                    shape_flags=shape_flags,
                    warmup=request.warmup_iterations,
                    iterations=request.benchmark_iterations,
                )
            else:
                try:
                    self._log_profile_stage(request, "render", "benchmark script")
                    script = self._generate_random_weight_benchmark_script(
                        family_id=family_id,
                        model_name=model_name,
                        runtime_id=request.runtime_id,
                        imgsz=imgsz,
                        warmup=request.warmup_iterations,
                        iterations=request.benchmark_iterations,
                    )
                except ValueError as e:
                    logger.warning(f"Cannot profile {request.model_id} {request.runtime_id}: {e}")
                    return self._result_from_benchmark_data(
                        {"status": "error", "error_type": "unsupported", "error_message": str(e)},
                        request,
                    )

            with open(benchmark_path, "w") as f:
                f.write(script)

            if request.runtime_id == RuntimeId.TENSORRT:
                run_script = self._generate_run_script_trtexec_host()
            else:
                extra_pip_packages: List[str] = []
                if family_id == "speechbrain":
                    # Some SpeechBrain checkpoints require transformers at runtime.
                    extra_pip_packages.append("transformers")
                run_script = self._generate_run_script_docker(
                    image,
                    has_gpu,
                    skip_pip_install=skip_pip,
                    extra_pip_packages=extra_pip_packages,
                )

            run_sh_path = os.path.join(job_dir, "run.sh")
            with open(run_sh_path, "w") as f:
                f.write("#!/usr/bin/env bash\n")
                f.write("set -euo pipefail\n\n")
                f.write(run_script)
            os.chmod(run_sh_path, 0o755)

            meta: Dict[str, str] = {
                "JOB_ID": job_id,
                "TIMEOUT_SECS": str(timeout),
                "WORKDIR": ".",  # run.sh does "cd payload"
                "DOCKER_IMAGE": image,
                "HAS_GPU": "1" if has_gpu else "0",
            }

            meta_path = os.path.join(job_dir, "meta.env")
            with open(meta_path, "w") as f:
                for key, value in meta.items():
                    f.write(f"{key}={value}\n")

            self._log_profile_stage(request, "submit", f"job_id={job_id}")
            run_t0 = time.perf_counter()
            result = self._run_edge_script(
                "run_job_and_collect.sh",
                [
                    "--edge", device_ip,
                    "--key", ssh_key,
                    "--job-dir", job_dir,
                    "--collect-out", str(self.results_dir),
                    "--stream",
                    "--collect-retries", "4",
                    "--collect-retry-delay", "3",
                    "--collect-extract",
                    "--collect-keep-job",
                    "--no-preflight-cleanup",
                    "--stream-grace-secs", "5",
                    "--wait-secs", str(timeout),
                ],
                timeout=timeout + 60,
            )
            self._log_profile_stage(
                request,
                "remote_done",
                f"elapsed={(time.perf_counter() - run_t0):.1f}s status={result.get('status')}",
            )

            self._log_profile_stage(request, "parse", "reading collected stdout")
            output = self._read_collected_stdout(job_id)
            data = self._parse_benchmark_json(output) if output else None
            if data is not None:
                # Prefer parsed benchmark JSON even when collect script reported error
                # (e.g. partial tar extraction after job already completed successfully).
                if result["status"] != "success":
                    logger.warning(
                        f"Recovered benchmark JSON for {job_id} despite collect error: "
                        f"{result.get('error', result.get('stderr', 'unknown'))}"
                    )
                self._log_profile_stage(request, "done", "metrics parsed")
                return self._result_from_benchmark_data(data, request)

            # Priority semantic: consume edge-runner result metadata first.
            result_meta = self._read_collected_result_json(job_id)
            exit_code = result_meta.get("exit_code")
            try:
                exit_code = int(exit_code) if exit_code is not None else None
            except Exception:
                exit_code = None
            if exit_code is not None and exit_code != 0:
                stderr_output = self._read_collected_stderr(job_id)
                stderr_tail = "\n".join(stderr_output.strip().splitlines()[-30:]) if stderr_output.strip() else ""
                meta_error = str(result_meta.get("error", "")).strip()
                if exit_code in {124, 137, 143}:
                    err_type = "timeout"
                else:
                    err_type = "runtime"
                msg_parts = [f"Remote job exited with code {exit_code}."]
                if meta_error:
                    msg_parts.append(f"edge_error={meta_error}")
                if stderr_tail:
                    msg_parts.append(f"stderr tail:\n{stderr_tail}")
                self._log_profile_stage(request, "done", "exit_code indicates failure")
                return self._result_from_benchmark_data(
                    {
                        "status": "error",
                        "error_type": err_type,
                        "error_message": "\n".join(msg_parts),
                    },
                    request,
                )

            if result["status"] == "success":
                stderr_output = self._read_collected_stderr(job_id)
                if stderr_output.strip():
                    stderr_tail = "\n".join(stderr_output.strip().splitlines()[-30:])
                    self._log_profile_stage(request, "done", "no JSON, runtime stderr captured")
                    return self._result_from_benchmark_data(
                        {
                            "status": "error",
                            "error_type": "runtime",
                            "error_message": (
                                "Benchmark exited without JSON output; stderr tail:\n"
                                f"{stderr_tail}"
                            ),
                        },
                        request,
                    )
                self._log_profile_stage(request, "done", "no JSON in output")
                return self._result_from_benchmark_data(
                    {
                        "status": "error",
                        "error_type": "parse",
                        "error_message": "No valid JSON in remote output",
                    },
                    request,
                )

            self._log_profile_stage(request, "done", "remote error")
            return self._result_from_benchmark_data(
                {
                    "status": "error",
                    "error_type": "remote",
                    "error_message": result.get("error", result.get("stderr", "Unknown error")),
                },
                request,
            )
        finally:
            move_to_trash(job_dir)

    def _export_onnx_for_trt_request(
        self,
        request: ProfileRequest,
        out_onnx_path: str,
        imgsz: int,
    ) -> List[str]:
        """Export model to ONNX on server side and return trtexec shape flags."""
        spec = FamilyRegistry.get_model(request.model_id)
        if spec is None:
            raise ValueError(f"Unknown model spec: {request.model_id}")
        family_id = spec.family_id
        model_name = spec.name

        if family_id == "ultralytics":
            from ultralytics import YOLO
            export_root = Path(out_onnx_path).resolve().parent
            export_root.mkdir(parents=True, exist_ok=True)
            prev_cwd = Path.cwd()
            try:
                os.chdir(str(export_root))
                model = YOLO(f"{model_name}.yaml")
                exported = Path(model.export(format="onnx", imgsz=imgsz, simplify=True, verbose=False))
                if not exported.is_absolute():
                    exported = export_root / exported
            finally:
                os.chdir(str(prev_cwd))
            shutil.copy2(str(exported), out_onnx_path)
            return []

        if family_id == "timm":
            import timm
            import torch
            model_ref = spec.pretrained_weights or model_name
            candidates: List[str] = []
            for cand in (model_ref, model_name, model_name.replace("maxxvit_", "maxvit_")):
                if cand and cand not in candidates:
                    candidates.append(cand)
                if cand and "." in cand:
                    base = cand.split(".", 1)[0]
                    if base not in candidates:
                        candidates.append(base)

            model = None
            last_error: Optional[Exception] = None
            for cand in candidates:
                try:
                    try:
                        model = timm.create_model(cand, pretrained=False, img_size=imgsz).eval().cpu()
                    except TypeError:
                        model = timm.create_model(cand, pretrained=False).eval().cpu()
                    break
                except Exception as exc:
                    last_error = exc
                    continue
            if model is None:
                raise ValueError(f"Unknown timm model ({model_name}); tried {candidates}; last_error={last_error}")

            export_h = imgsz
            export_w = imgsz
            cfg = getattr(model, "default_cfg", {}) or {}
            input_size = cfg.get("input_size")
            if isinstance(input_size, (list, tuple)) and len(input_size) == 3:
                try:
                    export_h = int(input_size[1])
                    export_w = int(input_size[2])
                except Exception:
                    export_h = imgsz
                    export_w = imgsz
            dummy = torch.randn(1, 3, export_h, export_w)
            self._torch_onnx_export(
                model,
                dummy,
                out_onnx_path,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
                # Some timm backbones (e.g. MobileViT/ViT) require opset >= 14
                # due to scaled_dot_product_attention export support.
                opset_version=17,
            )
            return [f"--shapes=input:1x3x{export_h}x{export_w}"]

        if family_id == "torchvision":
            import torch
            import torch.nn as nn
            import torchvision
            if TaskType.OBJECT_DETECTION in spec.supported_tasks:
                factory = {
                    "ssdlite320_mobilenet_v3_large": torchvision.models.detection.ssdlite320_mobilenet_v3_large,
                    "fasterrcnn_mobilenet_v3_large_fpn": torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn,
                    "fasterrcnn_resnet50_fpn_v2": torchvision.models.detection.fasterrcnn_resnet50_fpn_v2,
                    "retinanet_resnet50_fpn_v2": torchvision.models.detection.retinanet_resnet50_fpn_v2,
                    "fcos_resnet50_fpn": torchvision.models.detection.fcos_resnet50_fpn,
                    "ssd300_vgg16": torchvision.models.detection.ssd300_vgg16,
                    "maskrcnn_resnet50_fpn": torchvision.models.detection.maskrcnn_resnet50_fpn,
                }
                if model_name not in factory:
                    raise ValueError(f"Unsupported torchvision detection model for TRT export: {model_name}")
                base = factory[model_name](weights=None).eval().cpu()

                class DetectionWrapper(nn.Module):
                    def __init__(self, m, max_det=100):
                        super().__init__()
                        self.m = m
                        self.max_det = max_det

                    def forward(self, x):
                        pred = self.m(x)[0]
                        boxes = pred["boxes"]
                        scores = pred["scores"]
                        labels = pred["labels"].to(torch.float32)
                        n = min(boxes.shape[0], self.max_det)
                        out_boxes = torch.zeros(self.max_det, 4, device=boxes.device, dtype=boxes.dtype)
                        out_scores = torch.zeros(self.max_det, device=scores.device, dtype=scores.dtype)
                        out_labels = torch.zeros(self.max_det, device=labels.device, dtype=labels.dtype)
                        if n > 0:
                            out_boxes[:n] = boxes[:n]
                            out_scores[:n] = scores[:n]
                            out_labels[:n] = labels[:n]
                        return out_boxes, out_scores, out_labels

                model = DetectionWrapper(base).eval()
                dummy = torch.randn(1, 3, imgsz, imgsz)
                self._torch_onnx_export(
                    model,
                    dummy,
                    out_onnx_path,
                    input_names=["input"],
                    output_names=["boxes", "scores", "labels"],
                    dynamic_axes={"input": {0: "batch"}},
                    opset_version=13,
                )
                return [f"--shapes=input:1x3x{imgsz}x{imgsz}"]
            model = torchvision.models.get_model(model_name, weights=None).eval().cpu()
            dummy = torch.randn(1, 3, imgsz, imgsz)
            self._torch_onnx_export(
                model,
                dummy,
                out_onnx_path,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
                opset_version=17,
            )
            return [f"--shapes=input:1x3x{imgsz}x{imgsz}"]

        if family_id == "transformers":
            import torch
            from transformers import AutoModelForSequenceClassification  # pyright: ignore[reportMissingImports]
            model_ref = spec.pretrained_weights or model_name
            max_length = int(spec.default_hyperparams.get("max_length", 128))
            model = AutoModelForSequenceClassification.from_pretrained(model_ref).eval().cpu()
            input_ids = torch.ones((1, max_length), dtype=torch.long)
            attention_mask = torch.ones((1, max_length), dtype=torch.long)
            self._torch_onnx_export(
                model,
                (input_ids, attention_mask),
                out_onnx_path,
                input_names=["input_ids", "attention_mask"],
                output_names=["logits"],
                dynamic_axes={
                    "input_ids": {0: "batch", 1: "seq"},
                    "attention_mask": {0: "batch", 1: "seq"},
                    "logits": {0: "batch"},
                },
                opset_version=13,
            )
            shape = f"1x{max_length}"
            return [
                f"--minShapes=input_ids:{shape},attention_mask:{shape}",
                f"--optShapes=input_ids:{shape},attention_mask:{shape}",
                f"--maxShapes=input_ids:{shape},attention_mask:{shape}",
            ]

        if family_id == "whisper":
            import torch
            from transformers import WhisperForConditionalGeneration  # pyright: ignore[reportMissingImports]
            model_ref = spec.pretrained_weights or model_name.replace("whisper-", "")
            if not str(model_ref).startswith("openai/"):
                model_ref = f"openai/whisper-{model_ref}"
            model = WhisperForConditionalGeneration.from_pretrained(model_ref).eval().cpu()
            input_features = torch.randn(1, 80, 3000, dtype=torch.float32)
            decoder_input_ids = torch.ones((1, 2), dtype=torch.long)

            class WhisperWrapper(torch.nn.Module):
                def __init__(self, m):
                    super().__init__()
                    self.m = m

                def forward(self, input_features, decoder_input_ids):
                    out = self.m(input_features=input_features, decoder_input_ids=decoder_input_ids)
                    return out.logits

            wrapped = WhisperWrapper(model).eval()
            self._torch_onnx_export(
                wrapped,
                (input_features, decoder_input_ids),
                out_onnx_path,
                input_names=["input_features", "decoder_input_ids"],
                output_names=["logits"],
                dynamic_axes={
                    "input_features": {0: "batch"},
                    "decoder_input_ids": {0: "batch", 1: "decoder_seq"},
                    "logits": {0: "batch"},
                },
                opset_version=17,
            )
            return [
                "--minShapes=input_features:1x80x3000,decoder_input_ids:1x2",
                "--optShapes=input_features:1x80x3000,decoder_input_ids:1x2",
                "--maxShapes=input_features:1x80x3000,decoder_input_ids:1x2",
            ]

        if family_id == "speechbrain":
            raise ValueError(
                "SpeechBrain TensorRT profiling is currently unsupported in OfflineProfiler: "
                "the default SpeechBrain feature path relies on torch.stft complex tensors "
                "that cannot be exported in the current ONNX->TRT pipeline. "
                "Use ONNXRUNTIME for speechbrain profiles."
            )

        raise ValueError(f"Unsupported family for server-side TRT ONNX export: {family_id}")

    def _patch_torchaudio_for_speechbrain(self) -> None:
        """Compatibility shim for SpeechBrain on newer torchaudio versions.

        SpeechBrain currently calls `torchaudio.list_audio_backends()` during import.
        Some recent torchaudio builds removed this API, so we provide a minimal
        no-op shim to keep import/export flow working.
        """
        try:
            import torchaudio  # pyright: ignore[reportMissingImports]
        except Exception:
            return
        if not hasattr(torchaudio, "list_audio_backends"):
            def _list_audio_backends():
                return []
            setattr(torchaudio, "list_audio_backends", _list_audio_backends)
        if not hasattr(torchaudio, "set_audio_backend"):
            def _set_audio_backend(_backend: str):
                return None
            setattr(torchaudio, "set_audio_backend", _set_audio_backend)

    def _patch_huggingface_hub_for_speechbrain(self) -> None:
        """Compatibility shim: map deprecated use_auth_token -> token."""
        try:
            import huggingface_hub  # pyright: ignore[reportMissingImports]
        except Exception:
            return
        try:
            from huggingface_hub import file_download  # pyright: ignore[reportMissingImports]
        except Exception:
            file_download = None

        def _wrap(func):
            try:
                sig = inspect.signature(func)
            except Exception:
                return func
            if "use_auth_token" in sig.parameters:
                return func

            def _compat(*args, **kwargs):
                if "use_auth_token" in kwargs and "token" not in kwargs:
                    kwargs["token"] = kwargs.pop("use_auth_token")
                else:
                    kwargs.pop("use_auth_token", None)
                return func(*args, **kwargs)

            return _compat

        if hasattr(huggingface_hub, "hf_hub_download"):
            huggingface_hub.hf_hub_download = _wrap(huggingface_hub.hf_hub_download)
        if file_download is not None and hasattr(file_download, "hf_hub_download"):
            file_download.hf_hub_download = _wrap(file_download.hf_hub_download)

    def _torch_onnx_export(self, model: Any, args: Any, f: str, **kwargs: Any) -> None:
        """Export ONNX using legacy exporter first to avoid onnxscript hard dependency."""
        import torch
        with warnings.catch_warnings():
            try:
                warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
            except Exception:
                pass
            try:
                torch.onnx.export(model, args, f, dynamo=False, **kwargs)
            except TypeError:
                torch.onnx.export(model, args, f, **kwargs)

    def _generate_trtexec_benchmark_script(
        self,
        shape_flags: List[str],
        warmup: int,
        iterations: int,
    ) -> str:
        """Generate a lightweight trtexec benchmark script for edge side."""
        shape_flags_json = json.dumps(shape_flags)
        return f'''#!/usr/bin/env python3
import json
import os
import re
import shutil
import subprocess
import threading
import time

try:
    import psutil
except ImportError:
    psutil = None

CAPTURE_ENERGY = True

def _collect_tegrastats(interval_ms, stop_event, power_samples):
    import glob as _g
    use_sysfs = False
    try:
        proc = subprocess.Popen(
            ["tegrastats", "--interval", str(interval_ms)],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        )
        nopwr = 0
        for line in proc.stdout:
            if stop_event.is_set():
                break
            val = None
            for pat in [r"VIN_SYS_5V0\\s+(\\d+)mW", r"VDD_IN\\s+(\\d+)mW", r"POM_5V_IN\\s+(\\d+)"]:
                m = re.search(pat, line)
                if m:
                    val = float(m.group(1)) / 1000.0
                    break
            if val is None:
                m = re.search(r"(\\d+\\.?\\d*)\\s*W\\b", line)
                if m:
                    val = float(m.group(1))
            if val is not None:
                power_samples.append(val)
            else:
                nopwr += 1
                if nopwr >= 3:
                    use_sysfs = True
                    break
        proc.terminate()
    except (FileNotFoundError, Exception):
        use_sysfs = True
    if use_sysfs and not stop_event.is_set():
        vp = cp = None
        for hw in _g.glob("/sys/class/hwmon/hwmon*"):
            try:
                with open(hw + "/name") as f:
                    if f.read().strip() != "ina3221":
                        continue
                for i in range(1, 4):
                    try:
                        with open(hw + "/in%d_label" % i) as f:
                            if "VDD_IN" in f.read():
                                vp = hw + "/in%d_input" % i
                                cp = hw + "/curr%d_input" % i
                                break
                    except (IOError, OSError):
                        continue
            except (IOError, OSError):
                continue
            if vp:
                break
        if vp:
            while not stop_event.is_set():
                try:
                    with open(vp) as f:
                        v = float(f.read().strip())
                    with open(cp) as f:
                        c = float(f.read().strip())
                    pw = v * c / 1e6
                    if pw > 0:
                        power_samples.append(pw)
                except (IOError, OSError, ValueError):
                    pass
                stop_event.wait(interval_ms / 1000.0)

trtexec = shutil.which("trtexec")
if not trtexec:
    for candidate in (
        "/usr/src/tensorrt/bin/trtexec",
        "/usr/local/bin/trtexec",
        "/usr/bin/trtexec",
    ):
        if os.path.exists(candidate):
            trtexec = candidate
            break
if not trtexec:
    print(json.dumps({{"status":"error","error_type":"runtime","error_message":"trtexec not found in PATH"}}))
    raise SystemExit(1)
if not os.path.exists("model.onnx"):
    print(json.dumps({{"status":"error","error_type":"input","error_message":"model.onnx not found"}}))
    raise SystemExit(1)

power_samples = []
stop_event = threading.Event()
tegra_thread = None
if CAPTURE_ENERGY:
    tegra_thread = threading.Thread(target=_collect_tegrastats, args=(100, stop_event, power_samples))
    tegra_thread.daemon = True
    tegra_thread.start()
    time.sleep(0.5)

cmd = [
    trtexec,
    "--onnx=model.onnx",
    "--saveEngine=model.engine",
    "--fp16",
    "--useSpinWait",
    "--warmUp={warmup}",
    "--duration=0",
    "--iterations={iterations}",
    "--avgRuns=1",
]
cmd.extend({shape_flags_json})
result = subprocess.run(cmd, capture_output=True, text=True)

if CAPTURE_ENERGY:
    stop_event.set()
    if tegra_thread:
        tegra_thread.join(timeout=2)

combined = (result.stdout or "") + "\\n" + (result.stderr or "")
if result.returncode != 0:
    print(json.dumps({{"status":"error","error_type":"runtime","error_message":combined[-4000:]}}))
    raise SystemExit(1)

mean_match = re.search(r"mean\\s*=\\s*([0-9.]+)\\s*ms", combined)
p50_match = re.search(r"median\\s*=\\s*([0-9.]+)\\s*ms", combined)
p95_match = re.search(r"percentile\\(95%\\)\\s*=\\s*([0-9.]+)\\s*ms", combined)
p99_match = re.search(r"percentile\\(99%\\)\\s*=\\s*([0-9.]+)\\s*ms", combined)
min_match = re.search(r"min\\s*=\\s*([0-9.]+)\\s*ms", combined)
max_match = re.search(r"max\\s*=\\s*([0-9.]+)\\s*ms", combined)
if not mean_match:
    print(json.dumps({{"status":"error","error_type":"parse","error_message":"Could not parse trtexec latency output"}}))
    raise SystemExit(1)

lat_mean = float(mean_match.group(1))
lat_p50 = float(p50_match.group(1)) if p50_match else lat_mean
lat_p95 = float(p95_match.group(1)) if p95_match else lat_mean
lat_p99 = float(p99_match.group(1)) if p99_match else lat_p95
lat_min = float(min_match.group(1)) if min_match else min(lat_mean, lat_p50)
lat_max = float(max_match.group(1)) if max_match else max(lat_mean, lat_p50, lat_p95, lat_p99)
mem_mb = float(psutil.Process().memory_info().rss / (1024 * 1024)) if psutil else 0.0
avg_power_w = float(sum(power_samples) / len(power_samples)) if power_samples else None
energy_mj = (avg_power_w * lat_mean / 1000.0) if avg_power_w else None
print(json.dumps({{
    "status":"success",
    "latency_avg_ms": lat_mean,
    "latency_p50_ms": lat_p50,
    "latency_p95_ms": lat_p95,
    "latency_p99_ms": lat_p99,
    "latency_min_ms": lat_min,
    "latency_max_ms": lat_max,
    "latency_std_ms": 0.0,
    "throughput_fps": float(1000.0 / lat_mean) if lat_mean > 0 else 0.0,
    "peak_memory_mb": mem_mb,
    "power_w": avg_power_w,
    "energy_mj_per_inference": energy_mj,
    "raw_latencies": [],
}}))
'''

    def _generate_run_script_docker(
        self,
        docker_image: str,
        has_gpu: bool,
        skip_pip_install: bool = False,
        extra_pip_packages: Optional[List[str]] = None,
    ) -> str:
        """Generate run.sh for Docker mode.

        When skip_pip_install=True (pre-built image), only run benchmark.py.
        Otherwise pip install ultralytics timm psutil first.
        """
        gpu_opts = "--runtime=nvidia --gpus all" if has_gpu else ""
        extra_pkgs = extra_pip_packages or []
        extra_pkg_list = " ".join(extra_pkgs).strip()
        extra_install_prefix = (
            f"python3 -m pip install -q {extra_pkg_list} && " if extra_pkg_list else ""
        )
        base_pip_line = "python3 -m pip install -q ultralytics timm psutil"
        if extra_pkg_list:
            base_pip_line = f"{base_pip_line} {extra_pkg_list}"
        if skip_pip_install:
            prebuilt_args = offline_profiler_prebuilt_docker_args_bash(has_gpu)
            return f'''echo "[edgecraft] Running benchmark (Docker pre-built image)"
echo "[edgecraft] Docker image: {docker_image}"
echo "[edgecraft] Docker user: $(id -u):$(id -g)"

cd payload
docker run {prebuilt_args} {docker_image} \\
  bash -lc "mkdir -p /workspace/.cache/huggingface /workspace/.cache/torch && {extra_install_prefix}python3 benchmark.py"
'''
        return f'''echo "[edgecraft] Running benchmark (Docker mode)"
echo "[edgecraft] Docker image: {docker_image}"
echo "[edgecraft] Has GPU: {'1' if has_gpu else '0'}"

cd payload
PIP_CACHE="${{HOME:-/root}}/.cache/pip"
mkdir -p "$PIP_CACHE"

DOCKER_OPTS="--rm --privileged -v /usr/bin/tegrastats:/usr/bin/tegrastats:ro -v $(pwd):/workspace -w /workspace -v $PIP_CACHE:/root/.cache/pip {gpu_opts}"

echo "[edgecraft] Installing dependencies and running benchmark..."
docker run $DOCKER_OPTS {docker_image} bash -c "
  {base_pip_line}
  python3 benchmark.py
"
'''

    def _generate_run_script_trtexec_host(self) -> str:
        """Generate run.sh for host TRT execution (no docker dependency)."""
        return '''echo "[edgecraft] Running benchmark (host TRT mode)"
cd payload
python3 benchmark.py
'''

    def _run_edge_script(
        self,
        script_name: str,
        args: List[str],
        timeout: int = 600,
    ) -> Dict[str, Any]:
        """Run an edge-runner script."""
        script_path = os.path.join(self.edge_runner_path, script_name)
        if not os.path.exists(script_path):
            return {"status": "error", "error": f"Script not found: {script_path}"}

        try:
            result = subprocess.run(
                ["bash", script_path] + args,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=settings.EDGECRAFT_ROOT,
            )
            return {
                "status": "success" if result.returncode == 0 else "error",
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode,
                "error": None if result.returncode == 0 else (result.stderr or result.stdout or "Script failed"),
            }
        except subprocess.TimeoutExpired:
            return {"status": "error", "error": f"Script timed out after {timeout}s"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def build_profile_plan(
        self,
        device_id: str,
        family_ids: Optional[List[str]] = None,
        modalities: Optional[List[str]] = None,
        runtime_ids: Optional[List[RuntimeId]] = None,
        imgsz: int = 640,
    ) -> tuple[List[ProfileRequest], List[Dict[str, str]]]:
        """Build a profile execution plan with run-list and skip reasons."""
        from edgecraft.core.modality import Modality
        from edgecraft.models import ensure_registries_initialized

        ensure_registries_initialized()
        requests: List[ProfileRequest] = []
        skipped: List[Dict[str, str]] = []
        requested_runtimes = set(runtime_ids) if runtime_ids else None
        modality_filter = {Modality(m) for m in modalities} if modalities else None
        device_spec = DeviceRegistry.get(device_id)
        device_runtime_set = set(device_spec.supported_runtimes) if device_spec else set()
        device_class = device_spec.device_class if device_spec else DeviceClass.GENERIC

        specs = FamilyRegistry.list_models(edge_only=False)
        for spec in specs:
            if family_ids and spec.family_id not in family_ids:
                continue
            if modality_filter and spec.modality not in modality_filter:
                continue
            plugin = FamilyRegistry.get_plugin(spec.family_id)
            if plugin is None:
                skipped.append(
                    {"model_id": spec.model_id, "runtime": "-", "reason": "no_plugin"}
                )
                continue
            plugin_runtimes = set(plugin.get_supported_profile_runtimes(spec))
            if requested_runtimes is None:
                # Auto mode: only plan runtimes that are both declared by the
                # model spec and supported by the family's profiling provider.
                # This keeps plan output clean (avoid mass "no_provider").
                candidate_runtimes = [
                    r for r in spec.supported_runtimes if r in plugin_runtimes
                ]
            else:
                candidate_runtimes = [r for r in spec.supported_runtimes if r in requested_runtimes]
            for runtime_id in candidate_runtimes:
                if runtime_id not in spec.supported_runtimes:
                    skipped.append(
                        {
                            "model_id": spec.model_id,
                            "runtime": runtime_id.value,
                            "reason": "runtime_not_declared",
                        }
                    )
                    continue
                if device_runtime_set and runtime_id not in device_runtime_set:
                    skipped.append(
                        {
                            "model_id": spec.model_id,
                            "runtime": runtime_id.value,
                            "reason": "device_incompatible",
                        }
                    )
                    continue
                if runtime_id not in plugin_runtimes:
                    skipped.append(
                        {
                            "model_id": spec.model_id,
                            "runtime": runtime_id.value,
                            "reason": "no_provider",
                        }
                    )
                    continue
                try:
                    probe_script = plugin.render_profile_benchmark_script(
                        model_spec=spec,
                        runtime_id=runtime_id,
                        imgsz=imgsz,
                        warmup=1,
                        iterations=1,
                    )
                except Exception:
                    probe_script = ""
                if not probe_script:
                    skipped.append(
                        {
                            "model_id": spec.model_id,
                            "runtime": runtime_id.value,
                            "reason": "no_provider",
                        }
                    )
                    continue
                if spec.modality == Modality.VISION:
                    model_imgsz = int(spec.default_hyperparams.get("imgsz", imgsz))
                    input_signature = {"imgsz": model_imgsz, "batch_size": 1}
                elif spec.modality == Modality.AUDIO:
                    durations = spec.benchmark_input_signature.get("audio_duration_s", [3])
                    sample_rate = int(spec.default_hyperparams.get("sample_rate", 16000))
                    input_signature = {
                        "audio_duration_s": int(durations[0]) if durations else 3,
                        "sample_rate": sample_rate,
                        "batch_size": 1,
                    }
                elif spec.modality == Modality.TEXT:
                    input_signature = {
                        "max_length": int(spec.default_hyperparams.get("max_length", 128)),
                        "batch_size": 1,
                    }
                else:
                    input_signature = {"batch_size": 1}

                requests.append(
                    ProfileRequest(
                        model_id=spec.model_id,
                        family_id=spec.family_id,
                        task_type=spec.supported_tasks[0].value if spec.supported_tasks else "classification",
                        device_id=device_id,
                        device_class=device_class,
                        runtime_id=runtime_id,
                        quant_mode=QuantMode.FP16,
                        export_format=(
                            ExportFormat.ENGINE
                            if runtime_id == RuntimeId.TENSORRT
                            else (ExportFormat.PT if runtime_id == RuntimeId.PYTORCH else ExportFormat.ONNX)
                        ),
                        input_signature=input_signature,
                        warmup_iterations=10,
                        benchmark_iterations=100,
                    )
                )
        return requests, skipped

    def profile_registered_models(
        self,
        device_id: str,
        device_ip: str,
        ssh_key: str,
        docker_image: str,
        family_ids: Optional[List[str]] = None,
        modalities: Optional[List[str]] = None,
        runtime_ids: Optional[List[RuntimeId]] = None,
        imgsz: int = 640,
        store: bool = True,
        timeout: int = 600,
        store_path: Optional[Path] = None,
    ) -> List[ProfileResult]:
        """Profile planned model/runtime pairs on a remote edge device."""
        run_list, _ = self.build_profile_plan(
            device_id=device_id,
            family_ids=family_ids,
            modalities=modalities,
            runtime_ids=runtime_ids,
            imgsz=imgsz,
        )

        results: List[ProfileResult] = []
        for request in run_list:
            request.environment_versions["docker_image"] = docker_image
            logger.info(f"Profiling {request.model_id} {request.runtime_id.value}...")
            per_timeout = timeout
            if request.runtime_id == RuntimeId.TENSORRT and timeout <= 600:
                # Stable default for TRT build+benchmark on edge devices.
                per_timeout = 2700
            result = self.profile_remote(
                request=request,
                device_ip=device_ip,
                ssh_key=ssh_key,
                docker_image=docker_image,
                timeout=per_timeout,
            )
            results.append(result)
            if store:
                profile_store = ProfileStore(store_path) if store_path else get_profile_store()
                profile_store.store(result)
        return results
