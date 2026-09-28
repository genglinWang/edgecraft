"""Device analyzer: probe edge device via SSH for model, CPU, GPU, CUDA, jtop (Jetson), etc."""
import re
from typing import Optional, Dict, Any, List, Tuple
from pydantic import BaseModel, Field

from edgecraft.utils.network import run_remote_command


# Known device-tree / model strings -> devices.yaml key and display name
DEVICE_MODEL_MAP: List[Tuple[str, str, str]] = [
    # (pattern (re), device_id, display_name)
    (r"Jetson\s+AGX\s+Orin|Jetson\s+Orin\s+AGX", "jetson_orin_agx", "NVIDIA Jetson Orin AGX"),
    (r"Jetson\s+Orin\s+Nano|Jetson\s+Nano\s+Orin", "jetson_orin_nano", "NVIDIA Jetson Orin Nano"),
    (r"Jetson\s+Orin\s+NX|Jetson\s+NX\s+Orin", "jetson_orin_nx", "NVIDIA Jetson Orin NX"),
    (r"Jetson\s+Xavier\s+NX|Jetson\s+NX\s+Xavier", "jetson_xavier_nx", "NVIDIA Jetson Xavier NX"),
    (r"Jetson\s+AGX\s+Xavier|Jetson\s+Xavier\s+AGX|Jetson Xavier", "jetson_xavier", "NVIDIA Jetson Xavier"),
    (r"Jetson\s+TX2|\bquill\b", "jetson_tx2", "NVIDIA Jetson TX2"),
    (r"Jetson\s+Nano", "jetson_nano", "NVIDIA Jetson Nano"),
    (r"Raspberry\s+Pi\s+5", "raspberry_pi_5", "Raspberry Pi 5"),
    (r"Raspberry\s+Pi\s+4", "raspberry_pi_4", "Raspberry Pi 4"),
    (r"Raspberry\s+Pi\s+3", "raspberry_pi_3", "Raspberry Pi 3"),
    (r"Raspberry\s+Pi", "raspberry_pi", "Raspberry Pi"),
]


class DeviceInfo(BaseModel):
    """Structured device information from remote probe."""

    status: str = Field(description="success | error")
    device_family: str = Field(default="unknown", description="jetson | raspberry_pi | unknown")
    device_id: Optional[str] = Field(default=None, description="Key matching devices.yaml, e.g. jetson_xavier_nx")
    display_name: str = Field(default="Unknown device", description="Human-readable model name")
    has_gpu: bool = Field(default=False, description="Whether device has NVIDIA GPU (Jetson) or discrete GPU")
    arch: str = Field(default="", description="uname -m e.g. aarch64, armv7l, x86_64")
    cpu_model: Optional[str] = Field(default=None, description="CPU model name")
    cpu_cores: Optional[int] = Field(default=None, description="Number of CPU cores")
    memory_gb: Optional[float] = Field(default=None, description="Total RAM in GB")
    # Jetson-specific
    l4t_version: Optional[str] = Field(default=None, description="L4T version e.g. R35.4.1")
    jetpack_version: Optional[str] = Field(default=None, description="JetPack version if derivable")
    cuda_version: Optional[str] = Field(default=None, description="CUDA version string")
    gpu_name: Optional[str] = Field(default=None, description="GPU name from nvidia-smi")
    jtop_available: bool = Field(default=False, description="Whether jtop is installed (Jetson)")
    # Raw / extra
    raw_model: Optional[str] = Field(default=None, description="Raw /proc/device-tree/model or similar")
    error: Optional[str] = Field(default=None, description="Error message when status=error")


def _run_probe_script(device_ip: str, ssh_key: Optional[str] = None) -> str:
    """Run a single SSH command that collects all probe data with section markers."""
    # Single-line script so it runs reliably via ssh "cmd"
    script = (
        "echo '---MODEL---'; "
        "(cat /sys/firmware/devicetree/base/model 2>/dev/null || cat /proc/device-tree/model 2>/dev/null) | tr -d '\\0'; echo; "
        "echo '---L4T---'; cat /etc/nv_tegra_release 2>/dev/null; echo; "
        "echo '---UNAME---'; uname -m; echo; "
        "echo '---LSCPU---'; lscpu 2>/dev/null; echo; "
        "echo '---NVIDIA-SMI---'; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null; echo; "
        "echo '---JTOP---'; (which jtop >/dev/null 2>&1 && echo jtop_available) || echo jtop_not_found; echo; "
        "echo '---CUDA---'; (nvcc --version 2>/dev/null || cat /usr/local/cuda/version.txt 2>/dev/null); echo; "
        "echo '---MEM---'; free -b 2>/dev/null | awk '/^Mem:/{print int($2/1024/1024/1024)}'; echo"
    )
    code, out, err = run_remote_command(device_ip, script, ssh_key=ssh_key, timeout=20)
    if code != 0:
        return f"exit_code={code}\nstderr={err}\nstdout={out}"
    return out


def _parse_sections(output: str) -> Dict[str, str]:
    """Split probe output into sections by ---LABEL---."""
    sections: Dict[str, str] = {}
    current: Optional[str] = None
    buf: List[str] = []
    for line in output.splitlines():
        if line.strip().startswith("---") and line.strip().endswith("---"):
            if current is not None:
                sections[current] = "\n".join(buf).strip()
            current = line.strip().strip("-").strip().lower()
            buf = []
        else:
            if current is not None:
                buf.append(line)
    if current is not None:
        sections[current] = "\n".join(buf).strip()
    return sections


def _match_device_id(raw_model: str) -> Tuple[Optional[str], str]:
    """Map raw model string to device_id and display_name."""
    raw = "".join(ch for ch in (raw_model or "") if ch.isprintable()).strip()
    for pattern, device_id, display_name in DEVICE_MODEL_MAP:
        if re.search(pattern, raw, re.IGNORECASE):
            return device_id, display_name
    if raw:
        return None, raw
    return None, "Unknown device"


def _infer_device_from_arch(
    *,
    arch: str,
    cpu_model: Optional[str],
    raw_model: Optional[str],
) -> Tuple[Optional[str], str, str, bool]:
    """Infer generic non-device-tree targets from stable OS evidence.

    Device-tree model strings are absent on ordinary x86 machines.  Treating
    that as an Orin Nano is worse than admitting it is a generic CPU box.
    """
    arch_norm = (arch or "").strip().lower()
    raw = (raw_model or "").strip()
    cpu = (cpu_model or "").strip()
    if arch_norm in {"x86_64", "amd64"}:
        return "x86_cpu_desktop", "Desktop CPU", "x86_cpu", False
    if arch_norm in {"aarch64", "arm64"} and raw:
        # Unknown arm64 board: keep it unknown rather than inventing a Jetson.
        return None, raw, "unknown", False
    if arch_norm in {"aarch64", "arm64"}:
        label = cpu or "Generic ARM64 CPU"
        return None, label, "unknown", False
    return None, raw or cpu or "Unknown device", "unknown", False


def _parse_l4t(l4t_text: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract L4T version (e.g. R35.4.1) and optionally JetPack from /etc/nv_tegra_release."""
    if not l4t_text:
        return None, None
    # e.g. "# R35 (release), REVISION: 4.1, ..." -> R35.4.1
    m = re.search(r"R(\d+)\s*\([^)]*\),\s*REVISION:\s*([\d.]+)", l4t_text)
    if m:
        l4t = f"R{m.group(1)}.{m.group(2)}"
        # Rough JetPack from L4T: R32 -> 4.x, R35 -> 5.x, R36 -> 6.x
        major = int(m.group(1))
        if major >= 36:
            jp = "6.x"
        elif major >= 35:
            jp = "5.x"
        elif major >= 34:
            jp = "5.x"
        elif major >= 32:
            jp = "4.x"
        else:
            jp = None
        return l4t, jp
    return None, None


def _parse_lscpu(lscpu_text: str) -> Tuple[Optional[str], Optional[int]]:
    """Extract CPU model and core count from lscpu output."""
    if not lscpu_text:
        return None, None
    model = None
    cores = None
    for line in lscpu_text.splitlines():
        if "Model name:" in line:
            model = line.split(":", 1)[1].strip()
        if "CPU(s):" in line:
            try:
                cores = int(line.split(":", 1)[1].strip())
            except ValueError:
                pass
    return model, cores


def _parse_cuda(cuda_text: str) -> Optional[str]:
    """Extract CUDA version from nvcc --version or version.txt."""
    if not cuda_text:
        return None
    # nvcc: "release 12.2, V12.2.10"
    m = re.search(r"release\s+([\d.]+)|V([\d.]+)|CUDA\s+([\d.]+)", cuda_text, re.IGNORECASE)
    if m:
        return (m.group(1) or m.group(2) or m.group(3) or "").strip()
    return cuda_text.strip()[:30] if cuda_text.strip() else None


def run(device_ip: str, ssh_key: Optional[str] = None) -> DeviceInfo:
    """
    Probe the edge device via SSH and return structured device info.

    Identifies: Jetson (Xavier NX, Xavier, AGX Orin, Orin Nano, TX2, etc.),
    Raspberry Pi (3/4/5), CPU, GPU, CUDA (Jetson), jtop availability (Jetson), memory.

    Args:
        device_ip: SSH target (user@host or host).
        ssh_key: Optional path to SSH private key.

    Returns:
        DeviceInfo with status, device_id, display_name, has_gpu, cpu/gpu/cuda/jtop fields.
    """
    try:
        output = _run_probe_script(device_ip, ssh_key)
    except Exception as e:
        return DeviceInfo(status="error", error=str(e))

    sections = _parse_sections(output)

    # If output looks like an error message (exit_code=...)
    if output.strip().startswith("exit_code="):
        return DeviceInfo(status="error", error=output.strip()[:500])

    raw_model = sections.get("model", "").strip() or None
    arch = sections.get("uname", "").strip() or ""
    cpu_model, cpu_cores = _parse_lscpu(sections.get("lscpu", ""))
    device_id, display_name = _match_device_id(raw_model or "")
    device_family = "unknown"
    inferred_gpu = False
    if device_id:
        if device_id.startswith("jetson"):
            device_family = "jetson"
        elif device_id.startswith("raspberry_pi"):
            device_family = "raspberry_pi"
    else:
        device_id, display_name, device_family, inferred_gpu = _infer_device_from_arch(
            arch=arch,
            cpu_model=cpu_model,
            raw_model=raw_model,
        )

    has_gpu = bool(inferred_gpu)
    gpu_name = None
    nvidia_smi = sections.get("nvidia-smi", "").strip()
    if nvidia_smi and "nvidia" in nvidia_smi.lower():
        has_gpu = True
        # First line often: "GPU Name, 535.xx.xx, 8192 MiB"
        parts = nvidia_smi.split(",")
        if parts:
            gpu_name = parts[0].strip()

    if device_family == "jetson":
        # Jetson has integrated GPU even if nvidia-smi not available in container
        has_gpu = True
        if not gpu_name:
            gpu_name = "Jetson integrated GPU"

    l4t_version, jetpack_version = _parse_l4t(sections.get("l4t", ""))
    cuda_version = _parse_cuda(sections.get("cuda", ""))
    jtop_available = "jtop_available" in (sections.get("jtop", "") or "")

    memory_gb = None
    mem_str = sections.get("mem", "").strip()
    if mem_str and mem_str.isdigit():
        memory_gb = int(mem_str)

    return DeviceInfo(
        status="success",
        device_family=device_family,
        device_id=device_id,
        display_name=display_name,
        has_gpu=has_gpu,
        arch=arch,
        cpu_model=cpu_model,
        cpu_cores=cpu_cores,
        memory_gb=float(memory_gb) if memory_gb is not None else None,
        l4t_version=l4t_version,
        jetpack_version=jetpack_version,
        cuda_version=cuda_version,
        gpu_name=gpu_name,
        jtop_available=jtop_available,
        raw_model=raw_model,
    )


def get_device_info_display_rows(device_info: DeviceInfo) -> List[Tuple[str, str]]:
    """Return (field, value) rows for CLI display (e.g. in a Rich table)."""
    if device_info.status != "success":
        return [("Error", device_info.error or "Probe failed")]
    rows: List[Tuple[str, str]] = []
    rows.append(("Model", device_info.display_name))
    if device_info.device_id:
        rows.append(("Device ID", device_info.device_id))
    rows.append(("Family", device_info.device_family))
    rows.append(("GPU", "Yes" if device_info.has_gpu else "No"))
    if device_info.gpu_name:
        rows.append(("GPU Name", device_info.gpu_name))
    if device_info.arch:
        rows.append(("Arch", device_info.arch))
    if device_info.cpu_model:
        rows.append(("CPU", device_info.cpu_model))
    if device_info.cpu_cores is not None:
        rows.append(("Cores", str(device_info.cpu_cores)))
    if device_info.memory_gb is not None:
        rows.append(("Memory", f"{device_info.memory_gb:.1f} GB"))
    if device_info.l4t_version:
        rows.append(("L4T", device_info.l4t_version))
    if device_info.jetpack_version:
        rows.append(("JetPack", device_info.jetpack_version))
    if device_info.cuda_version:
        rows.append(("CUDA", device_info.cuda_version))
    if device_info.device_family == "jetson":
        rows.append(("jtop", "Available" if device_info.jtop_available else "Not installed"))
    return rows
