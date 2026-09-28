"""Evaluator-owned power sampling for edge inference jobs.

Generated ``infer.py`` files describe the timed inference window.  The edge
runner samples hardware power independently and only publishes energy metrics
when both streams can be aligned.  Missing or malformed evidence stays
unknown; generated numeric power values never become trusted by themselves.
"""
from __future__ import annotations

import base64
import math
import shlex
import statistics
from collections import Counter
from typing import Any, Dict, Tuple


POWER_SAMPLE_PREFIX = "EDGECRAFT_POWER_SAMPLE\t"
POWER_IDLE_END_PREFIX = "EDGECRAFT_POWER_IDLE_END_NS="

_ENERGY_KEYS = {
    "power_w",
    "power",
    "energy_mj",
    "energy",
    "energy_mj_per_inference",
    "energy_per_inference_mj",
    "power_std_w",
    "energy_std_mj",
    "dynamic_power_w",
    "dynamic_energy_mj",
    "energy_sample_count",
}


def _sampler_python(kind: str) -> str:
    common = """
import glob
import re
import subprocess
import time

def now_ns():
    return int(time.time() * 1_000_000_000)

def read_text(path):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return handle.read().strip()
    except Exception:
        return ""

def emit(power_w, source, scope):
    if power_w is not None and power_w > 0:
        print(f"{now_ns()}\\t{power_w:.9f}\\t{source}\\t{scope}", flush=True)
"""
    if kind == "rpi_pmic":
        body = """
def sample():
    completed = subprocess.run(
        ["vcgencmd", "pmic_read_adc"], capture_output=True, text=True, timeout=2
    )
    if completed.returncode != 0:
        return None
    currents = {}
    volts = {}
    pattern = re.compile(
        r"^\\s*(\\S+)_([AV])\\s+(?:current|volt)\\(\\d+\\)=([0-9.]+)[AV]\\s*$"
    )
    for line in completed.stdout.splitlines():
        match = pattern.match(line)
        if not match:
            continue
        rail, kind, raw = match.groups()
        (currents if kind == "A" else volts)[rail] = float(raw)
    paired = set(currents) & set(volts)
    return sum(currents[rail] * volts[rail] for rail in paired) if paired else None

while True:
    try:
        emit(sample(), "rpi_pmic_adc", "internal_pmic_rails")
    except Exception:
        pass
    time.sleep(0.1)
"""
    elif kind == "jetson_ina":
        body = """
def sample_hwmon():
    for root in glob.glob("/host-sys/class/hwmon/hwmon*"):
        if read_text(root + "/name") != "ina3221":
            continue
        for index in range(1, 9):
            if read_text(f"{root}/in{index}_label") != "VDD_IN":
                continue
            voltage_mv = read_text(f"{root}/in{index}_input")
            current_ma = read_text(f"{root}/curr{index}_input")
            if voltage_mv and current_ma:
                return float(voltage_mv) * float(current_ma) / 1_000_000.0
    return None

def sample_iio():
    for root in glob.glob("/host-sys/bus/iio/devices/iio:device*"):
        if read_text(root + "/name") != "ina3221x":
            continue
        for index in range(8):
            if read_text(f"{root}/rail_name_{index}") != "VDD_IN":
                continue
            power_mw = read_text(f"{root}/in_power{index}_input")
            if power_mw:
                return float(power_mw) / 1000.0
    return None

while True:
    try:
        emit(sample_hwmon() or sample_iio(), "jetson_ina3221", "board_input")
    except Exception:
        pass
    time.sleep(0.1)
"""
    else:
        return ""
    return common + body


def edge_power_sampler_bash(kind: str, docker_image: str = "") -> str:
    """Return small bash functions that own one hardware power sampler."""
    normalized_kind = str(kind or "").strip().lower()
    code = _sampler_python(normalized_kind)
    if not code:
        return """\
edgecraft_start_power_sampler() { :; }
edgecraft_stop_power_sampler() { :; }
"""
    else:
        encoded = base64.b64encode(code.encode("utf-8")).decode("ascii")
        python_command = (
            "import base64;exec(base64.b64decode(" + repr(encoded) + "))"
        )
        if normalized_kind == "jetson_ina":
            command = (
                "docker run --rm --name \"$EDGECRAFT_POWER_CONTAINER\" "
                "--privileged -v /sys:/host-sys:ro "
                "--entrypoint python3 "
                + shlex.quote(docker_image)
                + " -u -c "
                + shlex.quote(python_command)
            )
            container_setup = 'EDGECRAFT_POWER_CONTAINER="edgecraft-power-$$"'
            container_cleanup = '''
    docker kill "$EDGECRAFT_POWER_CONTAINER" >/dev/null 2>&1 || true
    docker rm -f "$EDGECRAFT_POWER_CONTAINER" >/dev/null 2>&1 || true'''
        else:
            command = "/usr/bin/python3 -u -c " + shlex.quote(python_command)
            container_setup = 'EDGECRAFT_POWER_CONTAINER=""'
            container_cleanup = ""

    return f'''\
EDGECRAFT_POWER_LOG="/tmp/edgecraft_power_$$.tsv"
EDGECRAFT_POWER_PID=""
EDGECRAFT_POWER_IDLE_END_NS=""
EDGECRAFT_POWER_STOPPED=0
{container_setup}

edgecraft_start_power_sampler() {{
    rm -f "$EDGECRAFT_POWER_LOG"
    {command} >"$EDGECRAFT_POWER_LOG" 2>/dev/null &
    EDGECRAFT_POWER_PID=$!
    trap edgecraft_stop_power_sampler EXIT
    for _ in {{1..30}}; do
        if [[ -f "$EDGECRAFT_POWER_LOG" ]] && [[ $(wc -l < "$EDGECRAFT_POWER_LOG") -ge 3 ]]; then
            break
        fi
        if ! kill -0 "$EDGECRAFT_POWER_PID" 2>/dev/null; then
            break
        fi
        sleep 0.1
    done
    EDGECRAFT_POWER_IDLE_END_NS=$(date +%s%N)
}}

edgecraft_stop_power_sampler() {{
    if [[ "$EDGECRAFT_POWER_STOPPED" == "1" ]]; then
        return 0
    fi
    EDGECRAFT_POWER_STOPPED=1
{container_cleanup}
    if [[ -n "$EDGECRAFT_POWER_PID" ]]; then
        kill "$EDGECRAFT_POWER_PID" 2>/dev/null || true
        wait "$EDGECRAFT_POWER_PID" 2>/dev/null || true
    fi
    echo "{POWER_IDLE_END_PREFIX}$EDGECRAFT_POWER_IDLE_END_NS"
    if [[ -f "$EDGECRAFT_POWER_LOG" ]]; then
        while IFS= read -r sample; do
            echo $'{POWER_SAMPLE_PREFIX}'"$sample"
        done < "$EDGECRAFT_POWER_LOG"
        rm -f "$EDGECRAFT_POWER_LOG"
    fi
}}
'''


def _without_untrusted_energy(metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in (metrics or {}).items()
        if str(key).strip().lower() not in _ENERGY_KEYS
    }


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _power_samples(stdout: str) -> Tuple[list[Tuple[int, float, str, str]], int | None]:
    samples: list[Tuple[int, float, str, str]] = []
    idle_end_ns: int | None = None
    for line in (stdout or "").splitlines():
        if line.startswith(POWER_IDLE_END_PREFIX):
            try:
                idle_end_ns = int(line[len(POWER_IDLE_END_PREFIX):].strip())
            except ValueError:
                idle_end_ns = None
            continue
        if not line.startswith(POWER_SAMPLE_PREFIX):
            continue
        fields = line[len(POWER_SAMPLE_PREFIX):].split("\t")
        if len(fields) != 4:
            continue
        try:
            timestamp_ns = int(fields[0])
            power_w = float(fields[1])
        except ValueError:
            continue
        if timestamp_ns > 0 and math.isfinite(power_w) and power_w > 0:
            samples.append((timestamp_ns, power_w, fields[2], fields[3]))
    return samples, idle_end_ns


def merge_trusted_energy(result: Dict[str, Any], stdout: str) -> Dict[str, Any]:
    """Strip self-reported power and merge aligned evaluator-owned samples."""
    merged = dict(result or {})
    original_metrics = dict(merged.get("metrics") or {})
    had_untrusted_energy = any(
        str(key).strip().lower() in _ENERGY_KEYS for key in original_metrics
    )
    metrics = _without_untrusted_energy(original_metrics)
    protocol = dict(merged.get("measurement_protocol") or {})
    samples, idle_end_ns = _power_samples(stdout)
    if not samples and not had_untrusted_energy:
        merged["metrics"] = metrics
        return merged

    start_ns = _finite_number(protocol.get("benchmark_start_ns"))
    end_ns = _finite_number(protocol.get("benchmark_end_ns"))
    repetitions = _finite_number(
        protocol.get("actual_repetitions", protocol.get("repetitions"))
    )
    if start_ns is None or end_ns is None or end_ns <= start_ns:
        protocol.update({
            "energy_status": "unknown",
            "energy_reason": "missing_benchmark_window",
            "energy_sample_count": len(samples),
        })
        merged["metrics"] = metrics
        merged["measurement_protocol"] = protocol
        return merged

    active = [item for item in samples if int(start_ns) <= item[0] <= int(end_ns)]
    if len(active) < 3:
        protocol.update({
            "energy_status": "unknown",
            "energy_reason": "insufficient_aligned_samples",
            "energy_sample_count": len(samples),
            "energy_active_sample_count": len(active),
        })
        merged["metrics"] = metrics
        merged["measurement_protocol"] = protocol
        return merged

    source_scope = Counter((item[2], item[3]) for item in active).most_common(1)[0][0]
    active_power = [item[1] for item in active if (item[2], item[3]) == source_scope]
    idle = [
        item[1]
        for item in samples
        if idle_end_ns is not None
        and item[0] <= idle_end_ns
        and (item[2], item[3]) == source_scope
    ]
    mean_power = statistics.fmean(active_power)
    power_std = statistics.pstdev(active_power) if len(active_power) > 1 else 0.0
    idle_power = statistics.fmean(idle) if idle else None
    dynamic_power = max(0.0, mean_power - idle_power) if idle_power is not None else None
    window_s = (float(end_ns) - float(start_ns)) / 1_000_000_000.0

    metrics.update({
        "Power_w": mean_power,
        "power_std_w": power_std,
        "energy_sample_count": float(len(active_power)),
    })
    scope = str(protocol.get("benchmark_scope") or "")
    if repetitions is not None and repetitions > 0 and scope == "inference_loop":
        energy_mj = mean_power * window_s * 1000.0 / repetitions
        metrics["Energy_mj"] = energy_mj
        metrics["energy_std_mj"] = power_std * window_s * 1000.0 / repetitions
        if dynamic_power is not None:
            metrics["Dynamic_power_w"] = dynamic_power
            metrics["Dynamic_energy_mj"] = dynamic_power * window_s * 1000.0 / repetitions

    protocol.update({
        "energy_status": "measured",
        "energy_source": source_scope[0],
        "energy_scope": source_scope[1],
        "energy_sample_count": len(samples),
        "energy_active_sample_count": len(active_power),
        "energy_idle_sample_count": len(idle),
        "energy_idle_power_w": idle_power,
        "energy_window_s": window_s,
        "energy_exclusions": (
            "external_usb_hat_nvme"
            if source_scope[0] == "rpi_pmic_adc"
            else "none_declared"
        ),
        "energy_trusted": True,
    })
    merged["metrics"] = metrics
    merged["measurement_protocol"] = protocol
    return merged
