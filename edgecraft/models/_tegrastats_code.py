"""Inject tegrastats power measurement into generated benchmark scripts.

The ``wrap_with_tegrastats`` function post-processes a rendered benchmark
script string and inserts a background thread that captures power readings
from ``tegrastats`` (NVIDIA Jetson platforms).  When tegrastats is not
available the helper thread silently no-ops and the power fields are ``null``.
"""
from __future__ import annotations

import re as _re

# ---------------------------------------------------------------------------
# Code fragments — plain Python to be inserted into generated scripts.
# ---------------------------------------------------------------------------

_FUNC = (
    "\n"
    "def _collect_tegrastats(_tg_interval, _tg_stop, _tg_samples):\n"
    "    import re as _tg_re, glob as _tg_glob, os as _tg_os\n"
    "    _tg_use_sysfs = False\n"
    "    try:\n"
    "        _tg_p = subprocess.Popen(\n"
    '            ["tegrastats", "--interval", str(_tg_interval)],\n'
    "            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,\n"
    "        )\n"
    "        _tg_nopwr = 0\n"
    "        for _tg_line in _tg_p.stdout:\n"
    "            if _tg_stop.is_set():\n"
    "                break\n"
    "            _tg_val = None\n"
    "            for _tg_pat in [\n"
    "                r'VIN_SYS_5V0\\s+(\\d+)mW',\n"
    "                r'VDD_IN\\s+(\\d+)mW',\n"
    "                r'POM_5V_IN\\s+(\\d+)',\n"
    "            ]:\n"
    "                _tg_m = _tg_re.search(_tg_pat, _tg_line)\n"
    "                if _tg_m:\n"
    "                    _tg_val = float(_tg_m.group(1)) / 1000.0\n"
    "                    break\n"
    "            if _tg_val is None:\n"
    "                _tg_m = _tg_re.search(r'(\\d+\\.?\\d*)\\s*W\\b', _tg_line)\n"
    "                if _tg_m:\n"
    "                    _tg_val = float(_tg_m.group(1))\n"
    "            if _tg_val is not None:\n"
    "                _tg_samples.append(_tg_val)\n"
    "            else:\n"
    "                _tg_nopwr += 1\n"
    "                if _tg_nopwr >= 3:\n"
    "                    _tg_use_sysfs = True\n"
    "                    break\n"
    "        _tg_p.terminate()\n"
    "    except (FileNotFoundError, Exception):\n"
    "        _tg_use_sysfs = True\n"
    "    if _tg_use_sysfs and not _tg_stop.is_set():\n"
    "        _tg_vp = _tg_cp = None\n"
    "        for _tg_hw in _tg_glob.glob('/sys/class/hwmon/hwmon*'):\n"
    "            try:\n"
    "                with open(_tg_hw + '/name') as _f:\n"
    "                    if _f.read().strip() != 'ina3221':\n"
    "                        continue\n"
    "                for _tg_i in range(1, 4):\n"
    "                    try:\n"
    "                        with open(_tg_hw + '/in%d_label' % _tg_i) as _f:\n"
    "                            if 'VDD_IN' in _f.read():\n"
    "                                _tg_vp = _tg_hw + '/in%d_input' % _tg_i\n"
    "                                _tg_cp = _tg_hw + '/curr%d_input' % _tg_i\n"
    "                                break\n"
    "                    except (IOError, OSError):\n"
    "                        continue\n"
    "            except (IOError, OSError):\n"
    "                continue\n"
    "            if _tg_vp:\n"
    "                break\n"
    "        if _tg_vp:\n"
    "            while not _tg_stop.is_set():\n"
    "                try:\n"
    "                    with open(_tg_vp) as _f:\n"
    "                        _v = float(_f.read().strip())\n"
    "                    with open(_tg_cp) as _f:\n"
    "                        _c = float(_f.read().strip())\n"
    "                    _pw = _v * _c / 1e6\n"
    "                    if _pw > 0:\n"
    "                        _tg_samples.append(_pw)\n"
    "                except (IOError, OSError, ValueError):\n"
    "                    pass\n"
    "                _tg_stop.wait(_tg_interval / 1000.0)\n"
    "\n"
)

_START_LINES = [
    "_tg_pwr = []",
    "_tg_evt = threading.Event()",
    "_tg_thr = threading.Thread(target=_collect_tegrastats, args=(100, _tg_evt, _tg_pwr), daemon=True)",
    "_tg_thr.start()",
    "time.sleep(0.3)",
]

_STOP_LINES = [
    "_tg_evt.set()",
    "_tg_thr.join(timeout=2)",
    "_tg_avg_w = (sum(_tg_pwr) / len(_tg_pwr)) if _tg_pwr else None",
]


def _indent_block(lines: list[str], indent: str) -> str:
    """Join *lines* with *indent* prefix and newline suffix."""
    return "".join(f"{indent}{l}\n" for l in lines)


def wrap_with_tegrastats(script: str) -> str:
    """Add tegrastats power measurement to a generated benchmark script.

    Handles both ONNX / PyTorch loop-based scripts and ``trtexec``-based
    scripts.  The injection is idempotent — calling it on an already-wrapped
    script is a no-op.
    """
    if not script or "_collect_tegrastats" in script:
        return script

    # ---- 1. Ensure required imports ----------------------------------------
    if "import threading" not in script:
        for anchor in ("import subprocess\n", "import time\n", "import json\n"):
            if anchor in script:
                script = script.replace(anchor, anchor + "import threading\n", 1)
                break
    if "import subprocess" not in script:
        script = script.replace(
            "import threading\n", "import subprocess\nimport threading\n", 1
        )
    if "import time" not in script:
        script = script.replace(
            "import threading\n", "import threading\nimport time\n", 1
        )

    # ---- 2. Insert helper function after first raise SystemExit(1) ---------
    m = _re.search(r"raise SystemExit\(1\)\n", script)
    if m:
        script = script[: m.end()] + _FUNC + script[m.end() :]

    # ---- 3. Insert START before benchmark input / trtexec cmd --------------
    start_m = _re.search(
        r"^(\s*)(dummy\w*\s*=|feeds\s*=|wavs\s*=|cmd\s*=\s*\[)",
        script,
        _re.MULTILINE,
    )
    if not start_m:
        # Fallback: first ``for _ in range(`` at any indentation
        start_m = _re.search(
            r"^(\s*)for\s+_\s+in\s+range\(", script, _re.MULTILINE
        )
    if start_m:
        indent = start_m.group(1)
        block = _indent_block(_START_LINES, indent)
        script = script[: start_m.start()] + block + script[start_m.start() :]

    # ---- 4. Insert STOP before last print(json.dumps( ----------------------
    matches = list(
        _re.finditer(r"^(\s*)print\(json\.dumps\(", script, _re.MULTILINE)
    )
    if matches:
        last = matches[-1]
        indent = last.group(1)
        block = _indent_block(_STOP_LINES, indent) + "\n"
        script = script[: last.start()] + block + script[last.start() :]

    # ---- 5. Add power fields to JSON output --------------------------------
    is_trtexec = "lat_mean" in script
    if is_trtexec:
        energy = "(_tg_avg_w * lat_mean / 1000.0) if _tg_avg_w else None"
    else:
        energy = "(_tg_avg_w * float(arr.mean()) / 1000.0) if _tg_avg_w else None"

    raw_m = _re.search(r'^(\s*)"raw_latencies"', script, _re.MULTILINE)
    if raw_m:
        indent = raw_m.group(1)
        fields = (
            f'{indent}"power_w": _tg_avg_w,\n'
            f'{indent}"energy_mj_per_inference": {energy},\n'
        )
        script = script[: raw_m.start()] + fields + script[raw_m.start() :]

    return script
