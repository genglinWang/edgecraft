#!/usr/bin/env bash
set -euo pipefail

# Run on a 64-bit Raspberry Pi OS / Debian 12 host with Python 3.11 + venv.
# The destination is new and separate from any existing runtime environment.
env_root="${1:-$HOME/.local/share/edgecraft/runtimes}"
if [[ -e "$env_root" ]]; then
  echo "Destination already exists; choose a new runtime directory: $env_root" >&2
  exit 1
fi
mkdir -p "$env_root"
for runtime in torch onnx litert; do
  python3 -m venv "$env_root/$runtime"
  "$env_root/$runtime/bin/python" -m pip install --upgrade pip
done
"$env_root/torch/bin/python" -m pip install torch==2.9.1 \
  torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cpu
"$env_root/torch/bin/python" -m pip install numpy==1.26.4 transformers==4.57.6 \
  soundfile pyyaml pandas scikit-learn
"$env_root/onnx/bin/python" -m pip install numpy==1.26.4 onnxruntime==1.24.3 \
  transformers==4.57.6 soundfile pyyaml pandas scikit-learn
"$env_root/litert/bin/python" -m pip install numpy==1.26.4 ai-edge-litert==2.1.5 \
  soundfile pyyaml
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$env_root/torch/bin/python" "$script_dir/check_runtime.py" pytorch
"$env_root/onnx/bin/python" "$script_dir/check_runtime.py" onnxruntime
"$env_root/litert/bin/python" "$script_dir/check_runtime.py" litert
echo "Runtime environments created under $env_root"
