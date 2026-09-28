#!/usr/bin/env bash
# EdgeCraft Edge Benchmark Runner
#
# This script runs inside the edge-runner job environment.
# It starts a Docker container and executes the benchmark.
#
# Required environment variables (from meta.env):
#   DOCKER_IMAGE   - Docker image to use
#   MODEL_FILE     - Model filename in payload/
#   BACKEND        - Benchmark backend (ultralytics, onnx)
#
# Optional environment variables:
#   ITERATIONS     - Number of benchmark iterations (default: 100)
#   WARMUP         - Number of warmup iterations (default: 10)
#   IMGSZ          - Image size for vision models (default: 640)
#   HAS_GPU        - Set to "1" if device has NVIDIA GPU

set -euo pipefail

echo "[edgecraft-benchmark] Starting edge benchmark"
echo "[edgecraft-benchmark] Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[edgecraft-benchmark] Host: $(hostname)"

# Load configuration from meta.env
if [[ -f "meta.env" ]]; then
    # shellcheck disable=SC1091
    source meta.env
fi

# Set defaults
DOCKER_IMAGE="${DOCKER_IMAGE:-dustynv/ultralytics:r36.4.0}"
MODEL_FILE="${MODEL_FILE:-model.pt}"
BACKEND="${BACKEND:-ultralytics}"
ITERATIONS="${ITERATIONS:-100}"
WARMUP="${WARMUP:-10}"
IMGSZ="${IMGSZ:-640}"
HAS_GPU="${HAS_GPU:-0}"

echo "[edgecraft-benchmark] Configuration:"
echo "  DOCKER_IMAGE: $DOCKER_IMAGE"
echo "  MODEL_FILE:   $MODEL_FILE"
echo "  BACKEND:      $BACKEND"
echo "  ITERATIONS:   $ITERATIONS"
echo "  WARMUP:       $WARMUP"
echo "  IMGSZ:        $IMGSZ"
echo "  HAS_GPU:      $HAS_GPU"

# Check if model exists
if [[ ! -f "payload/$MODEL_FILE" ]]; then
    echo "[edgecraft-benchmark] ERROR: Model not found: payload/$MODEL_FILE" >&2
    ls -la payload/ >&2 || true
    exit 1
fi

# Check if benchmark script exists
if [[ ! -f "payload/benchmark.py" ]]; then
    echo "[edgecraft-benchmark] ERROR: benchmark.py not found in payload/" >&2
    exit 1
fi

# Check if backends directory exists
if [[ ! -d "payload/backends" ]]; then
    echo "[edgecraft-benchmark] ERROR: backends/ directory not found in payload/" >&2
    exit 1
fi

# Build Docker run command
DOCKER_OPTS="--rm"
DOCKER_OPTS="$DOCKER_OPTS -v $(pwd)/payload:/workspace"
DOCKER_OPTS="$DOCKER_OPTS -w /workspace"

# Add GPU runtime if available
if [[ "$HAS_GPU" == "1" ]]; then
    # Check for NVIDIA runtime
    if docker info 2>/dev/null | grep -q "nvidia"; then
        DOCKER_OPTS="$DOCKER_OPTS --runtime=nvidia"
    elif [[ -e /dev/nvhost-ctrl ]]; then
        # Jetson-specific device passthrough
        DOCKER_OPTS="$DOCKER_OPTS --runtime=nvidia"
    fi
    # Pass all GPUs
    DOCKER_OPTS="$DOCKER_OPTS --gpus all"
fi

# Add network for potential model downloads (if needed during first run)
DOCKER_OPTS="$DOCKER_OPTS --network=host"

echo ""
echo "[edgecraft-benchmark] Pulling/checking Docker image..."
docker pull "$DOCKER_IMAGE" 2>/dev/null || echo "[edgecraft-benchmark] Image pull skipped or failed, using local"

echo ""
echo "[edgecraft-benchmark] Running benchmark in container..."
echo "[edgecraft-benchmark] Command: docker run $DOCKER_OPTS $DOCKER_IMAGE python benchmark.py ..."
echo ""

# Run benchmark
# shellcheck disable=SC2086
docker run $DOCKER_OPTS "$DOCKER_IMAGE" \
    python benchmark.py \
    --model "$MODEL_FILE" \
    --backend "$BACKEND" \
    --iterations "$ITERATIONS" \
    --warmup "$WARMUP" \
    --imgsz "$IMGSZ" \
    --output metrics.json

# Check if metrics.json was created
if [[ -f "payload/metrics.json" ]]; then
    echo ""
    echo "[edgecraft-benchmark] Benchmark completed successfully"
    echo "[edgecraft-benchmark] Results:"
    cat payload/metrics.json

    # Copy metrics to job output directory
    cp payload/metrics.json ./metrics.json
else
    echo "[edgecraft-benchmark] ERROR: metrics.json not found after benchmark" >&2
    exit 1
fi

echo ""
echo "[edgecraft-benchmark] Done"
