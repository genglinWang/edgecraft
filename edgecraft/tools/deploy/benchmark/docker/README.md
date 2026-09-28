# EdgeCraft Edge Benchmark Docker Images

This directory contains Dockerfiles for building benchmark images for different edge devices.

## Available Images

### Raspberry Pi (`Dockerfile.rpi`)

A lightweight Python image for CPU inference on Raspberry Pi 5B.

**Included packages:**
- `onnxruntime` - ONNX inference engine
- `ultralytics` - YOLO models (CPU mode)
- `opencv-python-headless` - Image processing
- `psutil` - System monitoring

**Building the image:**

```bash
# On Raspberry Pi (native build)
cd edgecraft/tools/deploy/benchmark/docker
docker build -t edgecraft/rpi-benchmark:latest -f Dockerfile.rpi .

# Cross-platform build (from x86 host)
docker buildx build --platform linux/arm64 \
  -t edgecraft/rpi-benchmark:latest \
  -f Dockerfile.rpi \
  --push .
```

**Usage:**

```bash
docker run --rm \
  -v /path/to/model:/workspace \
  -w /workspace \
  edgecraft/rpi-benchmark:latest \
  python benchmark.py --model model.onnx --backend onnx
```

## Jetson Images

For Jetson devices, we use pre-built images from [jetson-containers](https://github.com/dusty-nv/jetson-containers):

| Device | JetPack | Image |
|--------|---------|-------|
| Orin AGX/Nano | 6.x | `dustynv/ultralytics:r36.4.0` |
| Xavier NX/AGX | 5.x | `dustynv/ultralytics:r35.4.1` |
| TX2 | 4.x | `dustynv/l4t-pytorch:r32.7.1` |

These images include:
- PyTorch with CUDA support
- TensorRT for optimized inference
- Ultralytics YOLO

## Notes

- Models for Raspberry Pi should be exported to ONNX format
- Jetson devices support `.pt`, `.onnx`, and `.engine` formats
- TensorRT engines (`.engine`) must be built on the target device
