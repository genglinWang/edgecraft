# Device setup

The controller trains candidates; the target device loads and evaluates their
artifacts. Use a dedicated device account with SSH key access and a writable
home directory. The bundled one-shot runner requires `bash`, `tar`, `gzip`,
`timeout`, and `flock`. Docker targets additionally need permission to run Docker.
Keep several GB free for model bundles and runtime builds.

## Supported platform families

| Target | Execution | Platform used for runtime checks |
| --- | --- | --- |
| Jetson AGX Orin | Docker / NVIDIA runtime | JetPack 6.2.1, L4T 36.4.7, Ubuntu 22.04 |
| Jetson Xavier NX | Docker / NVIDIA runtime | JetPack 5.1.6, L4T 35.6.4, Ubuntu 20.04 |
| Jetson TX2 | Docker / NVIDIA runtime | JetPack 4.6.6, L4T 32.7.6, Ubuntu 18.04 |
| Raspberry Pi 5 | Native environments | 64-bit Debian 12, Python 3.11 |
| Desktop CPU | Docker | Linux x86_64, Python 3.11 container |

Jetson images must match the host JetPack generation. Check it with
`head -1 /etc/nv_tegra_release`. The Dockerfiles extend publicly available
[Jetson Containers](https://github.com/dusty-nv/jetson-containers) PyTorch images;
[NVIDIA's compatibility guide](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html)
explains the host/framework dependency. PyTorch, ONNX Runtime, and TensorRT are
separate execution choices: importing TensorRT does not imply that PyTorch or
ONNX Runtime uses the GPU. The runtime checks below print the actual backend.

## Docker targets

Copy the `devices/` directory to the target. Build the matching image there:

```bash
# Choose exactly one Dockerfile for this host.
docker build -f devices/Dockerfile.orin -t edgecraft-runtime:orin devices
docker build -f devices/Dockerfile.xavier -t edgecraft-runtime:xavier devices
docker build -f devices/Dockerfile.tx2 -t edgecraft-runtime:tx2 devices
docker build -f devices/Dockerfile.desktop -t edgecraft-runtime:desktop devices
```

These are inference environments; do not install the controller package in the
older Jetson Python environments. TX2 uses Python 3.6 and correspondingly older
runtime packages. Its candidate export must use an ONNX version supported by
the installed runtime. Runtime preflight exposes the actual package versions to
the synthesis agent.

After building, check a Jetson image (replace `orin` with the chosen tag):

```bash
for runtime in pytorch onnxruntime tensorrt; do
  docker run --rm --runtime nvidia edgecraft-runtime:orin \
    python3 /opt/edgecraft/check_runtime.py "$runtime"
done
```

For desktop CPU, omit `--runtime nvidia` and test `pytorch onnxruntime litert`.
The check loads a small synthetic model and verifies its output. It downloads
no model weights. Set `EDGE_DOCKER_IMAGE` on the controller to the image tag
present on the target device.

## Raspberry Pi 5

Use Raspberry Pi OS / Debian 12 **64-bit**, with `python3-venv` and `libsndfile1`
installed by the device administrator. Copy `devices/` to the Pi and run:

```bash
bash devices/setup_pi5.sh "$HOME/.local/share/edgecraft/runtimes"
```

This creates separate PyTorch, ONNX Runtime, and LiteRT environments and runs
one inference in each. Configure their paths on the **controller**:

```bash
export PI_RUNTIME_ROOT=/home/your-device-user/.local/share/edgecraft/runtimes
export EDGECRAFT_DEVICE_MANIFEST="$PWD/devices/pi5.example.json"
```

`PI_RUNTIME_ROOT` is an absolute path on the Pi, not on the controller. Supply
`--ip pi-user@pi-host --ssh-key /path/to/private_key` and omit `--docker-image`.
For several devices, copy the bundled device manifest to a local file and add
the Pi's `native_envs` mapping; set `EDGECRAFT_DEVICE_MANIFEST` to that file.

## Connection, workspaces, and measurements

Verify the device's SSH fingerprint on first connection, then use
`ssh -i "$SSH_KEY_PATH" -o IdentitiesOnly=yes "$DEVICE_HOST" true`. The same
address and key are passed to `edgecraft synth`. Keys stay on the controller.
The runner uses a device-side directory under `~/.edgecraft-runner`; override
`remote_base` in your local manifest when model bundles belong on a larger disk.

Specify only SLOs measurable on your device. Latency is measured by inference;
energy additionally needs a supported power source and valid samples. An absent
energy measurement is not treated as zero. Native and container package versions
are probed at run time and become part of verification context.
