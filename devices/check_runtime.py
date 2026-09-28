#!/usr/bin/env python3
"""Run a tiny inference after device setup (compatible with Python 3.6+)."""
import argparse
import base64
import ctypes
import ctypes.util
import io
import json
import sys

import numpy as np

# Synthetic y = x + 1, input/output float32[1,4]; ONNX opset 11 / IR 7.
ONNX = "CAc6XAoQCgF4CgNvbmUSAXkiA0FkZBINcnVudGltZS1jaGVjayoPCAEQASIEAACAP0IDb25lWhMKAXgSDgoMCAESCAoCCAEKAggEYhMKAXkSDgoMCAESCAoCCAEKAggEQgQKABAL"
# The same Add graph in the TFLite v3 format.
TFLITE = "GAAAAFRGTDMAAA4AFAAQAAgADAAAAAQADgAAABAAAAAYAAAAHAAAAAMAAAACAAAANAEAABgBAAABAAAADAAAAAEAAAAUAAAA6P7//wwAFAAQAAwACAAEAAwAAAAQAAAAFAAAABgAAAAcAAAAAQAAADQAAAABAAAAAgAAAAEAAAAAAAAAAwAAAKAAAABwAAAAQAAAAAAADgAUAAAAEAAMAAsABAAOAAAAEAAAAAAAAAsMAAAAEAAAAFj///8BAAAAAgAAAAIAAAAAAAAAAQAAALT///8IAAAAEAAAAAIAAAABAAAABAAAAAEAAAB5AAAADAAQAAgAAAAEAAwADAAAAAEAAAAIAAAADAAAAAEAAAABAAAAAwAAAG9uZQAMAAwABAAAAAAACAAMAAAACAAAABAAAAACAAAAAQAAAAQAAAABAAAAeAAGAAgABAAGAAAABAAAAAQAAAAAAIA//P///wQABAAEAAAA"


def run_tensorrt(x):
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(base64.b64decode(ONNX)):
        raise RuntimeError("; ".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    if hasattr(config, "set_memory_pool_limit"):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 16 * 1024 * 1024)
    else:
        config.max_workspace_size = 16 * 1024 * 1024
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build failed")
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized)
    context = engine.create_execution_context()
    cuda = ctypes.CDLL(ctypes.util.find_library("cudart") or "libcudart.so")
    def checked(name, *args):
        code = getattr(cuda, name)(*args)
        if code:
            raise RuntimeError("{} returned CUDA error {}".format(name, code))
    inp, out, stream = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()
    y = np.empty_like(x)
    try:
        checked("cudaMalloc", ctypes.byref(inp), ctypes.c_size_t(x.nbytes))
        checked("cudaMalloc", ctypes.byref(out), ctypes.c_size_t(y.nbytes))
        checked("cudaStreamCreate", ctypes.byref(stream))
        checked("cudaMemcpy", inp, ctypes.c_void_p(x.ctypes.data), ctypes.c_size_t(x.nbytes), 1)
        if hasattr(engine, "num_io_tensors"):
            context.set_tensor_address("x", inp.value)
            context.set_tensor_address("y", out.value)
            ok = context.execute_async_v3(stream_handle=stream.value)
        else:
            bindings = [0] * engine.num_bindings
            bindings[engine.get_binding_index("x")] = inp.value
            bindings[engine.get_binding_index("y")] = out.value
            ok = context.execute_async_v2(bindings=bindings, stream_handle=stream.value)
        if not ok:
            raise RuntimeError("TensorRT execution failed")
        checked("cudaStreamSynchronize", stream)
        checked("cudaMemcpy", ctypes.c_void_p(y.ctypes.data), out, ctypes.c_size_t(y.nbytes), 2)
    finally:
        if stream.value:
            cuda.cudaStreamDestroy(stream)
        for pointer in (inp, out):
            if pointer.value:
                cuda.cudaFree(pointer)
    return y, trt.__version__, "CUDA"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runtime", choices=["pytorch", "onnxruntime", "tensorrt", "litert"])
    args = parser.parse_args()
    x = np.array([[1, 2, 3, 4]], dtype=np.float32)
    if args.runtime == "pytorch":
        import torch

        torch.set_num_threads(1)
        class AddOne(torch.nn.Module):
            def forward(self, value):
                return value + 1
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tensor = torch.from_numpy(x).to(device)
        model = torch.jit.trace(AddOne().to(device), tensor)
        buffer = io.BytesIO()
        torch.jit.save(model, buffer)
        buffer.seek(0)
        restored = torch.jit.load(buffer, map_location=device)
        y = restored(tensor).cpu().numpy()
        version, backend = torch.__version__, device
    elif args.runtime == "onnxruntime":
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(base64.b64decode(ONNX), sess_options=options,
                                       providers=["CPUExecutionProvider"])
        y = session.run(None, {"x": x})[0]
        version, backend = ort.__version__, "CPUExecutionProvider"
    elif args.runtime == "litert":
        from ai_edge_litert.interpreter import Interpreter
        import ai_edge_litert

        interpreter = Interpreter(model_content=base64.b64decode(TFLITE), num_threads=1)
        interpreter.allocate_tensors()
        interpreter.set_tensor(interpreter.get_input_details()[0]["index"], x)
        interpreter.invoke()
        y = interpreter.get_tensor(interpreter.get_output_details()[0]["index"])
        version, backend = ai_edge_litert.__version__, "CPU"
    else:
        y, version, backend = run_tensorrt(x)
    np.testing.assert_allclose(y, x + 1, atol=1e-6)
    print(json.dumps({"runtime": args.runtime, "version": version, "backend": backend,
                      "python": sys.version.split()[0], "output": y.tolist(), "status": "pass"}))


if __name__ == "__main__":
    main()
