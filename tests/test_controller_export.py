"""Small real export check when the full controller dependencies are installed."""
import pytest


def test_tree_model_exports_and_runs_on_cpu():
    np = pytest.importorskip("numpy")
    sklearn = pytest.importorskip("sklearn.ensemble")
    converter = pytest.importorskip("skl2onnx")
    ort = pytest.importorskip("onnxruntime")
    inputs = np.arange(80, dtype=np.float32).reshape(20, 4)
    labels = np.arange(20) % 2
    model = sklearn.HistGradientBoostingClassifier(max_iter=2).fit(inputs, labels)
    artifact = converter.to_onnx(
        model, inputs, target_opset=17, options={id(model): {"zipmap": False}},
    )
    session = ort.InferenceSession(artifact.SerializeToString(), providers=["CPUExecutionProvider"])
    outputs = session.run(None, {session.get_inputs()[0].name: inputs})
    np.testing.assert_allclose(outputs[1], model.predict_proba(inputs), rtol=1e-5)
