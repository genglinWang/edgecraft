from edgecraft.sdk.client import EdgeCraftClient


def test_sdk_forwards_device_constraints_and_tenant_handles(monkeypatch) -> None:
    client = EdgeCraftClient(api_key="synthetic-review-token")
    captured = {}

    def fake_post(path, data):
        captured["path"] = path
        captured["data"] = data
        return {"task_id": "task-a", "status": "queued"}

    monkeypatch.setattr(client, "_post", fake_post)
    task = client.synthesize(
        "classify review data",
        target_device="review-device",
        device_ip="reviewer@device",
        constraints={
            "latency_ms": 10,
            "Accuracy": {"comparison": "gte", "target": 0.9},
        },
        tenant_id="tenant-a",
        dataset_handle="dataset-a",
        credential_handle="key-a",
        profile="paper",
    )

    assert task.task_id == "task-a"
    assert captured["path"] == "/tasks"
    assert captured["data"]["device_id"] == "review-device"
    assert captured["data"]["dataset_handle"] == "dataset-a"
    assert captured["data"]["credential_handle"] == "key-a"
    assert captured["data"]["constraints"] == [
        {"metric": "latency_ms", "comparison": "lte", "target": 10},
        {
            "metric": "Accuracy",
            "comparison": "gte",
            "target": 0.9,
            "unit": None,
            "scope": None,
        },
    ]
