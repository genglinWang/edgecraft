from __future__ import annotations

import asyncio

import httpx

from edgecraft.api import server
from edgecraft.api.schemas import TaskResponse
from edgecraft.api.task_manager import _TaskRecord
from edgecraft.agent.workspace.manager import tenant_workspace_key


class _FakeTaskManager:
    def __init__(self) -> None:
        self.created = None
        self.records = {
            "task-a": _TaskRecord(task_id="task-a", request={}, tenant_id="tenant-a"),
            "task-b": _TaskRecord(task_id="task-b", request={}, tenant_id="tenant-b"),
        }

    def create_task(self, req):
        self.created = req
        record = _TaskRecord(
            task_id="task-created",
            request=req.model_dump(mode="json"),
            tenant_id=req.tenant_id,
        )
        self.records[record.task_id] = record
        return record

    def shutdown(self) -> None:
        pass

    def get_task(self, task_id: str):
        return self.records.get(task_id)

    def list_tasks(self):
        return list(self.records.values())

    def list_tasks_for_tenant(self, tenant_id: str):
        return [item for item in self.records.values() if item.tenant_id == tenant_id]

    def list_tenants(self):
        return [
            {"tenant_id": "tenant-a", "task_count": 1, "active_tasks": 0, "run_ids": []},
            {"tenant_id": "tenant-b", "task_count": 1, "active_tasks": 0, "run_ids": []},
        ]

    def to_response(self, record: _TaskRecord) -> TaskResponse:
        return TaskResponse(
            task_id=record.task_id,
            status=record.status,
            created_at=record.created_at,
            updated_at=record.updated_at,
            tenant_id=record.tenant_id,
            request=record.request,
        )


async def _request(app, method: str, path: str, **kwargs):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://review.local",
    ) as client:
        return await client.request(method, path, **kwargs)


def test_authenticated_routes_hide_other_tenants_and_block_direct_paths(monkeypatch) -> None:
    manager = _FakeTaskManager()
    monkeypatch.setattr(server, "get_task_manager", lambda: manager)
    monkeypatch.setattr(
        server.settings,
        "API_TENANT_TOKENS",
        '{"0123456789abcdef":"tenant-a","fedcba9876543210":"tenant-b"}',
    )
    headers = {"Authorization": "Bearer 0123456789abcdef"}
    app = server.create_app()

    assert asyncio.run(_request(app, "GET", "/tasks")).status_code == 401
    assert asyncio.run(
        _request(
            app,
            "GET",
            "/tasks",
            headers={"Authorization": "Bearer 0000000000000000"},
        )
    ).status_code == 401

    visible = asyncio.run(_request(app, "GET", "/tasks", headers=headers))
    assert visible.status_code == 200
    assert [item["task_id"] for item in visible.json()] == ["task-a"]

    assert asyncio.run(
        _request(app, "GET", "/tasks/task-b", headers=headers)
    ).status_code == 404
    assert asyncio.run(
        _request(app, "GET", "/tenants/tenant-b/tasks", headers=headers)
    ).status_code == 404
    tenants = asyncio.run(_request(app, "GET", "/tenants", headers=headers))
    assert [item["tenant_id"] for item in tenants.json()] == ["tenant-a"]

    direct = asyncio.run(
        _request(
            app,
            "POST",
            "/deploy",
            headers=headers,
            json={
                "artifact_path": "/tmp/unowned.onnx",
                "device_id": "review-device",
                "device_ip": "reviewer@device",
                "ssh_key_path": "/review/key",
            },
        )
    )
    assert direct.status_code == 403


def test_authenticated_task_submission_cannot_choose_another_tenant(monkeypatch) -> None:
    manager = _FakeTaskManager()
    monkeypatch.setattr(server, "get_task_manager", lambda: manager)
    monkeypatch.setattr(
        server.settings,
        "API_TENANT_TOKENS",
        '{"0123456789abcdef":"tenant-a"}',
    )

    response = asyncio.run(
        _request(
            server.create_app(),
            "POST",
            "/tasks",
            headers={"Authorization": "Bearer 0123456789abcdef"},
            json={
                "intent": "review request",
                "dataset_path": "/review/data",
                "device_ip": "reviewer@device",
                "ssh_key_path": "/review/key",
                "tenant_id": "tenant-b",
            },
        )
    )
    assert response.status_code == 403


def test_authenticated_task_rejects_raw_server_paths(monkeypatch) -> None:
    manager = _FakeTaskManager()
    monkeypatch.setattr(server, "get_task_manager", lambda: manager)
    monkeypatch.setattr(
        server.settings,
        "API_TENANT_TOKENS",
        '{"0123456789abcdef":"tenant-a"}',
    )
    response = asyncio.run(
        _request(
            server.create_app(),
            "POST",
            "/tasks",
            headers={"Authorization": "Bearer 0123456789abcdef"},
            json={
                "intent": "review request",
                "dataset_path": "/server/other-tenant/data",
                "device_ip": "reviewer@device",
                "ssh_key_path": "/server/other-tenant/key",
                "tenant_id": "tenant-a",
            },
        )
    )
    assert response.status_code == 403
    assert manager.created is None


def test_authenticated_task_resolves_only_own_resource_handles(tmp_path, monkeypatch) -> None:
    manager = _FakeTaskManager()
    monkeypatch.setattr(server, "get_task_manager", lambda: manager)
    monkeypatch.setattr(
        server.settings,
        "API_TENANT_TOKENS",
        '{"0123456789abcdef":"tenant-a"}',
    )
    data_root = tmp_path / "data"
    key_root = tmp_path / "credentials"
    tenant_key = tenant_workspace_key("tenant-a")
    dataset = data_root / tenant_key / "review-dataset"
    dataset.mkdir(parents=True)
    credential = key_root / tenant_key / "device-key"
    credential.parent.mkdir(parents=True)
    credential.write_text("synthetic-test-key", encoding="utf-8")
    credential.chmod(0o600)
    monkeypatch.setattr(server.settings, "TENANT_DATA_ROOT", str(data_root))
    monkeypatch.setattr(server.settings, "TENANT_CREDENTIAL_ROOT", str(key_root))

    response = asyncio.run(
        _request(
            server.create_app(),
            "POST",
            "/tasks",
            headers={"Authorization": "Bearer 0123456789abcdef"},
            json={
                "intent": "review request",
                "dataset_handle": "review-dataset",
                "device_ip": "reviewer@device",
                "credential_handle": "device-key",
            },
        )
    )
    assert response.status_code == 200
    assert manager.created.tenant_id == "tenant-a"
    assert manager.created.dataset_path == str(dataset.resolve())
    assert manager.created.ssh_key_path == str(credential.resolve())

    other_dataset = data_root / tenant_workspace_key("tenant-b") / "private"
    other_dataset.mkdir(parents=True)
    cross_tenant = dataset.parent / "escape"
    cross_tenant.symlink_to(other_dataset, target_is_directory=True)
    rejected = asyncio.run(
        _request(
            server.create_app(),
            "POST",
            "/tasks",
            headers={"Authorization": "Bearer 0123456789abcdef"},
            json={
                "intent": "review request",
                "dataset_handle": "escape",
                "device_ip": "reviewer@device",
                "credential_handle": "device-key",
            },
        )
    )
    assert rejected.status_code == 404

    other_credential = key_root / tenant_workspace_key("tenant-b") / "private-key"
    other_credential.parent.mkdir(parents=True)
    other_credential.write_text("synthetic-other-key", encoding="utf-8")
    other_credential.chmod(0o600)
    credential_escape = credential.parent / "escape-key"
    credential_escape.symlink_to(other_credential)
    rejected_credential = asyncio.run(
        _request(
            server.create_app(),
            "POST",
            "/tasks",
            headers={"Authorization": "Bearer 0123456789abcdef"},
            json={
                "intent": "review request",
                "dataset_handle": "review-dataset",
                "device_ip": "reviewer@device",
                "credential_handle": "escape-key",
            },
        )
    )
    assert rejected_credential.status_code == 404


def test_api_rejects_ssh_option_injection_before_task_creation(monkeypatch) -> None:
    manager = _FakeTaskManager()
    monkeypatch.setattr(server, "get_task_manager", lambda: manager)
    monkeypatch.setattr(server.settings, "API_TENANT_TOKENS", "")
    response = asyncio.run(
        _request(
            server.create_app(),
            "POST",
            "/tasks",
            json={
                "intent": "review request",
                "dataset_path": "/review/data",
                "device_ip": "-oProxyCommand=unexpected",
                "ssh_key_path": "/review/key",
            },
        )
    )
    assert response.status_code == 422
