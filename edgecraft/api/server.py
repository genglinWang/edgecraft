"""FastAPI server for EdgeCraft (MCaaS prototype)."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse

from edgecraft.api.schemas import (
    BenchmarkRequest,
    CreateTaskRequest,
    DeployRequest,
    TaskResponse,
)
from edgecraft.api.task_manager import get_task_manager
from edgecraft.api.auth import TenantAuthenticationError, authenticate_tenant
from edgecraft.api.resources import (
    TenantResourceError,
    resolve_credential_handle,
    resolve_dataset_handle,
)
from edgecraft.config.settings import settings


def create_app() -> FastAPI:
    """Create and configure the EdgeCraft API application."""

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            get_task_manager().shutdown()

    app = FastAPI(
        title="EdgeCraft API",
        version="0.1.0",
        description="MCaaS prototype API for asynchronous synthesis tasks.",
        lifespan=lifespan,
    )

    def current_tenant(
        authorization: Optional[str] = Header(default=None),
    ) -> Optional[str]:
        try:
            return authenticate_tenant(
                authorization,
                token_config=settings.API_TENANT_TOKENS,
            )
        except TenantAuthenticationError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc

    def owned_task(task_id: str, tenant_id: Optional[str]):
        record = get_task_manager().get_task(task_id)
        if record is None or (tenant_id is not None and record.tenant_id != tenant_id):
            # Deliberately do not reveal whether another tenant owns the ID.
            raise HTTPException(status_code=404, detail="task not found")
        return record

    def require_local_operator_mode(tenant_id: Optional[str]) -> None:
        """Keep arbitrary filesystem paths out of token-authenticated requests."""
        if tenant_id is not None:
            raise HTTPException(
                status_code=403,
                detail=(
                    "direct path-based deployment is disabled in multi-tenant mode; "
                    "use an artifact owned by a synthesis task"
                ),
            )

    def authorized_synthesis_request(
        req: CreateTaskRequest,
        tenant_id: Optional[str],
    ) -> CreateTaskRequest:
        """Bind authenticated requests to server-owned tenant resources."""
        if tenant_id is None:
            if not req.dataset_path or not req.ssh_key_path:
                raise HTTPException(
                    status_code=422,
                    detail="local-operator mode requires dataset_path and ssh_key_path",
                )
            return req
        if req.tenant_id not in {"default", tenant_id}:
            raise HTTPException(status_code=403, detail="tenant_id does not match bearer token")
        if req.dataset_path or req.ssh_key_path:
            raise HTTPException(
                status_code=403,
                detail=(
                    "raw dataset and credential paths are disabled in multi-tenant mode; "
                    "use tenant-owned handles"
                ),
            )
        if not req.dataset_handle or not req.credential_handle:
            raise HTTPException(
                status_code=422,
                detail="dataset_handle and credential_handle are required in multi-tenant mode",
            )
        try:
            dataset = resolve_dataset_handle(
                settings.TENANT_DATA_ROOT,
                tenant_id,
                req.dataset_handle,
            )
            credential = resolve_credential_handle(
                settings.TENANT_CREDENTIAL_ROOT,
                tenant_id,
                req.credential_handle,
            )
        except (OSError, TenantResourceError):
            # Do not reveal whether a handle exists in another namespace.
            raise HTTPException(status_code=404, detail="tenant resource not found") from None
        return req.model_copy(
            update={
                "tenant_id": tenant_id,
                "dataset_path": str(dataset),
                "ssh_key_path": str(credential),
            }
        )

    @app.get("/health")
    def health(tenant_id: Optional[str] = Depends(current_tenant)) -> Dict[str, Any]:
        mgr = get_task_manager()
        tasks = mgr.list_tasks() if tenant_id is None else mgr.list_tasks_for_tenant(tenant_id)
        active = sum(1 for t in tasks if t.status in {"preflight", "running"})
        return {
            "status": "ok",
            "service": "edgecraft-api",
            "queued_tasks": sum(1 for t in tasks if t.status == "queued"),
            "active_tasks": active,
            "total_tasks": len(tasks),
        }

    @app.post("/tasks", response_model=TaskResponse)
    def create_task(
        req: CreateTaskRequest,
        tenant_id: Optional[str] = Depends(current_tenant),
    ) -> TaskResponse:
        mgr = get_task_manager()
        if req.profile != str(settings.PROFILE or "paper").strip().lower():
            raise HTTPException(
                status_code=409,
                detail="request profile does not match the service-wide active profile",
            )
        resolved = authorized_synthesis_request(req, tenant_id)
        rec = mgr.create_task(resolved)
        return mgr.to_response(rec)

    @app.get("/tasks", response_model=List[TaskResponse])
    def list_tasks(tenant_id: Optional[str] = Depends(current_tenant)) -> List[TaskResponse]:
        mgr = get_task_manager()
        records = mgr.list_tasks() if tenant_id is None else mgr.list_tasks_for_tenant(tenant_id)
        return [mgr.to_response(t) for t in records]

    @app.get("/tasks/{task_id}", response_model=TaskResponse)
    def get_task(
        task_id: str,
        tenant_id: Optional[str] = Depends(current_tenant),
    ) -> TaskResponse:
        mgr = get_task_manager()
        rec = owned_task(task_id, tenant_id)
        return mgr.to_response(rec)

    @app.get("/tenants")
    def list_tenants(tenant_id: Optional[str] = Depends(current_tenant)) -> List[Dict[str, Any]]:
        mgr = get_task_manager()
        if tenant_id is not None:
            return [
                item for item in mgr.list_tenants() if item["tenant_id"] == tenant_id
            ]
        return mgr.list_tenants()

    @app.get("/tenants/{tenant_id}/tasks", response_model=List[TaskResponse])
    def list_tenant_tasks(
        tenant_id: str,
        authenticated_tenant: Optional[str] = Depends(current_tenant),
    ) -> List[TaskResponse]:
        mgr = get_task_manager()
        if authenticated_tenant is not None and tenant_id != authenticated_tenant:
            raise HTTPException(status_code=404, detail="tenant not found")
        return [mgr.to_response(t) for t in mgr.list_tasks_for_tenant(tenant_id)]

    @app.get("/tasks/{task_id}/tree")
    def get_task_tree(
        task_id: str,
        tenant_id: Optional[str] = Depends(current_tenant),
    ) -> Dict[str, Any]:
        mgr = get_task_manager()
        owned_task(task_id, tenant_id)
        tree = mgr.tree_lifecycle(task_id)
        if tree is None:
            raise HTTPException(status_code=404, detail="task not found")
        return tree

    @app.get("/tasks/{task_id}/artifacts")
    def get_task_artifacts(
        task_id: str,
        tenant_id: Optional[str] = Depends(current_tenant),
    ) -> Dict[str, Any]:
        mgr = get_task_manager()
        owned_task(task_id, tenant_id)
        manifest = mgr.artifact_manifest(task_id)
        if manifest is None:
            raise HTTPException(status_code=404, detail="task not found")
        return manifest

    @app.get("/tasks/{task_id}/artifacts/{artifact_key}")
    def download_task_artifact(
        task_id: str,
        artifact_key: str,
        tenant_id: Optional[str] = Depends(current_tenant),
    ) -> FileResponse:
        mgr = get_task_manager()
        owned_task(task_id, tenant_id)
        path = mgr.artifact_path(task_id, artifact_key)
        if path is None:
            raise HTTPException(status_code=404, detail="artifact not found")
        return FileResponse(str(path), filename=path.name)

    @app.delete("/tasks/{task_id}")
    def cancel_task(
        task_id: str,
        tenant_id: Optional[str] = Depends(current_tenant),
    ) -> Dict[str, Any]:
        mgr = get_task_manager()
        owned_task(task_id, tenant_id)
        ok, msg = mgr.cancel_task(task_id)
        if not ok and msg == "task not found":
            raise HTTPException(status_code=404, detail="task not found")
        return {"ok": ok, "message": msg}

    @app.get("/devices")
    def list_devices(
        _tenant_id: Optional[str] = Depends(current_tenant),
    ) -> List[Dict[str, Any]]:
        from edgecraft.models import ensure_registries_initialized
        from edgecraft.models.family_registry import DeviceRegistry

        ensure_registries_initialized()
        devices = DeviceRegistry.list_devices()
        return [
            {
                "device_id": d.device_id,
                "display_name": d.display_name,
                "device_class": d.device_class.value,
                "has_gpu": d.has_gpu,
                "memory_gb": d.memory_gb,
                "gpu_memory_gb": d.gpu_memory_gb,
                "supported_runtimes": [r.value for r in d.supported_runtimes],
            }
            for d in devices
        ]

    @app.post("/deploy")
    def deploy(
        req: DeployRequest,
        tenant_id: Optional[str] = Depends(current_tenant),
    ) -> Dict[str, Any]:
        require_local_operator_mode(tenant_id)
        from edgecraft.tools.deploy.edge_runner import EdgeRunner

        runner = EdgeRunner(ssh_key_path=req.ssh_key_path)
        return runner.deploy(
            artifact_path=req.artifact_path,
            device_id=req.device_id,
            device_ip=req.device_ip,
            ssh_key=req.ssh_key_path,
            docker_image=req.docker_image,
        )

    @app.post("/benchmark")
    def benchmark(
        req: BenchmarkRequest,
        tenant_id: Optional[str] = Depends(current_tenant),
    ) -> Dict[str, Any]:
        require_local_operator_mode(tenant_id)
        from edgecraft.tools.deploy.edge_runner import EdgeRunner

        runner = EdgeRunner(ssh_key_path=req.ssh_key_path)
        return runner.run_benchmark(
            device_id=req.device_id,
            device_ip=req.device_ip,
            model_path=req.model_path,
            ssh_key=req.ssh_key_path,
            docker_image=req.docker_image,
            timeout=req.timeout,
        )

    return app


app = create_app()
