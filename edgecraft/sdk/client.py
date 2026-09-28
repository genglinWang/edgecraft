"""EdgeCraft Python SDK client."""
from pathlib import Path
import time
from typing import Dict, Any, Optional, Iterator
import requests


class EdgeCraftTask:
    """Represents an EdgeCraft synthesis task."""

    def __init__(self, client: "EdgeCraftClient", task_id: str, data: Dict[str, Any]):
        self._client = client
        self.task_id = task_id
        self._data = data

    @property
    def status(self) -> str:
        """Get task status."""
        return self._data.get("status", "unknown")

    @property
    def modality(self) -> Optional[str]:
        """Get detected modality."""
        return self._data.get("modality")

    @property
    def task_type(self) -> Optional[str]:
        """Get detected task type."""
        return self._data.get("task_type")

    @property
    def artifacts(self) -> list:
        """Get list of artifacts."""
        manifest = self._data.get("artifact_manifest") or {}
        artifacts = manifest.get("artifacts") or {}
        if artifacts:
            return [item.get("path") for item in artifacts.values() if item.get("path")]
        return list(((self._data.get("result") or {}).get("best_trial") or {}).get("artifact_paths", {}).values())

    @property
    def plan(self) -> list:
        """Get execution plan."""
        return []

    @property
    def error(self) -> Optional[str]:
        """Get error message if any."""
        return self._data.get("error")

    def refresh(self) -> "EdgeCraftTask":
        """Refresh task data from server."""
        self._data = self._client._get(f"/tasks/{self.task_id}")
        return self

    def tree(self) -> Dict[str, Any]:
        """Return the server-side run/tree lifecycle summary."""
        return self._client.get_tree(self.task_id)

    def artifact_manifest(self) -> Dict[str, Any]:
        """Return the best-trial artifact manifest."""
        return self._client.get_artifacts(self.task_id)

    def download_artifact(self, key: str, output_path: str) -> str:
        """Download one artifact by manifest key."""
        return self._client.download_artifact(self.task_id, key, output_path)

    def wait(self, poll_interval: float = 2.0, timeout: float = 600) -> "EdgeCraftTask":
        """Wait for task to complete.

        Args:
            poll_interval: Seconds between status checks.
            timeout: Maximum seconds to wait.

        Returns:
            Updated task object.

        Raises:
            TimeoutError: If task doesn't complete within timeout.
        """
        start = time.time()

        while time.time() - start < timeout:
            self.refresh()

            if self.status in ("completed", "completed_infeasible", "failed", "cancelled"):
                return self

            time.sleep(poll_interval)

        raise TimeoutError(f"Task {self.task_id} did not complete within {timeout}s")

    def stream_status(self, poll_interval: float = 2.0) -> Iterator[Dict[str, Any]]:
        """Stream task status updates.

        Yields:
            Status updates as dicts.
        """
        last_status = None

        while True:
            status = self._client._get(f"/tasks/{self.task_id}")

            if status.get("status") != last_status:
                last_status = status.get("status")
                yield status

            if status.get("status") in (
                "completed",
                "completed_infeasible",
                "failed",
                "cancelled",
            ):
                break

            time.sleep(poll_interval)

    def deploy(self, device_ip: str, ssh_key_path: str = None) -> Dict[str, Any]:
        """Deploy the synthesized model to an edge device.

        Args:
            device_ip: Target device IP (user@host format).
            ssh_key_path: Path to SSH private key.

        Returns:
            Deployment results.
        """
        if not self.artifacts:
            raise ValueError("No artifacts available for deployment")
        if not ssh_key_path:
            raise ValueError("ssh_key_path is required for explicit SSH identity selection")

        return self._client.deploy(
            artifact_path=self.artifacts[0],
            device_id=self.task_id,
            device_ip=device_ip,
            ssh_key_path=ssh_key_path
        )


class EdgeCraftClient:
    """Client for the EdgeCraft API."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: str = None
    ):
        """Initialize the client.

        Args:
            base_url: EdgeCraft API base URL.
            api_key: Optional API key for authentication.
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._session = requests.Session()

        if api_key:
            self._session.headers["Authorization"] = f"Bearer {api_key}"

    def _get(self, path: str) -> Dict[str, Any]:
        """Make GET request."""
        response = self._session.get(f"{self.base_url}{path}")
        response.raise_for_status()
        return response.json()

    def _post(self, path: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make POST request."""
        response = self._session.post(f"{self.base_url}{path}", json=data)
        response.raise_for_status()
        return response.json()

    def _delete(self, path: str) -> Dict[str, Any]:
        """Make DELETE request."""
        response = self._session.delete(f"{self.base_url}{path}")
        response.raise_for_status()
        return response.json()

    def synthesize(
        self,
        intent: str,
        dataset_path: str = None,
        target_device: str = "jetson_orin_agx",
        device_ip: Optional[str] = None,
        ssh_key_path: Optional[str] = None,
        docker_image: Optional[str] = None,
        constraints: Dict[str, Any] = None,
        max_iterations: int = 24,
        branching_factor: Optional[int] = None,
        tenant_id: str = "default",
        run_id: Optional[str] = None,
        dataset_handle: Optional[str] = None,
        credential_handle: Optional[str] = None,
        profile: str = "paper",
    ) -> EdgeCraftTask:
        """Create a model synthesis task.

        Args:
            intent: Natural language description of the task.
            dataset_path: Local-operator dataset path. Authenticated clients use
                ``dataset_handle`` instead.
            target_device: Target edge device.
            device_ip: Explicit endpoint for the target device.
            ssh_key_path: Local-operator credential path. Authenticated clients
                use ``credential_handle`` instead.
            docker_image: Optional execution image override.
            constraints: Mapping from metric names to targets or structured
                ``target``/``comparison`` dictionaries.
            max_iterations: Maximum agent iterations.
            branching_factor: Optional constraint-aware tree fan-out.
            tenant_id: Requested tenant in local mode; authenticated service
                mode binds the effective tenant to the bearer token.
            run_id: Optional caller-supplied run identifier.
            dataset_handle: Tenant-relative dataset handle for authenticated mode.
            credential_handle: Tenant-relative credential handle for
                authenticated mode.
            profile: Named execution profile; must match the service profile.

        Returns:
            EdgeCraftTask object for tracking the synthesis.

        Example:
            >>> client = EdgeCraftClient()
            >>> task = client.synthesize(
            ...     intent="Detect defects on industrial products",
            ...     dataset_path="./defect_images/",
            ...     target_device="jetson_orin_agx",
            ...     constraints={"latency_ms": 50, "accuracy": 0.9}
            ... )
            >>> task.wait()
            >>> print(task.artifacts)
        """
        structured_constraints = []
        for metric, raw in (constraints or {}).items():
            if isinstance(raw, dict):
                if "target" not in raw:
                    raise ValueError(f"constraint {metric!r} requires a target")
                item = {
                    "metric": metric,
                    "comparison": raw.get("comparison", "gte"),
                    "target": raw["target"],
                    "unit": raw.get("unit"),
                    "scope": raw.get("scope"),
                }
            else:
                metric_lower = str(metric).lower()
                comparison = (
                    "lte"
                    if any(
                        token in metric_lower
                        for token in ("latency", "memory", "energy", "power", "size")
                    )
                    else "gte"
                )
                item = {
                    "metric": metric,
                    "comparison": comparison,
                    "target": raw,
                }
            structured_constraints.append(item)
        if not device_ip:
            raise ValueError("device_ip is required by the current EdgeCraft API")
        if not ssh_key_path and not credential_handle:
            raise ValueError(
                "ssh_key_path (local mode) or credential_handle (authenticated mode) is required"
            )
        if not dataset_path and not dataset_handle:
            raise ValueError(
                "dataset_path (local mode) or dataset_handle (authenticated mode) is required"
            )
        payload = {
            "intent": intent,
            "dataset_path": dataset_path,
            "dataset_handle": dataset_handle,
            "device_id": target_device,
            "device_ip": device_ip,
            "ssh_key_path": ssh_key_path,
            "credential_handle": credential_handle,
            "docker_image": docker_image,
            "constraints": structured_constraints,
            "iterations": max_iterations,
            "branching_factor": branching_factor,
            "tenant_id": tenant_id,
            "run_id": run_id,
            "profile": profile,
        }
        data = self._post(
            "/tasks",
            {key: value for key, value in payload.items() if value is not None},
        )

        return EdgeCraftTask(self, data["task_id"], data)

    def get_task(self, task_id: str) -> EdgeCraftTask:
        """Get an existing task.

        Args:
            task_id: Task identifier.

        Returns:
            EdgeCraftTask object.
        """
        data = self._get(f"/tasks/{task_id}")
        return EdgeCraftTask(self, task_id, data)

    def get_tree(self, task_id: str) -> Dict[str, Any]:
        """Get one task's run/tree lifecycle summary."""
        return self._get(f"/tasks/{task_id}/tree")

    def get_artifacts(self, task_id: str) -> Dict[str, Any]:
        """Get one task's best-trial artifact manifest."""
        return self._get(f"/tasks/{task_id}/artifacts")

    def download_artifact(self, task_id: str, key: str, output_path: str) -> str:
        """Download a managed artifact file from serve mode."""
        response = self._session.get(f"{self.base_url}/tasks/{task_id}/artifacts/{key}", stream=True)
        response.raise_for_status()
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("wb") as fh:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    fh.write(chunk)
        return str(output)

    def list_tenants(self) -> list:
        """List tenants known to the serve process."""
        return self._get("/tenants")

    def list_tenant_tasks(self, tenant_id: str) -> list:
        """List tasks owned by one tenant."""
        return self._get(f"/tenants/{tenant_id}/tasks")

    def cancel_task(self, task_id: str) -> Dict[str, Any]:
        """Cancel a running task.

        Args:
            task_id: Task identifier.

        Returns:
            Cancellation result.
        """
        return self._delete(f"/tasks/{task_id}")

    def list_devices(self) -> list:
        """List available edge devices.

        Returns:
            List of device info dicts.
        """
        return self._get("/devices")

    def deploy(
        self,
        artifact_path: str,
        device_id: str,
        device_ip: str,
        ssh_key_path: str = None
    ) -> Dict[str, Any]:
        """Deploy a model to an edge device.

        Args:
            artifact_path: Path to model artifact.
            device_id: Device identifier.
            device_ip: Device IP (user@host format).
            ssh_key_path: Path to SSH private key.

        Returns:
            Deployment result.
        """
        if not ssh_key_path:
            raise ValueError("ssh_key_path is required for explicit SSH identity selection")
        return self._post("/deploy", {
            "artifact_path": artifact_path,
            "device_id": device_id,
            "device_ip": device_ip,
            "ssh_key_path": ssh_key_path
        })

    def benchmark(
        self,
        model_path: str,
        device_id: str,
        device_ip: str,
        ssh_key_path: str = None,
        num_iterations: int = 100,
        warmup: int = 10
    ) -> Dict[str, Any]:
        """Run inference benchmark on edge device.

        Args:
            model_path: Path to model.
            device_id: Device identifier.
            device_ip: Device IP.
            ssh_key_path: SSH key path.
            num_iterations: Number of inference iterations.
            warmup: Warmup iterations.

        Returns:
            Benchmark results.
        """
        if not ssh_key_path:
            raise ValueError("ssh_key_path is required for explicit SSH identity selection")
        return self._post("/benchmark", {
            "model_path": model_path,
            "device_id": device_id,
            "device_ip": device_ip,
            "ssh_key_path": ssh_key_path,
            "num_iterations": num_iterations,
            "warmup": warmup
        })

    def health(self) -> Dict[str, Any]:
        """Check API health.

        Returns:
            Health status.
        """
        return self._get("/health")
