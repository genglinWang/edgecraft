import asyncio
import threading
import time

from edgecraft.scheduler.gpu_pool import GpuPool
from edgecraft.scheduler.scheduler import Scheduler
from edgecraft.scheduler.shared_runtime import SharedSchedulerRuntime
from edgecraft.scheduler.edge_pool import EdgeDevicePool
from edgecraft.scheduler.types import JobResult, JobSpec, JobStatus, WorkloadEstimate


def job(job_id: str, tenant: str, seconds: float, prior: float) -> JobSpec:
    return JobSpec(
        job_id=job_id,
        tenant_id=tenant,
        estimate=WorkloadEstimate(gpu_wall_time_s=seconds, edge_wall_time_s=0),
        prior_score=prior,
    )


def test_cost_aware_order_is_tenant_fair_then_shortest_ready() -> None:
    scheduler = Scheduler(policy="cost_aware")
    ordered = scheduler._order_specs(
        [
            job("a-long", "tenant-a", 20, 0.9),
            job("b-long", "tenant-b", 30, 0.9),
            job("a-short", "tenant-a", 5, 0.1),
            job("b-short", "tenant-b", 10, 0.1),
        ]
    )
    assert [item.job_id for item in ordered] == [
        "a-short",
        "b-short",
        "a-long",
        "b-long",
    ]


def test_default_scheduler_uses_paper_cost_aware_policy() -> None:
    assert Scheduler().policy == "cost_aware"


def test_edge_only_probe_uses_edge_pool_without_training(monkeypatch) -> None:
    calls = {"train": 0, "edge": 0}

    def unexpected_train(spec, gpu_id):
        calls["train"] += 1
        raise AssertionError("an edge-only P1/P2 replay must not acquire or run training")

    def measured_edge(spec, train_result, artifact_manifest):
        calls["edge"] += 1
        assert spec.stage_label == "p1_efficiency"
        assert train_result.artifact_paths["artifact"] == "synthetic.onnx"
        assert artifact_manifest == {"schema": "synthetic"}
        train_result.status = JobStatus.SUCCESS
        train_result.stage_reached = "edge"
        train_result.edge_metrics["latency_ms"] = 3.0
        return train_result

    monkeypatch.setattr("edgecraft.scheduler.scheduler.run_train_stage", unexpected_train)
    monkeypatch.setattr("edgecraft.scheduler.scheduler.run_edge_stage", measured_edge)
    scheduler = Scheduler(policy="cost_aware")
    result = asyncio.run(
        scheduler.submit_batch(
            [
                JobSpec(
                    job_id="p1-a",
                    tenant_id="tenant-a",
                    edge_device_id="review-device",
                    edge_only=True,
                    prebuilt_artifact_paths={"artifact": "synthetic.onnx"},
                    artifact_manifest={"schema": "synthetic"},
                    stage_label="p1_efficiency",
                )
            ]
        )
    )[0]
    assert result.status == JobStatus.SUCCESS
    assert result.edge_metrics["latency_ms"] == 3.0
    assert calls == {"train": 0, "edge": 1}


def test_same_device_measurements_are_exclusive() -> None:
    async def exercise() -> int:
        pool = EdgeDevicePool()
        active = 0
        peak = 0

        async def lease(job_id: str) -> None:
            nonlocal active, peak
            async with pool.acquire("review-device", job_id):
                active += 1
                peak = max(peak, active)
                await asyncio.sleep(0.01)
                active -= 1

        await asyncio.gather(lease("one"), lease("two"))
        return peak

    assert asyncio.run(exercise()) == 1


def test_staggered_submissions_share_persistent_tenant_ready_queue(monkeypatch) -> None:
    first_started = threading.Event()
    release_first = threading.Event()
    starts: list[str] = []

    def measured_train(spec, gpu_id):
        starts.append(spec.job_id)
        if spec.job_id == "a-running":
            first_started.set()
            assert release_first.wait(timeout=2)
        return JobResult(
            job_id=spec.job_id,
            status=JobStatus.SUCCESS,
            stage_reached="train",
            gpu_id=gpu_id,
        )

    monkeypatch.setattr("edgecraft.scheduler.scheduler.run_train_stage", measured_train)

    async def exercise() -> None:
        scheduler = Scheduler(
            policy="cost_aware",
            gpu_pool=GpuPool(gpu_count=1, vram_gb_per_gpu=24),
        )
        running = asyncio.create_task(
            scheduler.submit_batch([job("a-running", "tenant-a", 30, 0.5)])
        )
        assert await asyncio.to_thread(first_started.wait, 2)
        later_b = asyncio.create_task(
            scheduler.submit_batch([job("b-short", "tenant-b", 2, 0.5)])
        )
        later_a = asyncio.create_task(
            scheduler.submit_batch([job("a-short", "tenant-a", 1, 0.5)])
        )
        await asyncio.sleep(0)
        release_first.set()
        await asyncio.gather(running, later_b, later_a)

    asyncio.run(exercise())
    assert starts == ["a-running", "b-short", "a-short"]


def test_completed_train_enqueues_edge_stage_while_next_tenant_trains(monkeypatch) -> None:
    edge_started = threading.Event()
    release_edge = threading.Event()
    second_train_started = threading.Event()

    def measured_train(spec, gpu_id):
        if spec.job_id == "second":
            second_train_started.set()
        return JobResult(
            job_id=spec.job_id,
            status=JobStatus.SUCCESS,
            stage_reached="train",
            gpu_id=gpu_id,
            artifact_paths={"artifact": "synthetic.onnx"},
        )

    def measured_edge(spec, train_result, artifact_manifest):
        edge_started.set()
        assert release_edge.wait(timeout=2)
        train_result.status = JobStatus.SUCCESS
        train_result.stage_reached = "edge"
        return train_result

    monkeypatch.setattr("edgecraft.scheduler.scheduler.run_train_stage", measured_train)
    monkeypatch.setattr("edgecraft.scheduler.scheduler.run_edge_stage", measured_edge)

    async def exercise() -> None:
        scheduler = Scheduler(
            policy="cost_aware",
            gpu_pool=GpuPool(gpu_count=1, vram_gb_per_gpu=24),
        )
        first = JobSpec(
            job_id="first",
            tenant_id="tenant-a",
            edge_device_id="review-device",
            estimate=WorkloadEstimate(gpu_wall_time_s=1, edge_wall_time_s=20),
        )
        first_task = asyncio.create_task(scheduler.submit_batch([first]))
        assert await asyncio.to_thread(edge_started.wait, 2)
        second = job("second", "tenant-b", 1, 0.5)
        second.skip_edge = True
        second_task = asyncio.create_task(scheduler.submit_batch([second]))
        assert await asyncio.to_thread(second_train_started.wait, 2)
        release_edge.set()
        await asyncio.gather(first_task, second_task)

    asyncio.run(exercise())


def test_shared_runtime_forwards_staggered_thread_submissions_to_one_queue(
    monkeypatch,
) -> None:
    monkeypatch.setenv("EDGECRAFT_GPU_COUNT", "1")
    first_started = threading.Event()
    release_first = threading.Event()
    starts: list[str] = []
    errors: list[Exception] = []

    def measured_train(spec, gpu_id):
        starts.append(spec.job_id)
        if spec.job_id == "a-running":
            first_started.set()
            assert release_first.wait(timeout=2)
        return JobResult(
            job_id=spec.job_id,
            status=JobStatus.SUCCESS,
            stage_reached="train",
            gpu_id=gpu_id,
        )

    monkeypatch.setattr("edgecraft.scheduler.scheduler.run_train_stage", measured_train)
    runtime = SharedSchedulerRuntime()

    def submit(spec: JobSpec) -> None:
        try:
            runtime.submit_batch([spec])
        except Exception as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    first_spec = job("a-running", "tenant-a", 30, 0.5)
    first_spec.skip_edge = True
    first = threading.Thread(target=submit, args=(first_spec,))
    first.start()
    assert first_started.wait(timeout=2)

    second_spec = job("b-short", "tenant-b", 2, 0.5)
    second_spec.skip_edge = True
    second = threading.Thread(target=submit, args=(second_spec,))
    second.start()
    deadline = time.monotonic() + 2
    while runtime.scheduler._sequence < 2 and time.monotonic() < deadline:
        time.sleep(0.001)
    assert runtime.scheduler._sequence >= 2

    third_spec = job("a-short", "tenant-a", 1, 0.5)
    third_spec.skip_edge = True
    third = threading.Thread(target=submit, args=(third_spec,))
    third.start()
    deadline = time.monotonic() + 2
    while runtime.scheduler._sequence < 3 and time.monotonic() < deadline:
        time.sleep(0.001)
    assert runtime.scheduler._sequence >= 3

    release_first.set()
    for thread in (first, second, third):
        thread.join(timeout=2)
        assert not thread.is_alive()
    runtime.close()

    assert errors == []
    assert starts == ["a-running", "b-short", "a-short"]
