"""EdgeDevicePool: per-device exclusive lease.

Latency measurements require exclusive device access for noise-free signal.
Phase 1 implements one asyncio.Lock per ``edge_device_id``; an empty
device_id (e.g. surrogate-only run) returns a no-op lease.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Dict

from loguru import logger


class _NullLock:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class EdgeDevicePool:
    """Per-device mutex registry."""

    def __init__(self):
        self._locks: Dict[str, asyncio.Lock] = {}
        self._registry_lock = asyncio.Lock()

    async def _get_lock(self, device_id: str) -> asyncio.Lock:
        async with self._registry_lock:
            lock = self._locks.get(device_id)
            if lock is None:
                lock = asyncio.Lock()
                self._locks[device_id] = lock
            return lock

    @asynccontextmanager
    async def acquire(self, device_id: str, job_id: str):
        """Acquire exclusive lease on a device. Empty device_id = no-op."""
        if not device_id:
            async with _NullLock():
                yield
            return
        lock = await self._get_lock(device_id)
        logger.debug(f"EdgePool: job={job_id} waiting for device={device_id}")
        async with lock:
            logger.debug(f"EdgePool: job={job_id} leased device={device_id}")
            try:
                yield
            finally:
                logger.debug(f"EdgePool: job={job_id} released device={device_id}")
