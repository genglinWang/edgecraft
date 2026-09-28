#!/usr/bin/env python3
"""Controller runner for constructing an initialized P1 artifact.

The candidate's efficiency branch runs in this process.  The runner blocks
child-process escape and intercepts common optimizer, gradient, and fit entry
points.  Its final JSON line binds the outcome to both source files so the
verifier can grant P1 authority from controller evidence rather than a
candidate declaration alone.
"""
from __future__ import annotations

import argparse
import builtins
from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import threading
import traceback
from typing import Any, Callable


SCHEMA_VERSION = "edgecraft_p1_guard_v1"
PRODUCER = "edgecraft-controller"
FORBIDDEN_PYTHON_CALLS = {
    "apply_gradients",
    "backward",
    "fit",
    "fit_generator",
    "minimize",
    "optimizer_step",
    "partial_fit",
    "train_on_batch",
    "train_step",
    "training_step",
}


class P1TrainingMutation(RuntimeError):
    """Raised when the initialized-artifact path invokes training work."""


class _Tee(io.TextIOBase):
    def __init__(self, visible: Any, captured: io.StringIO) -> None:
        self.visible = visible
        self.captured = captured

    def write(self, text: str) -> int:
        self.visible.write(text)
        self.captured.write(text)
        return len(text)

    def flush(self) -> None:
        self.visible.flush()
        self.captured.flush()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.visible, name)


class _Guard:
    def __init__(self) -> None:
        self.mutations: list[str] = []
        self.hooks: set[str] = {"python-profile", "process-spawn"}
        self._patched: set[tuple[int, str]] = set()

    def block(self, label: str) -> None:
        self.mutations.append(label)
        raise P1TrainingMutation(f"P1 construction blocked training call: {label}")

    def blocked_call(self, label: str) -> Callable[..., Any]:
        def _blocked(*_args: Any, **_kwargs: Any) -> Any:
            self.block(label)

        setattr(_blocked, "__edgecraft_p1_guard__", True)
        return _blocked

    def patch_method(self, owner: Any, name: str, label: str) -> None:
        key = (id(owner), name)
        if key in self._patched or not hasattr(owner, name):
            return
        try:
            current = getattr(owner, name)
            if getattr(current, "__edgecraft_p1_guard__", False):
                self._patched.add(key)
                return
            setattr(owner, name, self.blocked_call(label))
        except (AttributeError, TypeError):
            return
        self._patched.add(key)
        self.hooks.add(label)

    def patch_loaded_frameworks(self) -> None:
        torch = sys.modules.get("torch")
        if torch is not None:
            tensor = getattr(torch, "Tensor", None)
            if tensor is not None:
                self.patch_method(tensor, "backward", "torch.Tensor.backward")
            autograd = getattr(torch, "autograd", None)
            if autograd is not None:
                self.patch_method(autograd, "backward", "torch.autograd.backward")
            optim = getattr(torch, "optim", None) or sys.modules.get("torch.optim")
            base = getattr(optim, "Optimizer", None) if optim is not None else None
            if optim is not None and isinstance(base, type):
                for value in vars(optim).values():
                    if not isinstance(value, type):
                        continue
                    try:
                        is_optimizer = issubclass(value, base)
                    except TypeError:
                        is_optimizer = False
                    if is_optimizer:
                        self.patch_method(
                            value,
                            "step",
                            f"torch.optim.{value.__name__}.step",
                        )

        tensorflow = sys.modules.get("tensorflow")
        if tensorflow is not None:
            tape = getattr(tensorflow, "GradientTape", None)
            if tape is not None:
                self.patch_method(tape, "gradient", "tensorflow.GradientTape.gradient")
            keras = getattr(tensorflow, "keras", None)
            model = getattr(keras, "Model", None) if keras is not None else None
            if model is not None:
                self.patch_method(model, "fit", "tensorflow.keras.Model.fit")
                self.patch_method(
                    model,
                    "train_on_batch",
                    "tensorflow.keras.Model.train_on_batch",
                )
            optimizers = getattr(keras, "optimizers", None) if keras is not None else None
            optimizer = getattr(optimizers, "Optimizer", None) if optimizers is not None else None
            if optimizer is not None:
                self.patch_method(
                    optimizer,
                    "apply_gradients",
                    "tensorflow.keras.Optimizer.apply_gradients",
                )
                self.patch_method(
                    optimizer,
                    "minimize",
                    "tensorflow.keras.Optimizer.minimize",
                )

        jax = sys.modules.get("jax")
        if jax is not None:
            self.patch_method(jax, "grad", "jax.grad")
            self.patch_method(jax, "value_and_grad", "jax.value_and_grad")

        optax = sys.modules.get("optax")
        if optax is not None:
            self.patch_method(optax, "apply_updates", "optax.apply_updates")

        transformers_trainer = sys.modules.get("transformers.trainer")
        if transformers_trainer is not None:
            trainer = getattr(transformers_trainer, "Trainer", None)
            if trainer is not None:
                self.patch_method(trainer, "train", "transformers.Trainer.train")

        accelerate_module = sys.modules.get("accelerate.accelerator")
        if accelerate_module is not None:
            accelerator = getattr(accelerate_module, "Accelerator", None)
            if accelerator is not None:
                self.patch_method(
                    accelerator,
                    "backward",
                    "accelerate.Accelerator.backward",
                )

        lightning_trainer = sys.modules.get(
            "lightning.pytorch.trainer.trainer"
        ) or sys.modules.get("pytorch_lightning.trainer.trainer")
        if lightning_trainer is not None:
            trainer = getattr(lightning_trainer, "Trainer", None)
            if trainer is not None:
                self.patch_method(trainer, "fit", "lightning.Trainer.fit")

    def profile(self, frame: Any, event: str, _arg: Any) -> Any:
        if event == "call":
            name = str(frame.f_code.co_name or "").lower()
            if name in FORBIDDEN_PYTHON_CALLS:
                module = str(frame.f_globals.get("__name__") or "candidate")
                self.block(f"python:{module}.{name}")
        return self.profile


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _last_json(text: str) -> dict[str, Any]:
    for line in reversed(text.splitlines()):
        try:
            value = json.loads(line.strip())
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(value, dict):
            return value
    return {}


def _block_process_spawns(guard: _Guard) -> None:
    for name in (
        "Popen",
        "run",
        "call",
        "check_call",
        "check_output",
    ):
        guard.patch_method(subprocess, name, f"subprocess.{name}")
    for name in (
        "fork",
        "forkpty",
        "execl",
        "execle",
        "execlp",
        "execlpe",
        "execv",
        "execve",
        "execvp",
        "execvpe",
        "posix_spawn",
        "posix_spawnp",
        "spawnl",
        "spawnle",
        "spawnlp",
        "spawnlpe",
        "spawnv",
        "spawnve",
        "spawnvp",
        "spawnvpe",
        "system",
    ):
        guard.patch_method(os, name, f"os.{name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", required=True)
    parser.add_argument("candidate_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    candidate = Path(args.script).resolve(strict=True)
    runner = Path(__file__).resolve()
    candidate_source_sha256 = _sha256(candidate)
    guard_source_sha256 = _sha256(runner)
    candidate_args = list(args.candidate_args)
    if candidate_args[:1] == ["--"]:
        candidate_args = candidate_args[1:]

    guard = _Guard()
    _block_process_spawns(guard)
    original_import = builtins.__import__

    def guarded_import(*import_args: Any, **import_kwargs: Any) -> Any:
        imported = original_import(*import_args, **import_kwargs)
        guard.patch_loaded_frameworks()
        return imported

    captured = io.StringIO()
    candidate_error = ""
    candidate_exit_code = 0
    original_sys_setprofile = sys.setprofile
    original_threading_setprofile = threading.setprofile
    builtins.__import__ = guarded_import
    original_sys_setprofile(guard.profile)
    original_threading_setprofile(guard.profile)
    sys.setprofile = guard.blocked_call("sys.setprofile")
    threading.setprofile = guard.blocked_call("threading.setprofile")
    try:
        sys.argv = [str(candidate), *candidate_args]
        with redirect_stdout(_Tee(sys.__stdout__, captured)):
            try:
                runpy.run_path(str(candidate), run_name="__main__")
            except SystemExit as exc:
                candidate_exit_code = int(exc.code or 0) if isinstance(exc.code, int) else 1
                if candidate_exit_code:
                    candidate_error = f"candidate exited with code {candidate_exit_code}"
            except P1TrainingMutation as exc:
                candidate_error = str(exc)
            except Exception as exc:  # noqa: BLE001
                candidate_error = str(exc)
                traceback.print_exc(file=sys.stderr)
    finally:
        original_sys_setprofile(None)
        original_threading_setprofile(None)
        sys.setprofile = original_sys_setprofile
        threading.setprofile = original_threading_setprofile
        builtins.__import__ = original_import

    payload = _last_json(captured.getvalue())
    if not payload:
        payload = {
            "status": "error",
            "error": candidate_error or "candidate did not emit a final JSON object",
        }
    candidate_source_unchanged = _sha256(candidate) == candidate_source_sha256
    guard_source_unchanged = _sha256(runner) == guard_source_sha256
    passed = bool(
        not candidate_error
        and not guard.mutations
        and candidate_exit_code == 0
        and payload.get("status") == "success"
        and candidate_source_unchanged
        and guard_source_unchanged
    )
    if not passed:
        payload["status"] = "error"
        payload["error"] = candidate_error or "controller P1 construction guard failed"
    payload["p1_guard_attestation"] = {
        "schema_version": SCHEMA_VERSION,
        "producer": PRODUCER,
        "status": "passed" if passed else "blocked",
        "candidate_source_sha256": candidate_source_sha256,
        "guard_source_sha256": guard_source_sha256,
        "candidate_source_unchanged": candidate_source_unchanged,
        "guard_source_unchanged": guard_source_unchanged,
        "process_spawn_policy": "blocked",
        "observed_training_mutations": list(guard.mutations),
        "training_mutation_count": len(guard.mutations),
        "installed_hooks": sorted(guard.hooks),
        "candidate_exit_code": candidate_exit_code,
    }
    print(json.dumps(payload, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
