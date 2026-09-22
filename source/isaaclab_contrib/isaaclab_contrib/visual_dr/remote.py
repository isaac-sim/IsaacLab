# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run generation in worker processes on their own GPUs.

The simulator cannot share a GPU with a diffusion model and still go fast: one
frame costs seconds, and while it runs the simulator waits. Moving generation to
dedicated GPUs removes that contention, and running several workers lets the
environments randomized in one step be generated at once -- which is the only
parallelism available while Cosmos rejects batched transfer inference.

Payloads move by CUDA IPC. Sending a CUDA tensor through a ``torch.multiprocessing``
queue shares the allocation rather than copying it, and the receiving worker's
device-to-device copy is a peer transfer, so image data never touches host memory.
Only prompts, seeds and identifiers travel as ordinary Python objects.

Same-node only. Crossing machines needs a real transport -- NIXL is the obvious
candidate -- and that fits behind this same class without the runtime noticing.
"""

from __future__ import annotations

import contextlib
import os
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import torch
import torch.multiprocessing as mp

from .backends import DRFrame, DRRequest

if TYPE_CHECKING:
    from .cfg import RemoteCosmosBackendCfg

_READY = "ready"
_FAILED = "failed"


def _worker_main(device: int, cfg, requests, responses, ready) -> None:
    """Serve generation requests on one GPU until told to stop.

    Runs without Omniverse Kit, which is incidentally why the cuDNN attention
    fallback is inert here: Kit's startup is what breaks runtime kernel
    compilation, and no Kit ever starts in this process.
    """
    try:
        torch.cuda.set_device(device)
        # Import here rather than at module scope: the parent imports this module
        # to spawn workers, and it must not pull in Cosmos to do that.
        from .cosmos import CosmosBackend

        local = cfg.copy()
        local.device = f"cuda:{device}"
        backend = CosmosBackend(local)
        ready.put((_READY, device))
    except Exception:
        ready.put((_FAILED, f"worker on cuda:{device} failed to start:\n{traceback.format_exc()}"))
        return

    while True:
        message = requests.get()
        kind = message["kind"]
        if kind == "close":
            try:
                backend.close()
            finally:
                responses.put({"ok": True})
            return
        try:
            if kind == "activate":
                backend.activate()
                responses.put({"ok": True})
            elif kind == "offload":
                backend.offload()
                responses.put({"ok": True})
            elif kind == "generate":
                frame = DRFrame(
                    message["rgb"].to(device, non_blocking=True),
                    message["depth"].to(device, non_blocking=True),
                    message["preserve"].to(device, non_blocking=True),
                    None if message["segmentation"] is None else message["segmentation"].to(device, non_blocking=True),
                )
                request = DRRequest(
                    message["seeds"].to(device, non_blocking=True), message["prompts"], message["camera"]
                )
                responses.put({"ok": True, "rgb": backend.generate(frame, request)})
            else:
                responses.put({"ok": False, "error": f"unknown request {kind!r}"})
        except Exception:
            responses.put({"ok": False, "error": traceback.format_exc()})


class _Worker:
    """One worker process and the queues that talk to it."""

    def __init__(self, device: int, cfg, context):
        self.device = device
        self.requests = context.Queue()
        self.responses = context.Queue()
        self._ready = context.Queue()
        self.process = context.Process(
            target=_worker_main,
            args=(device, cfg, self.requests, self.responses, self._ready),
            daemon=True,
        )

    def start(self) -> None:
        self.process.start()

    def await_ready(self, timeout: float) -> None:
        status, payload = self._ready.get(timeout=timeout)
        if status is _FAILED:
            raise RuntimeError(payload)

    def call(self, message: dict, timeout: float) -> dict:
        """Send one request and wait for its reply.

        Each worker has its own queue pair and is only ever used by one thread at
        a time, so replies cannot be attributed to the wrong request.
        """
        if not self.process.is_alive():
            raise RuntimeError(f"visual DR worker on cuda:{self.device} is not running")
        self.requests.put(message)
        reply = self.responses.get(timeout=timeout)
        if not reply.get("ok"):
            raise RuntimeError(f"visual DR worker on cuda:{self.device} failed:\n{reply.get('error')}")
        return reply


class RemoteCosmosBackend:
    """Fans generation out across worker processes, one per configured GPU.

    Implements the same contract as the in-process backend, so the runtime cannot
    tell the difference. ``max_batch`` should equal the number of workers: the
    runtime chunks the environments it wants randomized by that number, and this
    backend gives each worker one frame of the chunk.
    """

    def __init__(self, cfg: RemoteCosmosBackendCfg):
        if not cfg.devices:
            raise ValueError("RemoteCosmosBackendCfg.devices must name at least one GPU")
        visible = torch.cuda.device_count()
        for device in cfg.devices:
            if device >= visible:
                raise ValueError(
                    f"device cuda:{device} is not visible ({visible} GPUs). Every worker GPU and the "
                    "simulator's own GPU must be visible to this process; narrow CUDA_VISIBLE_DEVICES "
                    "to the union rather than to one device."
                )
        self.cfg = cfg
        self._workers: list[_Worker] = []
        self._pool: ThreadPoolExecutor | None = None
        self._active = False
        self._closed = False

    # -- residency ---------------------------------------------------------

    def activate(self) -> None:
        if self._closed:
            raise RuntimeError("Remote visual DR backend is closed")
        if not self._workers:
            self._start_workers()
        if not self._active:
            self._broadcast({"kind": "activate"}, self.cfg.startup_timeout_s)
            self._active = True

    def _start_workers(self) -> None:
        # Spawn rather than fork: the parent holds a live CUDA context (and an
        # Omniverse Kit instance), neither of which survives being forked.
        context = mp.get_context("spawn")
        self._workers = [_Worker(device, self.cfg, context) for device in self.cfg.devices]
        for worker in self._workers:
            worker.start()
        try:
            for worker in self._workers:
                worker.await_ready(self.cfg.startup_timeout_s)
        except Exception:
            self.close()
            raise
        self._pool = ThreadPoolExecutor(max_workers=len(self._workers), thread_name_prefix="visual_dr")

    def offload(self) -> None:
        """Release model memory on every worker; the processes stay up."""
        if self._active:
            self._active = False
            self._broadcast({"kind": "offload"}, self.cfg.request_timeout_s)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._active = False
        for worker in self._workers:
            if worker.process.is_alive():
                # A worker that is already wedged still gets terminated below, and
                # shutdown must not mask whatever failure brought us here.
                with contextlib.suppress(Exception):
                    worker.call({"kind": "close"}, self.cfg.request_timeout_s)
        for worker in self._workers:
            worker.process.join(timeout=30)
            if worker.process.is_alive():
                worker.process.terminate()
        self._workers = []
        if self._pool is not None:
            self._pool.shutdown(wait=False)
            self._pool = None

    def _broadcast(self, message: dict, timeout: float) -> None:
        """Send one message to every worker at once and wait for all replies."""
        if self._pool is None:
            for worker in self._workers:
                worker.call(message, timeout)
            return
        for future in [self._pool.submit(w.call, message, timeout) for w in self._workers]:
            future.result()

    # -- generation --------------------------------------------------------

    @torch.no_grad()
    def generate(self, frame: DRFrame, request: DRRequest) -> torch.Tensor:
        """Generate one frame per worker, concurrently, and reassemble in order."""
        if not self._active:
            raise RuntimeError("Activate the remote visual DR backend before generating")
        count = frame.num_envs
        if count > len(self._workers):
            raise ValueError(
                f"batch of {count} exceeds {len(self._workers)} workers; set max_batch to the worker count "
                "so the runtime chunks to fit"
            )

        def one(index: int) -> torch.Tensor:
            single = frame.index(torch.tensor([index], device=frame.rgb.device))
            reply = self._workers[index].call(
                {
                    "kind": "generate",
                    # Sent as CUDA tensors: torch shares the allocation by IPC handle
                    # rather than serializing pixels through the queue.
                    "rgb": single.rgb,
                    "depth": single.depth,
                    "preserve": single.preserve,
                    "segmentation": single.segmentation,
                    "seeds": request.seeds[index : index + 1],
                    "prompts": (request.prompts[index],),
                    "camera": request.camera,
                },
                self.cfg.request_timeout_s,
            )
            # Peer copy back onto the simulator's device; still no host staging.
            return reply["rgb"].to(frame.rgb.device, non_blocking=True)

        if self._pool is None:
            results = [one(i) for i in range(count)]
        else:
            results = [f.result() for f in [self._pool.submit(one, i) for i in range(count)]]
        return torch.cat(results, dim=0)


def default_worker_devices(simulator_device: int = 0) -> tuple[int, ...]:
    """Every visible GPU except the simulator's, as a starting point."""
    return tuple(d for d in range(torch.cuda.device_count()) if d != simulator_device) or (simulator_device,)


def visible_devices() -> str:
    """Report ``CUDA_VISIBLE_DEVICES`` as configured, for error messages."""
    return os.environ.get("CUDA_VISIBLE_DEVICES", "<unset: all GPUs visible>")
