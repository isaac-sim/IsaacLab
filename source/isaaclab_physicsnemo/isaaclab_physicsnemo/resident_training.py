# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""GPU-resident, mixed-precision, resumable transition-model training for FlashNeRD-scale budgets."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
import torch.distributed as dist
from physicsnemo.distributed import DistributedManager
from physicsnemo_robotics.data import TransitionWindowReader
from physicsnemo_robotics.dynamics import FieldNormalizer, TransitionAdapter
from physicsnemo_robotics.models.parallel_nerd import model as parallel_nerd_model
from physicsnemo_robotics.training import LoaderOptions, normalize_transition
from physicsnemo_robotics.training.distributed import build_stock_data_pipeline

_LOGGER = logging.getLogger("isaaclab_physicsnemo_training")
_RANK_SEED_STRIDE = 1_000_003
_MATERIALIZE_BATCH = 4096


@dataclass(frozen=True)
class ResidentOptions:
    """Optimization controls of :func:`fit_resident`."""

    max_updates: int
    batch_size: int
    learning_rate: float
    final_learning_rate: float
    weight_decay: float
    gradient_clip_norm: float
    seed: int
    log_every: int
    mixed_precision: bool
    checkpoint_every: int
    validate_every: int
    compile: bool = False
    max_overflow_retries: int = 32


@dataclass(frozen=True)
class ResidentWindows:
    """Normalized transition windows resident on the training device."""

    inputs: torch.Tensor
    """Normalized model inputs, shape [windows, history, input_dim]."""

    targets: torch.Tensor
    """Normalized prediction targets, shape [windows, prediction_dim]."""


class _PredictLast(torch.nn.Module):
    """Expose the newest-frame prediction as ``forward`` so data-parallel gradient hooks apply."""

    def __init__(self, network: torch.nn.Module) -> None:
        super().__init__()
        self.network = network

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        predict_last = getattr(self.network, "predict_last", None)
        return cast(torch.Tensor, predict_last(inputs) if callable(predict_last) else self.network(inputs)[:, -1])


@contextmanager
def _fp32_parallel_nerd_attention() -> Iterator[None]:
    """Keep ParallelNeRD attention logits in FP32 under autocast.

    ``_ParallelBlock._full_attention`` casts queries and keys to FP32 for its logits, but autocast casts
    the matmul back to FP16, where logits beyond 65,504 overflow and the softmax returns NaN. Trained
    models reach such logits within tens of thousands of updates at FlashNeRD learning rates.
    """
    block_class: Any = parallel_nerd_model._ParallelBlock  # pyright: ignore[reportPrivateUsage]
    attention = block_class._full_attention

    def full_precision_attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        with torch.autocast("cuda", enabled=False):
            return attention(self, query, key, value)

    block_class._full_attention = full_precision_attention
    try:
        yield
    finally:
        block_class._full_attention = attention


def materialize_windows(
    windows: TransitionWindowReader,
    adapter: TransitionAdapter,
    normalizer: FieldNormalizer,
    *,
    device: torch.device,
    input_dtype: torch.dtype,
    num_workers: int = 0,
) -> ResidentWindows:
    """Normalize every window once, exactly as the streaming training pipeline does, and keep it on ``device``.

    Args:
        windows: Transition windows to materialize. Every window is read once, in order.
        adapter: Transition adapter that builds model inputs and targets.
        normalizer: Fitted transition normalizer.
        device: Device holding the windows.
        input_dtype: Storage dtype of the inputs; targets stay float32.
        num_workers: Loader worker processes.

    Returns:
        The resident normalized windows.
    """
    pipeline = build_stock_data_pipeline(
        windows,
        device=device,
        batch_size=_MATERIALIZE_BATCH,
        seed=0,
        options=LoaderOptions(num_workers=num_workers),
    )
    inputs, targets = [], []
    try:
        for loaded in pipeline.loader:
            transition, target = normalize_transition(adapter, normalizer, loaded)
            inputs.append(transition.inputs.to(dtype=input_dtype))
            targets.append(target.to(dtype=torch.float32))
    finally:
        pipeline.close()
    return ResidentWindows(torch.cat(inputs), torch.cat(targets))


def fit_resident(
    network: torch.nn.Module,
    train: ResidentWindows,
    validation: ResidentWindows,
    normalizer: FieldNormalizer,
    options: ResidentOptions,
    *,
    device: torch.device,
    checkpoint_path: Path,
    identity: Mapping[str, Any],
    on_checkpoint: Callable[[int], None] | None = None,
) -> tuple[int, dict[str, float]]:
    """Train ``network`` on resident windows with uniform sampling with replacement.

    Every rank samples its share of the global batch independently from all training windows. With
    ``compile``, the training forward and backward passes run through :func:`torch.compile`. With
    ``mixed_precision``, projections run under FP16 autocast while attention logits, parameters, optimizer
    state, loss, and metrics stay FP32, and a batch whose gradients overflow is retried with a lower loss
    scale without advancing the sampler or the learning-rate schedule. A checkpoint is written every
    ``checkpoint_every`` updates and at the end; training resumes from it when its ``identity`` matches.

    Args:
        network: Model to train, already on ``device``.
        train: Resident training windows.
        validation: Resident validation windows.
        normalizer: Normalizer of the windows, used for physical-unit metrics.
        options: Optimization controls.
        device: Training device.
        checkpoint_path: Resumable checkpoint file.
        identity: Run identity that a checkpoint must match to be resumed.
        on_checkpoint: Called on rank zero after each checkpoint with the completed update count.

    Returns:
        The completed update count and the final validation metrics.

    Raises:
        ValueError: If the global batch is not divisible by the number of ranks, or a checkpoint of a
            different run exists.
        FloatingPointError: If the loss is not finite, or a batch still overflows after the allowed retries.
    """
    with _fp32_parallel_nerd_attention():
        return _fit_resident(
            network,
            train,
            validation,
            normalizer,
            options,
            device=device,
            checkpoint_path=checkpoint_path,
            identity=identity,
            on_checkpoint=on_checkpoint,
        )


def _fit_resident(
    network: torch.nn.Module,
    train: ResidentWindows,
    validation: ResidentWindows,
    normalizer: FieldNormalizer,
    options: ResidentOptions,
    *,
    device: torch.device,
    checkpoint_path: Path,
    identity: Mapping[str, Any],
    on_checkpoint: Callable[[int], None] | None,
) -> tuple[int, dict[str, float]]:
    manager = DistributedManager()
    world_size, rank = manager.world_size, manager.rank
    if options.batch_size % world_size:
        raise ValueError(f"The batch size {options.batch_size} is not divisible by {world_size} ranks.")
    local_batch = options.batch_size // world_size
    predictor = _PredictLast(network)
    model = (
        torch.nn.parallel.DistributedDataParallel(predictor, device_ids=[device.index]) if world_size > 1 else predictor
    )
    if options.compile:
        # Fusing the many small attention and normalization kernels roughly doubles the update rate.
        model = torch.compile(model)
    optimizer = torch.optim.AdamW(
        network.parameters(),
        lr=options.learning_rate,
        weight_decay=options.weight_decay,
        fused=device.type == "cuda",
    )
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=options.final_learning_rate / options.learning_rate,
        total_iters=options.max_updates,
    )
    scaler = torch.amp.grad_scaler.GradScaler("cuda", enabled=options.mixed_precision)
    fingerprint = hashlib.sha256(json.dumps(dict(identity), sort_keys=True, default=str).encode()).hexdigest()

    completed = 0
    if checkpoint_path.is_file():
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if state["fingerprint"] != fingerprint:
            raise ValueError(f"The checkpoint at {checkpoint_path} belongs to a different run; remove it to restart.")
        network.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        completed = int(state["completed_updates"])
        _LOGGER.info(f"resumed from update {completed} of {options.max_updates}")
    # A resumed run draws a fresh, deterministic index stream for the remaining updates.
    generator = torch.Generator(device=device).manual_seed(options.seed + rank * _RANK_SEED_STRIDE + completed * 7_919)

    def save_checkpoint() -> None:
        if rank == 0:
            payload = {
                "fingerprint": fingerprint,
                "identity": dict(identity),
                "completed_updates": completed,
                "model": network.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
            }
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = checkpoint_path.with_suffix(".tmp")
            torch.save(payload, temporary)
            os.replace(temporary, checkpoint_path)
            if on_checkpoint is not None:
                on_checkpoint(completed)
        if world_size > 1:
            dist.barrier()

    window_count = train.inputs.shape[0]
    loss_sum, loss_count, started = 0.0, 0, time.perf_counter()
    model.train()
    while completed < options.max_updates:
        index = torch.randint(window_count, (local_batch,), device=device, generator=generator)
        inputs, target = train.inputs[index], train.targets[index]
        for _ in range(options.max_overflow_retries):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=options.mixed_precision):
                prediction = model(inputs)
            loss = (prediction.float() - target).square().mean()
            # A lower loss scale cannot repair a non-finite forward pass, so stop instead of retrying.
            if not bool(torch.isfinite(loss.detach())):
                raise FloatingPointError(f"Update {completed + 1} produced a non-finite loss.")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(network.parameters(), options.gradient_clip_norm)
            scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            # GradScaler lowers its scale exactly when it skipped the step for non-finite gradients.
            if not options.mixed_precision or scaler.get_scale() >= scale:
                break
        else:
            raise FloatingPointError(f"Update {completed + 1} overflowed {options.max_overflow_retries} times.")
        scheduler.step()
        completed += 1
        loss_sum, loss_count = loss_sum + loss.detach(), loss_count + 1
        if completed % options.log_every == 0 and rank == 0:
            elapsed = time.perf_counter() - started
            _LOGGER.info(
                f"update {completed}/{options.max_updates} ({100.0 * completed / options.max_updates:.1f}%)"
                f"  normalized_delta_mse={float(loss_sum) / loss_count:.3e}  lr={scheduler.get_last_lr()[0]:.2e}"
                f"  {1000.0 * elapsed / loss_count:.1f} ms/update ({loss_count / elapsed:.1f}/s)"
            )
            loss_sum, loss_count, started = 0.0, 0, time.perf_counter()
        if options.validate_every and completed % options.validate_every == 0 and rank == 0:
            metrics = validate_resident(predictor, validation, normalizer, mixed_precision=options.mixed_precision)
            _LOGGER.info(f"update {completed} validation {metrics}")
            model.train()
        if options.checkpoint_every and completed % options.checkpoint_every == 0 and completed < options.max_updates:
            save_checkpoint()
    save_checkpoint()
    return completed, validate_resident(predictor, validation, normalizer, mixed_precision=options.mixed_precision)


@torch.inference_mode()
def validate_resident(
    model: torch.nn.Module,
    windows: ResidentWindows,
    normalizer: FieldNormalizer,
    *,
    mixed_precision: bool,
    batch_size: int = 16_384,
) -> dict[str, float]:
    """Return the pooled one-step delta RMSE of resident windows, normalized and in physical units."""
    model.eval()
    normalized_sse = physical_sse = 0.0
    for start in range(0, windows.inputs.shape[0], batch_size):
        inputs, target = windows.inputs[start : start + batch_size], windows.targets[start : start + batch_size]
        with torch.autocast("cuda", dtype=torch.float16, enabled=mixed_precision):
            prediction = model(inputs).float()
        normalized_sse += float((prediction - target).double().square().sum())
        physical_error = normalizer.denormalize("target", prediction) - normalizer.denormalize("target", target)
        physical_sse += float(physical_error.double().square().sum())
    count = windows.targets.numel()
    return {"delta_rmse": (physical_sse / count) ** 0.5, "normalized_delta_rmse": (normalized_sse / count) ** 0.5}
