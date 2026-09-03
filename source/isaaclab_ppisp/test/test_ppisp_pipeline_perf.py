# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Opt-in PPISP pipeline performance sweep.

This test is skipped by default because it is a benchmarking sweep, not a
correctness test. The configured resolutions are final tiled render-target
resolutions; the benchmark derives each per-environment image size from the
same near-square layout used by tiled camera visualization. Run manually with:

```
./isaaclab.sh -p -m pip install pytest
ISAACLAB_PPISP_PERF=1 ./isaaclab.sh -p -m pytest \
    source/isaaclab_ppisp/test/test_ppisp_pipeline_perf.py -s
```
"""

from __future__ import annotations

import csv
import gc
import math
import os
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
import warp as wp
from isaaclab_ppisp import PPISP_CONTROLLER_EXPECTED_WEIGHTS_LEN, PpispCfg, PpispPipeline
from isaaclab_ppisp.kernels import (
    PPISP_CONTROLLER_FEATURE_LEN,
    PPISP_CONTROLLER_HIDDEN_DIM,
    PPISP_CONTROLLER_MLP_THREAD_GROUP_SIZE,
    PPISP_CONTROLLER_OFF_CONV1_B,
    PPISP_CONTROLLER_OFF_CONV2_B,
    PPISP_CONTROLLER_OFF_CONV3_B,
    PPISP_CONTROLLER_OFF_EXP_B,
    PPISP_CONTROLLER_OFF_TRUNK0_B,
    PPISP_CONTROLLER_OFF_TRUNK1_B,
    PPISP_CONTROLLER_OFF_TRUNK2_B,
    PPISP_CONTROLLER_PARAM_COUNT,
    PPISP_CONTROLLER_POOL_CELL_COUNT,
    PPISP_CONTROLLER_POOL_THREAD_GROUP_SIZE,
    _ppisp_controller_mlp_native_kernel,
    _ppisp_controller_pool_features_native_kernel,
    apply_ppisp_to_rgba,
    apply_ppisp_to_rgba_with_controller_params,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("ISAACLAB_PPISP_PERF", "0") != "1",
    reason="PPISP performance sweep is opt-in; set ISAACLAB_PPISP_PERF=1 to run.",
)

_DEFAULT_RESOLUTIONS = ((256, 256), (512, 512), (1024, 1024), (2048, 2048))
_DEFAULT_NUM_ENVS = (1, 8, 32, 128, 512)
_DEFAULT_WARMUP_ITERS = 2
_DEFAULT_MEASURE_ITERS = 5
_DEFAULT_MEMORY_FRACTION = 0.85


def _parse_resolutions(value: str | None) -> tuple[tuple[int, int], ...]:
    if not value:
        return _DEFAULT_RESOLUTIONS
    resolutions = []
    for item in value.split(","):
        item = item.strip().lower()
        if not item:
            continue
        if "x" in item:
            height, width = item.split("x", maxsplit=1)
            resolutions.append((int(height), int(width)))
        else:
            size = int(item)
            resolutions.append((size, size))
    return tuple(resolutions)


def _parse_ints(value: str | None, default: tuple[int, ...]) -> tuple[int, ...]:
    if not value:
        return default
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _parse_variants(value: str | None) -> tuple[str, ...]:
    if not value:
        return ("static", "controller")
    variants = tuple(item.strip().lower() for item in value.split(",") if item.strip())
    valid = {"static", "controller"}
    unknown = set(variants) - valid
    if unknown:
        raise ValueError(f"Unknown PPISP perf variants: {sorted(unknown)}. Expected any of {sorted(valid)}.")
    return variants


def _tile_grid_shape(num_envs: int) -> tuple[int, int]:
    cols = max(1, math.ceil(math.sqrt(max(1, num_envs))))
    rows = math.ceil(max(1, num_envs) / cols)
    return rows, cols


def _per_env_resolution(final_height: int, final_width: int, num_envs: int) -> tuple[int, int, int, int]:
    tile_rows, tile_cols = _tile_grid_shape(num_envs)
    return max(1, final_height // tile_rows), max(1, final_width // tile_cols), tile_rows, tile_cols


def _estimated_bytes(num_envs: int, height: int, width: int, *, controller: bool) -> int:
    hdr_bytes = num_envs * height * width * 3 * np.dtype(np.float32).itemsize
    rgba_bytes = num_envs * height * width * 4 * np.dtype(np.uint8).itemsize
    controller_bytes = 0
    if controller:
        controller_bytes += PPISP_CONTROLLER_EXPECTED_WEIGHTS_LEN * np.dtype(np.float32).itemsize
        controller_bytes += num_envs * (1600 + 9) * np.dtype(np.float32).itemsize
    return hdr_bytes + rgba_bytes + controller_bytes


@lru_cache(maxsize=1)
def _synthetic_controller_weights() -> tuple[float, ...]:
    rng = np.random.default_rng(11)
    weights = rng.normal(0.0, 0.003, size=PPISP_CONTROLLER_EXPECTED_WEIGHTS_LEN).astype(np.float32)
    for offset, size in (
        (PPISP_CONTROLLER_OFF_CONV1_B, 16),
        (PPISP_CONTROLLER_OFF_CONV2_B, 32),
        (PPISP_CONTROLLER_OFF_CONV3_B, 64),
        (PPISP_CONTROLLER_OFF_TRUNK0_B, PPISP_CONTROLLER_HIDDEN_DIM),
        (PPISP_CONTROLLER_OFF_TRUNK1_B, PPISP_CONTROLLER_HIDDEN_DIM),
        (PPISP_CONTROLLER_OFF_TRUNK2_B, PPISP_CONTROLLER_HIDDEN_DIM),
    ):
        weights[offset : offset + size] += 0.04
    weights[PPISP_CONTROLLER_OFF_EXP_B] = 0.0
    return tuple(float(value) for value in weights)


def _is_controller_variant(variant: str) -> bool:
    return variant != "static"


def _make_pipeline(variant: str) -> PpispPipeline:
    if variant == "static":
        return PpispPipeline(PpispCfg(inputs={"exposureOffset": 0.0}))
    return PpispPipeline(PpispCfg(controller_weights=_synthetic_controller_weights()))


def _time_pipeline_apply(
    pipeline: PpispPipeline,
    hdr: wp.array,
    rgba: wp.array,
    *,
    warmup_iters: int,
    measure_iters: int,
) -> float:
    for _ in range(warmup_iters):
        pipeline.apply(hdr, rgba)
    wp.synchronize()

    start = wp.Event(enable_timing=True)
    end = wp.Event(enable_timing=True)
    wp.record_event(start)
    for _ in range(measure_iters):
        pipeline.apply(hdr, rgba)
    wp.record_event(end)
    return float(wp.get_event_elapsed_time(start, end)) / float(measure_iters)


def _time_gpu_operation(operation, *, warmup_iters: int, measure_iters: int) -> float:
    for _ in range(warmup_iters):
        operation()
    wp.synchronize()

    start = wp.Event(enable_timing=True)
    end = wp.Event(enable_timing=True)
    wp.record_event(start)
    for _ in range(measure_iters):
        operation()
    wp.record_event(end)
    return float(wp.get_event_elapsed_time(start, end)) / float(measure_iters)


def _launch_controller_pool(
    hdr: wp.array,
    weights: wp.array,
    features: wp.array,
    responsivity: float,
) -> None:
    wp.launch_tiled(
        _ppisp_controller_pool_features_native_kernel,
        dim=(int(hdr.shape[0]), PPISP_CONTROLLER_POOL_CELL_COUNT),
        inputs=[hdr, weights, features, int(hdr.shape[2]), int(hdr.shape[1]), float(responsivity)],
        device=str(hdr.device),
        block_dim=PPISP_CONTROLLER_POOL_THREAD_GROUP_SIZE,
    )


def _launch_controller_mlp(
    features: wp.array,
    weights: wp.array,
    controller_params: wp.array,
    prior_exposure: float,
) -> None:
    wp.launch_tiled(
        _ppisp_controller_mlp_native_kernel,
        dim=(int(features.shape[0]),),
        inputs=[features, weights, controller_params, float(prior_exposure)],
        device=str(features.device),
        block_dim=PPISP_CONTROLLER_MLP_THREAD_GROUP_SIZE,
    )


def _print_perf_table(rows: list[dict[str, object]]) -> None:
    print(
        "\n[ppisp pipeline perf]\n"
        "variant,num_envs,final_height,final_width,tile_rows,tile_cols,per_env_height,per_env_width,"
        "processed_mpix,final_mpix,estimated_gib,mean_ms,staged_total_ms,controller_pool_ms,"
        "controller_mlp_ms,apply_ms,stage_delta_ms,ms_per_env,throughput_mpix_s,status,error"
    )
    for row in rows:
        print(
            "{variant},{num_envs},{final_height},{final_width},{tile_rows},{tile_cols},"
            "{per_env_height},{per_env_width},{processed_mpix:.3f},{final_mpix:.3f},{estimated_gib:.2f},"
            "{mean_ms:.3f},{staged_total_ms:.3f},{controller_pool_ms:.3f},{controller_mlp_ms:.3f},"
            "{apply_ms:.3f},{stage_delta_ms:.3f},{ms_per_env:.5f},{throughput_mpix_s:.2f},{status},{error}".format(
                **row
            )
        )


def _is_allocation_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return bool(re.search(r"out[ -]of[ -]memory|failed to allocate|allocation", message))


def test_ppisp_pipeline_perf_sweep(tmp_path):
    if not wp.is_cuda_available():
        pytest.skip("PPISP performance sweep requires CUDA.")

    device_name = os.environ.get("ISAACLAB_PPISP_PERF_DEVICE", "cuda:0")
    device = wp.get_device(device_name)
    if not device.is_cuda:
        pytest.skip(f"PPISP performance sweep requires a CUDA device, got {device_name}.")

    resolutions = _parse_resolutions(os.environ.get("ISAACLAB_PPISP_PERF_RESOLUTIONS"))
    num_envs_values = _parse_ints(os.environ.get("ISAACLAB_PPISP_PERF_NUM_ENVS"), _DEFAULT_NUM_ENVS)
    variants = _parse_variants(os.environ.get("ISAACLAB_PPISP_PERF_VARIANTS"))
    warmup_iters = int(os.environ.get("ISAACLAB_PPISP_PERF_WARMUP_ITERS", _DEFAULT_WARMUP_ITERS))
    measure_iters = int(os.environ.get("ISAACLAB_PPISP_PERF_MEASURE_ITERS", _DEFAULT_MEASURE_ITERS))
    memory_fraction = float(os.environ.get("ISAACLAB_PPISP_PERF_MEMORY_FRACTION", _DEFAULT_MEMORY_FRACTION))
    max_case_bytes = int(device.total_memory * memory_fraction)
    output_path = Path(os.environ.get("ISAACLAB_PPISP_PERF_OUTPUT", tmp_path / "ppisp_pipeline_perf.csv"))

    rows: list[dict[str, object]] = []
    pipelines = {variant: _make_pipeline(variant) for variant in variants}
    with wp.ScopedDevice(device):
        for final_height, final_width in resolutions:
            for num_envs in num_envs_values:
                height, width, tile_rows, tile_cols = _per_env_resolution(final_height, final_width, num_envs)
                processed_mpix = (num_envs * height * width) / 1.0e6
                for variant in variants:
                    estimated_bytes = _estimated_bytes(
                        num_envs,
                        height,
                        width,
                        controller=_is_controller_variant(variant),
                    )
                    row = {
                        "variant": variant,
                        "num_envs": num_envs,
                        "final_height": final_height,
                        "final_width": final_width,
                        "tile_rows": tile_rows,
                        "tile_cols": tile_cols,
                        "per_env_height": height,
                        "per_env_width": width,
                        "processed_mpix": processed_mpix,
                        "final_mpix": (final_height * final_width) / 1.0e6,
                        "estimated_gib": estimated_bytes / 1024**3,
                        "mean_ms": 0.0,
                        "staged_total_ms": 0.0,
                        "controller_pool_ms": 0.0,
                        "controller_mlp_ms": 0.0,
                        "apply_ms": 0.0,
                        "stage_delta_ms": 0.0,
                        "ms_per_env": 0.0,
                        "throughput_mpix_s": 0.0,
                        "status": "ok",
                        "error": "",
                    }
                    if estimated_bytes > max_case_bytes:
                        row["status"] = "skipped_memory_guard"
                        rows.append(row)
                        continue

                    hdr = None
                    rgba = None
                    weights = None
                    features = None
                    controller_params = None
                    try:
                        hdr = wp.full(
                            (num_envs, height, width, 3),
                            value=wp.float32(0.25),
                            dtype=wp.float32,
                            device=device,
                        )
                        rgba = wp.empty((num_envs, height, width, 4), dtype=wp.uint8, device=device)
                        pipeline = pipelines[variant]
                        mean_ms = _time_pipeline_apply(
                            pipeline,
                            hdr,
                            rgba,
                            warmup_iters=warmup_iters,
                            measure_iters=measure_iters,
                        )
                        if _is_controller_variant(variant):
                            weights = wp.array(pipeline.cfg.controller_weights, dtype=wp.float32, device=device)
                            features = wp.empty(
                                (num_envs, PPISP_CONTROLLER_FEATURE_LEN), dtype=wp.float32, device=device
                            )
                            controller_params = wp.empty(
                                (num_envs, PPISP_CONTROLLER_PARAM_COUNT), dtype=wp.float32, device=device
                            )
                            _launch_controller_pool(hdr, weights, features, float(pipeline.cfg.controller_responsivity))
                            _launch_controller_mlp(
                                features,
                                weights,
                                controller_params,
                                pipeline.cfg.controller_prior_exposure,
                            )
                            wp.synchronize()
                            controller_pool_ms = _time_gpu_operation(
                                lambda: _launch_controller_pool(
                                    hdr, weights, features, float(pipeline.cfg.controller_responsivity)
                                ),
                                warmup_iters=warmup_iters,
                                measure_iters=measure_iters,
                            )
                            controller_mlp_ms = _time_gpu_operation(
                                lambda: _launch_controller_mlp(
                                    features,
                                    weights,
                                    controller_params,
                                    pipeline.cfg.controller_prior_exposure,
                                ),
                                warmup_iters=warmup_iters,
                                measure_iters=measure_iters,
                            )
                            apply_ms = _time_gpu_operation(
                                lambda: apply_ppisp_to_rgba_with_controller_params(
                                    hdr, rgba, pipeline.cfg, controller_params
                                ),
                                warmup_iters=warmup_iters,
                                measure_iters=measure_iters,
                            )
                            staged_total_ms = controller_pool_ms + controller_mlp_ms + apply_ms
                            row["controller_pool_ms"] = controller_pool_ms
                            row["controller_mlp_ms"] = controller_mlp_ms
                        else:
                            apply_ms = _time_gpu_operation(
                                lambda: apply_ppisp_to_rgba(hdr, rgba, pipeline.cfg),
                                warmup_iters=warmup_iters,
                                measure_iters=measure_iters,
                            )
                            staged_total_ms = apply_ms
                        row["staged_total_ms"] = staged_total_ms
                        row["apply_ms"] = apply_ms
                        row["stage_delta_ms"] = mean_ms - staged_total_ms
                        row["mean_ms"] = mean_ms
                        row["ms_per_env"] = mean_ms / float(num_envs)
                        row["throughput_mpix_s"] = row["processed_mpix"] / (mean_ms / 1000.0)
                    except Exception as exc:
                        if _is_allocation_error(exc):
                            row["status"] = "skipped_allocation"
                        else:
                            row["status"] = f"failed:{type(exc).__name__}"
                        row["error"] = str(exc).replace("\n", " ")[:160]
                    finally:
                        hdr = None
                        rgba = None
                        weights = None
                        features = None
                        controller_params = None
                        gc.collect()
                        wp.synchronize()
                    rows.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    _print_perf_table(rows)
    print(f"[ppisp pipeline perf] wrote {output_path}")
    assert rows
    assert any(row["status"] == "ok" for row in rows)
    failed_rows = [row for row in rows if str(row["status"]).startswith("failed:")]
    assert not failed_rows
