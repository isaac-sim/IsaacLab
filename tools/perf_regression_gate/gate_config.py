# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import json
from pathlib import Path


DEFAULT_K_WARN = 2.5
DEFAULT_K_BLOCK = 4.0
MIN_BASELINE_SAMPLES = 5
MAX_BASELINE_SAMPLES = 20
MIN_BLOCK_REGRESSION_PCT = 3.0
BASELINE_PUSH_RETRIES = 3
DEFAULT_RUNTIME_COMPATIBILITY = {
    "contract_version": 1,
    "always": [
        "software.isaacsim",
        "software.isaaclab",
        "software.torch",
        "software.warp",
    ],
    "by_physics_backend": {
        "physx": [
            "software.isaaclab_physx",
        ],
        "newton": [
            "software.isaaclab_newton",
            "software.newton",
        ],
    },
    "by_render_backend": {
        "newton_renderer": [
            "software.isaaclab_ov",
        ],
        "ovrtx_renderer": [
            "software.isaaclab_ov",
        ],
        "warp_renderer": [
            "software.isaaclab_ov",
        ],
        "rtx_renderer": [
            "software.isaaclab_ov",
        ],
    },
    "publish_only": [
        "hardware.gpu_compute_capability",
        "hardware.gpu_total_memory_gb",
        "gpu_diag.cuda_version",
        "gpu_diag.nvidia_driver_version",
    ],
}


def _merge_runtime_compatibility(raw: dict | None) -> dict:
    policy = json.loads(json.dumps(DEFAULT_RUNTIME_COMPATIBILITY))
    if not raw:
        return policy
    for key, value in raw.items():
        if isinstance(value, dict) and isinstance(policy.get(key), dict):
            policy[key].update(value)
        else:
            policy[key] = value
    return policy


def load_gate_config(path: Path | str) -> dict:
    config = {
        "blocking": False,
        "min_baseline_samples": MIN_BASELINE_SAMPLES,
        "max_baseline_samples": MAX_BASELINE_SAMPLES,
        "min_block_regression_pct": MIN_BLOCK_REGRESSION_PCT,
        "baseline_push_retries": BASELINE_PUSH_RETRIES,
        "runtime_compatibility": _merge_runtime_compatibility(None),
    }
    p = Path(path)
    if p.exists():
        with p.open() as fh:
            loaded = json.load(fh)
        runtime_policy = _merge_runtime_compatibility(loaded.pop("runtime_compatibility", None))
        config.update(loaded)
        config["runtime_compatibility"] = runtime_policy
    return config
