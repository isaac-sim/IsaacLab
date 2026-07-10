# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Canonical GPU identity helpers for baseline bucket selection"""

from __future__ import annotations

import re
import subprocess
from typing import Any

_UNKNOWN_GPU = "unknown_gpu"


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or _UNKNOWN_GPU


def canonical_gpu_model(value: Any) -> str:
    """Return the canonical baseline bucket key for a raw GPU model string."""
    raw = _clean(value)
    if not raw:
        return _UNKNOWN_GPU
    normalized = re.sub(r"\s+", " ", raw.lower()).strip()
    compact = re.sub(r"[^a-z0-9]+", "", normalized)

    if "l40s" in compact:
        return "l40s"
    if compact.endswith("l40") or compact == "nvidial40" or "teslal40" in compact:
        return "l40"
    if "rtxpro6000" in compact and "blackwell" in compact:
        return "rtx_pro_6000_blackwell"
    if "rtxpro6000" in compact:
        return "rtx_pro_6000"
    if "rtx6000adageneration" in compact or ("rtx6000" in compact and "ada" in compact):
        return "rtx_6000_ada"
    if compact in {"rtx6000", "nvidiartx6000"} or "rtx6000" in compact:
        return "rtx_6000"
    if "rtxa6000" in compact or "a6000" in compact:
        return "rtx_a6000"
    if "geforcertx5090" in compact:
        return "geforce_rtx_5090"
    if "geforcertx4090" in compact:
        return "geforce_rtx_4090"
    return _slug(raw)


def gpu_model_config_keys(value: Any) -> list[str]:
    """Return candidate keys for reading GPU-keyed config dictionaries.

    Config (``tasks.json`` thresholds) and baseline buckets are keyed by the
    canonical slug (e.g. ``l40s``); the raw string is kept as a defensive
    fallback for a config hand-keyed to the exact detected name.
    """
    raw = _clean(value)
    canonical = canonical_gpu_model(raw)
    keys: list[str] = []
    for key in (canonical, raw):
        if key and key not in keys:
            keys.append(key)
    return keys


def normalize_gpu_fields(value: Any) -> dict[str, str]:
    raw = _clean(value)
    return {
        "gpu_model": canonical_gpu_model(raw),
        "gpu_model_raw": raw or _UNKNOWN_GPU,
    }


def detect_gpu_model(explicit: str = "") -> str:
    """Return the raw GPU model label, preferring ``explicit`` else nvidia-smi.

    Falls back to ``"unknown-gpu"`` when nvidia-smi is unavailable or reports
    nothing. Shared by the local runner and the baseline seeder.
    """
    explicit = _clean(explicit)
    if explicit:
        return explicit
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                name = line.strip()
                if name:
                    return name
    except Exception:
        pass
    return "unknown-gpu"
