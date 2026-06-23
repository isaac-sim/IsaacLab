# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Canonical GPU identity helpers for baseline bucket selection"""

from __future__ import annotations

import re
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
    """Return candidate keys for reading existing GPU-keyed config dictionaries.

    `gpu_model` is canonical for new artifacts, but existing task floor configs may
    still use legacy display keys such as `L40S`.
    """
    raw = _clean(value)
    canonical = canonical_gpu_model(raw)
    keys: list[str] = []
    for key in (canonical, raw):
        if key and key not in keys:
            keys.append(key)

    legacy = {
        "l40s": ["L40S"],
        "l40": ["L40"],
        "rtx_pro_6000_blackwell": ["RTX6000", "RTX PRO 6000", "RTX PRO 6000 Blackwell"],
        "rtx_pro_6000": ["RTX6000", "RTX PRO 6000"],
        "rtx_6000_ada": ["RTX6000", "RTX 6000 Ada"],
        "rtx_6000": ["RTX6000", "RTX 6000"],
        "rtx_a6000": ["RTXA6000", "RTX A6000"],
    }
    for key in legacy.get(canonical, []):
        if key not in keys:
            keys.append(key)
    return keys


def normalize_gpu_fields(value: Any) -> dict[str, str]:
    raw = _clean(value)
    return {
        "gpu_model": canonical_gpu_model(raw),
        "gpu_model_raw": raw or _UNKNOWN_GPU,
    }
