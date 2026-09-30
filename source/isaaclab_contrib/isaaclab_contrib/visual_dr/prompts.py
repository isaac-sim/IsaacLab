# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Resolution of a prompt bank from its configured source."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .cfg import PromptBankCfg


def load_prompt_bank(cfg: PromptBankCfg | None) -> tuple[str, ...]:
    """Resolve a bank to a tuple of prompts, ordered so index selection is stable.

    Order matters: a style is chosen by index, so reordering a bank changes which
    background a given episode receives. Keep banks append-only if you want runs
    to stay comparable.
    """
    if cfg is None:
        return ("",)
    if cfg.variants:
        variants = tuple(cfg.variants)
    elif cfg.path is not None:
        path = Path(cfg.path).expanduser()
        text = path.read_text()
        if path.suffix == ".json":
            loaded = json.loads(text)
            if not isinstance(loaded, list):
                raise ValueError(f"Prompt bank {path} must contain a JSON list of strings")
            variants = tuple(str(item) for item in loaded)
        else:
            variants = tuple(line.strip() for line in text.splitlines() if line.strip())
    elif cfg.ref is not None:
        module_path, _, attribute = cfg.ref.partition(":")
        if not module_path or not attribute:
            raise ValueError(f"Prompt bank ref must be 'package.module:NAME', got {cfg.ref!r}")
        variants = tuple(getattr(importlib.import_module(module_path), attribute))
    else:
        raise ValueError("Prompt bank has no source")

    if not variants:
        raise ValueError("Prompt bank is empty")
    return variants
