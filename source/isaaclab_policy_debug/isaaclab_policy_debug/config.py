# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PolicyDebugCfg:
    """Configuration for an interactive policy comparison session."""

    run_dir: Path
    max_policies: int = 8
    ghost_opacity: float = 0.25

    def __post_init__(self) -> None:
        run_dir = Path(self.run_dir).expanduser().resolve()
        if not run_dir.is_dir():
            raise ValueError(f"Policy debug folder does not exist or is not a directory: {run_dir}")
        try:
            next(run_dir.iterdir(), None)
        except OSError as exc:
            raise ValueError(f"Policy debug folder is not readable: {run_dir}: {exc}") from exc
        if self.max_policies <= 0:
            raise ValueError("max_policies must be greater than zero")
        if not 0.0 <= self.ghost_opacity <= 1.0:
            raise ValueError("ghost_opacity must be in the range [0, 1]")
        object.__setattr__(self, "run_dir", run_dir)
