# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""End-to-end test that typed preset selectors reach the resolver from each
unified train/play entrypoint.

The four supported RL libraries (rl_games, rsl_rl, sb3, skrl) each have a
``train_<library>.py`` / ``play_<library>.py`` script that the unified
``scripts/reinforcement_learning/{train,play}.py`` dispatchers route to via
``--rl_library``. Each entrypoint must wire :func:`setup_preset_cli` and pass
its remainder through to Hydra so that user-typed ``physics=NAME`` /
``renderer=NAME`` / ``presets=NAME`` tokens reach
:func:`~isaaclab_tasks.utils.hydra.register_task`, which parses them directly.
If an entrypoint dropped the remainder (or routed the token to Hydra as a raw
override), the token would hit Hydra as a struct override against a
non-existent top-level key and raise ``Key 'physics' is not in struct`` -- the
original symptom of the #5715 regression.

This test invokes each entrypoint via the unified dispatcher with
``physics=does_not_exist`` and asserts:

* the Hydra struct error is **absent** (would indicate the token was not
  routed to ``register_task``), and
* the resolver's own ``Unknown preset(s)`` error is **present** (the token
  reached the resolver).

Resolve fails before any Kit/sim launch, so each subprocess exits in a
few seconds without needing GPU.

``rlinf`` is intentionally excluded: those scripts manage their own
``GlobalHydra`` instance and have never been integrated with the typed
preset CLI.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]

# Each (action, library) pair runs through the unified dispatcher at
# ``scripts/reinforcement_learning/{action}.py``. Dispatching matches how
# ``./isaaclab.sh train`` / ``./isaaclab.sh play`` invoke these in practice.
_ENTRYPOINT_CASES = [
    (action, library) for action in ("train", "play") for library in ("rl_games", "rsl_rl", "sb3", "skrl")
]


@pytest.mark.parametrize("action,library", _ENTRYPOINT_CASES)
def test_typed_preset_reaches_resolver(action: str, library: str) -> None:
    """``physics=<unknown>`` must reach the resolver, not crash Hydra's struct check.

    Confirms that the dispatched entrypoint wired ``setup_preset_cli`` and
    passed its remainder through, so the typed selector reached
    ``register_task``, which surfaced its own ``Unknown preset(s)`` error
    against the deliberately invalid name.
    """
    dispatcher = REPO_ROOT / "scripts" / "reinforcement_learning" / f"{action}.py"
    assert dispatcher.exists(), f"missing dispatcher: {dispatcher}"

    cmd = [
        sys.executable,
        str(dispatcher),
        "--rl_library",
        library,
        "--task=Isaac-Ant-v0",
        "physics=does_not_exist",
        "--headless",
    ]
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        env=os.environ.copy(),
    )

    combined = result.stdout + result.stderr
    label = f"{action}.py --rl_library {library}"
    assert "is not in struct" not in combined, (
        f"{label}: Hydra's struct-override error reached the user, meaning the typed preset "
        f"selector was routed to Hydra as a raw override instead of register_task. The entrypoint "
        f"must call setup_preset_cli and pass its remainder through to set_hydra_args / sys.argv.\n"
        f"--- stderr tail ---\n{result.stderr[-2000:]}\n"
        f"--- stdout tail ---\n{result.stdout[-2000:]}\n"
    )
    assert "Unknown preset(s): does_not_exist" in combined, (
        f"{label}: resolver's 'Unknown preset(s)' error did not appear. Either the token did not "
        f"reach the resolver, or the script exited earlier with a different error.\n"
        f"--- stderr tail ---\n{result.stderr[-2000:]}\n"
        f"--- stdout tail ---\n{result.stdout[-2000:]}\n"
    )
