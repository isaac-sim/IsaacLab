# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Temporary NVBug 7071342 helper: replay Kit cases before the Valiant Isaac RTX color cases. Do not merge.

In CI the Valiant cases run after the 20 cases that precede them in ``KIT_RENDERING_CASES``, inside the same Kit
process. ``make_test`` replays a chosen subset of those cases, snapshots renderer settings and GPU memory, then
repeats the Valiant RGB and HDR cases without their xfail marks.
"""

import dataclasses
import json
from pathlib import Path

import pytest
import torch
from rendering_cases import KIT_RENDERING_CASES
from rendering_runner import run_rendering_case

from isaaclab.renderers import RenderBufferKind

_OUT = Path.cwd() / "tests" / "comparison-images" / "teal-order"
_SETTING_ROOTS = ("/rtx", "/rtx-transient", "/omni", "/renderer", "/app", "/isaaclab", "/persistent", "/exts", "/ngx")
VALIANT_CASES = [
    case
    for case in KIT_RENDERING_CASES
    if case.scene == "gaussian_splats_valiant_1env"
    and case.renderer == "isaac_rtx"
    and case.aovs in ((RenderBufferKind.RGB,), (RenderBufferKind.RGB_HDR,))
]
PRECEDING_CASES = KIT_RENDERING_CASES[: next(i for i, c in enumerate(KIT_RENDERING_CASES) if c is VALIANT_CASES[0])]


def select_preceding(selection: str) -> list:
    """Return ``all``, ``none`` or comma-separated 1-based positions and ranges of the preceding cases (``1-8,19``)."""
    if selection == "all":
        return list(PRECEDING_CASES)
    if selection in ("", "none"):
        return []
    positions = set()
    for part in selection.split(","):
        low, _, high = part.partition("-")
        positions.update(range(int(low), int(high or low) + 1))
    return [PRECEDING_CASES[i - 1] for i in sorted(positions)]


def _snapshot(label: str) -> None:
    import carb

    settings = carb.settings.get_settings()
    snapshot = {root: settings.get(root) for root in _SETTING_ROOTS}
    free, total = torch.cuda.mem_get_info()
    snapshot["gpu_memory_mib"] = {"free": free // 2**20, "total": total // 2**20}
    _OUT.mkdir(parents=True, exist_ok=True)
    (_OUT / f"settings-{label}.json").write_text(json.dumps(snapshot, indent=1, sort_keys=True, default=str))


def _apply_overrides(overrides: dict) -> None:
    import carb

    settings = carb.settings.get_settings()
    for key, value in overrides.items():
        settings.set(key, value)


def make_test(selection: str, repeats: int, label: str, overrides: dict | None = None):
    """Build the parametrized test: preceding cases, carb overrides, a settings snapshot, then Valiant RGB/HDR pairs."""
    steps = [("pre", case, 0) for case in select_preceding(selection)]
    if overrides:
        steps.append(("overrides", None, 0))
    steps.append(("snapshot", None, 0))
    steps += [("valiant", case, i) for i in range(repeats) for case in VALIANT_CASES]
    ids = [
        f"pre-{c.id}" if k == "pre" else f"valiant{i:02d}-{c.id}" if k == "valiant" else k for k, c, i in steps
    ]

    @pytest.mark.parametrize("step", steps, ids=ids)
    def test_valiant_teal_order(step, request: pytest.FixtureRequest) -> None:
        kind, case, iteration = step
        if kind == "pre":
            # Failures of the replayed cases do not matter; only the state they leave behind does.
            try:
                run_rendering_case(dataclasses.replace(case, marks=()), request)
            except Exception as exc:  # noqa: BLE001
                request.node.user_properties.append(("replay_error", f"{type(exc).__name__}: {str(exc)[:200]}"))
        elif kind == "overrides":
            _apply_overrides(overrides)
        elif kind == "snapshot":
            _snapshot(label)
        else:
            import omni.kit.app

            free, _ = torch.cuda.mem_get_info()
            request.node.user_properties.extend(
                [
                    ("kit_update_number", str(omni.kit.app.get_app().get_update_number())),
                    ("gpu_free_mib", str(free // 2**20)),
                ]
            )
            namespace = f"{label}-iter{iteration:02d}"
            run_rendering_case(dataclasses.replace(case, marks=()), request, artifact_namespace=namespace)

    return test_valiant_teal_order
