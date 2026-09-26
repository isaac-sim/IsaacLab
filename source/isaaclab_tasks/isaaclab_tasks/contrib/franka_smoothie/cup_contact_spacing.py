# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task-local cup rigid-contact spacing and read-only native startup identity."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import math
from pathlib import Path

import newton
import numpy as np
import warp as wp

OPTIONS = {
    "version": 1,
    "margin_m": 0.00025,
    "baseline_margin_m": 0.00015,
    "expected_shape_count": 33,
    "scope": "Only receiving-cup rigid Floor and Wall/S000..S031, after the inherited world hook.",
    "excluded": "Particle-only _MPM proxies, threads, handle, fruits, robot and all other shapes.",
    "rationale": (
        "Conditional contact-spacing candidate after S18 completed milk delivery and supported release, "
        "but blackberry_3 repeatedly crossed authored floor/wall planes by micrometres. "
        "Native startup verified the cup surfaces, witness pair eligibility and existing 0.150 mm margins. "
        "The extra 0.100 mm changes compliant contact reference spacing; it does not guarantee equal "
        "measured separation or packed-cup success. Receipt geometry and zero-tolerance gates are unchanged."
    ),
    "physical_validation_passed": False,
}
_COMPARISON_TOLERANCE_M = 1e-10
_NEWTON_ROOT = Path(importlib.util.find_spec("newton").origin).parent
DEPENDENCY_PATHS = (
    Path(__file__).resolve(),
    _NEWTON_ROOT / "_src/solvers/mujoco/kernels.py",
    _NEWTON_ROOT / "_src/solvers/mujoco/solver_mujoco.py",
)


def cup_contact_spacing_contract() -> dict:
    """Return the declared rigid cup margin [m] without mutable shared metadata."""
    return copy.deepcopy(OPTIONS)


def _selected_indices(labels: list[str], env_id: int) -> list[int]:
    prefix = f"/World/envs/env_{env_id}/Cup/Cup/"
    expected = [prefix + "Floor", *(prefix + f"Wall/S{i:03d}" for i in range(32))]
    selected = [
        i
        for i, label in enumerate(labels)
        if not label.endswith("_MPM") and (label == prefix + "Floor" or label.startswith(prefix + "Wall/"))
    ]
    actual = [labels[i] for i in selected]
    if len(actual) != OPTIONS["expected_shape_count"] or set(actual) != set(expected):
        raise ValueError("Require exactly one cup Floor and 32 declared rigid Wall shapes.")
    return selected


def apply_cup_contact_spacing(builder: newton.ModelBuilder, env_id: int) -> None:
    """Set only the selected cup rigid margins [m], after inherited builder configuration."""
    selected = _selected_indices(list(builder.shape_label), env_id)
    for i in selected:
        flags = int(builder.shape_flags[i])
        if (
            builder.shape_world[i] != env_id
            or not flags & int(newton.ShapeFlags.COLLIDE_SHAPES)
            or flags & int(newton.ShapeFlags.COLLIDE_PARTICLES)
            or not math.isclose(
                builder.shape_margin[i], OPTIONS["baseline_margin_m"], rel_tol=0, abs_tol=_COMPARISON_TOLERANCE_M
            )
        ):
            raise ValueError("Cup rigid spacing requires the unchanged inherited world-hook baseline.")
    # Validate the complete selection before mutating any builder field.
    for i in selected:
        builder.shape_margin[i] = OPTIONS["margin_m"]


def validate_cup_contact_runtime(rigid) -> dict:
    """Verify native selected rigid margins [m] before stepping, without writing solver state."""
    view = rigid.model
    if rigid.model is not view or rigid._use_mujoco_contacts:
        raise ValueError("Require the actual external-contact rigid solver and its entry view.")
    wp.synchronize_device(view.device)
    labels, bodies = list(view.shape_label), list(view.body_label)
    selected = _selected_indices(labels, 0)
    arrays = {
        "margin": np.array(view.shape_margin.numpy(), copy=True),
        "flags": np.array(view.shape_flags.numpy(), copy=True),
        "world": np.array(view.shape_world.numpy(), copy=True),
        "owner": np.array(view.shape_body.numpy(), copy=True),
        "mapping": np.array(rigid.mjc_geom_to_newton_shape.numpy(), copy=True),
        "body_mapping": np.array(rigid.mjc_body_to_newton.numpy(), copy=True),
        "geom_body": np.array(rigid.mjw_model.geom_bodyid.numpy(), copy=True),
        "native_margin": np.array(rigid.mjw_model.geom_margin.numpy(), copy=True),
    }
    for name, array in arrays.items():
        kind = "f" if name in ("margin", "native_margin") else "iu"
        if array.dtype.kind not in kind or not np.isfinite(array).all():
            raise ValueError(f"Invalid cup native readback array: {name}.")
    for name in ("margin", "flags", "world", "owner"):
        if arrays[name].shape != (len(labels),):
            raise ValueError(f"Invalid cup entry array shape: {name}.")
    ngeom, nbody = rigid.mj_model.ngeom, rigid.mj_model.nbody
    if (
        arrays["mapping"].shape != (1, ngeom)
        or arrays["native_margin"].shape != (1, ngeom)
        or arrays["body_mapping"].shape != (1, nbody)
        or arrays["geom_body"].shape != (ngeom,)
        or np.any((arrays["mapping"] < -1) | (arrays["mapping"] >= len(labels)))
        or np.any((arrays["geom_body"] < 0) | (arrays["geom_body"] >= nbody))
    ):
        raise ValueError("Require valid one-world cup native shape/body mappings.")
    rows = []
    for i in selected:
        geoms = np.flatnonzero(arrays["mapping"][0] == i)
        owner = int(arrays["owner"][i])
        flags = int(arrays["flags"][i])
        if (
            len(geoms) != 1
            or not 0 <= owner < len(bodies)
            or bodies[owner] != "/World/envs/env_0/Cup/Cup"
            or arrays["world"][i] != 0
            or not flags & int(newton.ShapeFlags.COLLIDE_SHAPES)
            or flags & int(newton.ShapeFlags.COLLIDE_PARTICLES)
        ):
            raise ValueError("Missing or invalid selected rigid cup geometry ownership/mapping.")
        g = int(geoms[0])
        if arrays["body_mapping"][0, arrays["geom_body"][g]] != owner:
            raise ValueError("Selected cup native body mapping differs from entry ownership.")
        margin, native_margin = float(arrays["margin"][i]), float(arrays["native_margin"][0, g])
        if not all(
            math.isclose(value, OPTIONS["margin_m"], rel_tol=0, abs_tol=_COMPARISON_TOLERANCE_M)
            for value in (margin, native_margin)
        ):
            raise ValueError("Selected cup Newton/native margin differs from the declared candidate.")
        rows.append(
            {
                "label": labels[i],
                "entry_shape": i,
                "native_geom": g,
                "newton_margin_m": margin,
                "native_margin_m": native_margin,
                "shape_flags": flags,
            }
        )
    return {
        "schema": "cup_contact_runtime_v1",
        "passed": True,
        "shape_count": len(rows),
        "margin_m": OPTIONS["margin_m"],
        "comparison_tolerance_m": _COMPARISON_TOLERANCE_M,
        "geometries": rows,
        "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in DEPENDENCY_PATHS},
        "physics_state_modified": False,
        "physics_steps_executed_by_check": 0,
        "scope": "Startup margin/mapping readback only; no solved-force, separation or task-success claim.",
    }
