# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Activation of authored pad spin friction, with native startup and active-contact evidence."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from typing import Any

import newton
import numpy as np
import warp as wp

PAD_SUFFIXES = ("/panda_leftfinger/left_finger_pad", "/panda_rightfinger/right_finger_pad")
BRIDGE_SUFFIX = "/Cup/Handle/Bridge1"
_MATERIAL_FIELDS = ("shape_material_mu", "shape_material_mu_torsional", "shape_material_mu_rolling")


def _indices(labels: list[str], env_id: int) -> list[int]:
    selected = []
    for suffix in (*PAD_SUFFIXES, BRIDGE_SUFFIX):
        matches = [
            i
            for i, label in enumerate(labels)
            if label.startswith(f"/World/envs/env_{env_id}/") and label.endswith(suffix)
        ]
        if len(matches) != 1:
            raise ValueError(f"Require exactly one active shape for {suffix}.")
        selected.append(matches[0])
    return selected


def apply_pad_torsional_contact(builder: Any, env_id: int) -> dict:
    """Change only two pad contact dimensions to four, preserving friction [m for spin/roll]."""
    selected = _indices(list(builder.shape_label), env_id)
    condim = builder.custom_attributes["mujoco:condim"]
    priority = builder.custom_attributes["mujoco:geom_priority"]
    if condim.default != 3 or not isinstance(condim.values, dict) or not isinstance(priority.values, dict):
        raise ValueError("Require the registered sparse contact attributes with baseline condim three.")
    rows = []
    for index in selected:
        flags = int(builder.shape_flags[index])
        friction = [float(getattr(builder, name)[index]) for name in _MATERIAL_FIELDS]
        if (
            builder.shape_world[index] != env_id
            or not flags & int(newton.ShapeFlags.COLLIDE_SHAPES)
            or flags & int(newton.ShapeFlags.COLLIDE_PARTICLES)
            or condim.values.get(index, condim.default) != 3
            or priority.values.get(index, priority.default) != 0
            or not np.isfinite(friction).all()
            or any(value < 0 for value in friction)
        ):
            raise ValueError("Require unchanged active rigid pad/Bridge1 contact baseline.")
        if index in selected[:2] and not np.allclose(friction, [1.0, 0.005, 0.0001], rtol=1e-6, atol=1e-10):
            raise ValueError("Require the pinned pad's existing authored friction values.")
        rows.append(
            {
                "label": builder.shape_label[index],
                "builder_shape": index,
                "friction": friction,
                "priority": 0,
                "before_condim": 3,
                "requested_condim": 4 if index in selected[:2] else 3,
            }
        )
    # Validate the complete selection before touching either sparse entry.
    for index in selected[:2]:
        condim.values[index] = 4
    root = Path(importlib.util.find_spec("newton").origin).parent
    dependencies = (
        Path(__file__).resolve(),
        root / "_src/solvers/mujoco/kernels.py",
        root / "_src/solvers/mujoco/solver_mujoco.py",
    )
    return {
        "schema": "pad_torsional_contact_v1",
        "geometries": rows,
        "native_verified": False,
        "active_bilateral_verified": False,
        "changed_field": "mujoco:condim",
        "changed_shape_count": 2,
        "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in dependencies},
        "scope": "Rigid pad contact formulation only; no full-task or physical success claim.",
    }


def _array(value: Any) -> np.ndarray:
    return np.array(value.numpy() if hasattr(value, "numpy") else value, copy=True)


def verify_pad_torsional_contact(rigid: Any, receipt: dict) -> None:
    """Verify imported, compiled and native pad contact settings before issuing robot actions."""
    view = rigid.model
    if rigid.model is not view or rigid._use_mujoco_contacts or int(rigid.mjw_model.opt.cone) != 1:
        raise ValueError("Require the actual external-contact elliptic rigid solver.")
    wp.synchronize_device(view.device)
    labels, bodies = list(view.shape_label), list(view.body_label)
    selected = _indices(labels, 0)
    mapping = _array(rigid.mjc_geom_to_newton_shape)
    body_mapping = _array(rigid.mjc_body_to_newton)
    owner = _array(view.shape_body)
    native = rigid.mjw_model
    geom_owner = _array(native.geom_bodyid)
    n = rigid.mj_model.ngeom
    if mapping.shape != (1, n) or body_mapping.shape != (1, rigid.mj_model.nbody) or geom_owner.shape != (n,):
        raise ValueError("Require valid single-world native shape/body mappings.")
    imported_dim, imported_priority = _array(view.mujoco.condim), _array(view.mujoco.geom_priority)
    imported_friction = np.stack([_array(getattr(view, name)) for name in _MATERIAL_FIELDS], axis=-1)
    native_dim, native_priority, native_friction = (
        _array(getattr(native, name)) for name in ("geom_condim", "geom_priority", "geom_friction")
    )
    if native_dim.shape != (n,) or native_priority.shape != (n,) or native_friction.shape != (1, n, 3):
        raise ValueError("Invalid native pad contact readback array dimensions.")
    for index, row in zip(selected, receipt["geometries"], strict=True):
        geoms = np.flatnonzero(mapping[0] == index)
        if len(geoms) != 1 or row["label"] != labels[index]:
            raise ValueError("Missing, duplicated or mismatched pad/Bridge1 geometry mapping.")
        geom = int(geoms[0])
        if (
            not 0 <= owner[index] < len(bodies)
            or not 0 <= geom_owner[geom] < rigid.mj_model.nbody
            or body_mapping[0, geom_owner[geom]] != owner[index]
        ):
            raise ValueError("Native pad/Bridge1 geometry body ownership differs from the entry view.")
        states = {
            "imported": {
                "condim": int(imported_dim[index]),
                "priority": int(imported_priority[index]),
                "friction": imported_friction[index].tolist(),
            },
            "compiled": {
                "condim": int(rigid.mj_model.geom_condim[geom]),
                "priority": int(rigid.mj_model.geom_priority[geom]),
                "friction": rigid.mj_model.geom_friction[geom].tolist(),
            },
            "native": {
                "condim": int(native_dim[geom]),
                "priority": int(native_priority[geom]),
                "friction": native_friction[0, geom].tolist(),
            },
        }
        row.update(entry_shape=index, native_geom=geom, body_label=bodies[int(owner[index])], readback=states)
        if any(
            state["condim"] != row["requested_condim"]
            or state["priority"] != 0
            or not np.allclose(state["friction"], row["friction"], rtol=1e-6, atol=1e-10)
            for state in states.values()
        ):
            raise ValueError(f"Pad/Bridge1 contact native mismatch: {row}")
    receipt["native_verified"] = True


def capture_pad_bridge_contacts(rigid: Any, receipt: dict, step: int) -> dict:
    """Require actual bilateral pad/Bridge1 spin constraint rows at the measured held step."""
    if not receipt["native_verified"]:
        raise ValueError("Require startup native verification before recording active contacts.")
    wp.synchronize_device(rigid.model.device)
    data, contact = rigid.mjw_data, rigid.mjw_data.contact
    count_array, nefc = _array(data.nacon), _array(data.nefc)
    arrays = {
        name: _array(getattr(contact, name)) for name in ("geom", "worldid", "dim", "friction", "efc_address", "dist")
    }
    if count_array.shape != (1,) or nefc.shape != (1,):
        raise ValueError("Require single-world active-contact counts.")
    count = int(count_array[0])
    if not 0 <= count <= len(arrays["geom"]) or any(len(value) != len(arrays["geom"]) for value in arrays.values()):
        raise ValueError("Invalid active-contact count or array capacities.")
    bridge = receipt["geometries"][2]
    rows = []
    for pad in receipt["geometries"][:2]:
        wanted = {pad["native_geom"], bridge["native_geom"]}
        resolved = np.maximum(pad["friction"], bridge["friction"])
        expected = np.maximum(resolved[[0, 0, 1, 2, 2]], 1e-5)
        active = []
        for index in range(count):
            if arrays["worldid"][index] != 0 or set(arrays["geom"][index].tolist()) != wanted:
                continue
            addresses = arrays["efc_address"][index, :4]
            if np.all(arrays["efc_address"][index] < 0):
                continue
            if (
                arrays["dim"][index] != 4
                or len(addresses) != 4
                or np.any(addresses < 0)
                or np.any(addresses >= nefc[0])
                or len(set(addresses.tolist())) != 4
                or not np.allclose(arrays["friction"][index], expected, rtol=1e-6, atol=1e-10)
                or not np.isfinite(arrays["dist"][index])
            ):
                raise ValueError(
                    "Active pad/Bridge1 contact does not carry the expected four-dimensional friction constraint."
                )
            active.append(
                {
                    "contact_index": index,
                    "geom": arrays["geom"][index].tolist(),
                    "dim": int(arrays["dim"][index]),
                    "friction": arrays["friction"][index].tolist(),
                    "efc_address": addresses.tolist(),
                    "distance_m": float(arrays["dist"][index]),
                }
            )
        if not active:
            raise ValueError(f"No active four-dimensional pad/Bridge1 contact for {pad['label']} at held step {step}.")
        rows.append({"pad_label": pad["label"], "bridge_label": bridge["label"], "contacts": active})
    receipt["active_bilateral_verified"] = True
    receipt["active_contact_evidence"] = {
        "step": step,
        "nacon": count,
        "nefc": int(nefc[0]),
        "pairs": rows,
        "scope": "Actual active constraint rows; no solved-force or task-success claim.",
    }
    return receipt["active_contact_evidence"]
