# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for Newton passive joint damping."""

from __future__ import annotations

import sys
import types

import torch
import warp as wp
from newton import ModelBuilder

import isaaclab.cloner as cloner

# The installed Newton package resolves to the main checkout in this isolated
# worktree. Supply only the two compatibility symbols needed to import the
# worktree's production articulation module for this CPU-only unit test.
if not hasattr(cloner, "queue_usd_replication"):
    cloner.queue_usd_replication = lambda cfg: None
if "isaaclab.cloner.cloner_utils" not in sys.modules:
    cloner_utils = types.ModuleType("isaaclab.cloner.cloner_utils")
    cloner_utils.replace_path_prefix = lambda path, source, destination: path.replace(source, destination, 1)
    sys.modules[cloner_utils.__name__] = cloner_utils

from isaaclab_newton.assets.articulation.articulation import Articulation
from isaaclab_newton.assets.articulation.articulation_data import ArticulationData
from isaaclab_newton.physics import NewtonManager
from isaaclab.utils.warp.proxy_array import ProxyArray


def test_viscous_writer_updates_finalized_newton_model(monkeypatch):
    """The production viscous writer updates a finalized Newton model binding."""
    builder = ModelBuilder()
    link = builder.add_link(mass=1.0, inertia=wp.mat33(1.0))
    joint = builder.add_joint_revolute(-1, link, label="joint")
    builder.add_articulation([joint], label="articulation")
    model = builder.finalize(device="cpu")
    model_damping = wp.array(
        ptr=model.joint_damping.ptr,
        dtype=wp.float32,
        shape=(1, 1),
        strides=(model.joint_damping.strides[0], model.joint_damping.strides[0]),
        device="cpu",
        copy=False,
    )

    data_type = type(
        "_Data",
        (),
        {"joint_viscous_friction_coeff": ArticulationData.joint_viscous_friction_coeff},
    )
    data = data_type()
    data.has_joint_ordering = False
    data.joint_ordering = None
    data._joint_viscous_friction_user = None
    data._sim_bind_joint_viscous_friction_coeff = model_damping
    data._joint_viscous_friction_coeff_ta = ProxyArray(model_damping)

    articulation = object.__new__(Articulation)
    articulation._device = "cpu"
    articulation._data = data
    articulation._ALL_INDICES = wp.array([0], dtype=wp.int32, device="cpu")
    articulation._ALL_JOINT_INDICES = wp.array([0], dtype=wp.int32, device="cpu")
    articulation._initialize_handle = None
    articulation._invalidate_initialize_handle = None
    articulation._prim_deletion_handle = None
    monkeypatch.setattr(NewtonManager, "add_model_change", lambda flags: None)

    articulation.write_joint_viscous_friction_coefficient_to_sim_index(
        joint_viscous_friction_coeff=torch.tensor([[0.25]], dtype=torch.float32),
    )

    torch.testing.assert_close(data.joint_viscous_friction_coeff.torch, torch.tensor([[0.25]]))
    torch.testing.assert_close(torch.from_numpy(model.joint_damping.numpy()), torch.tensor([0.25]))
