# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""End-to-end equivalence test for the batched Newton model builder on a real task.

Creates ``Isaac-Lift-KukaAllegro-Camera`` (a camera/rendering scene) with the
``newton_mjwarp`` physics preset, then rebuilds the Newton model from the published
clone plan through both the legacy and the batched replication paths and compares
the resulting builder states and finalized models.
"""

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch the simulator
app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import pytest
import sys
import torch

import isaaclab.sim as sim_utils
from isaaclab.cloner import ClonePlan

import isaaclab_tasks  # noqa: F401
from isaaclab_newton.cloner.builder_diff import compare_builder_states, compare_finalized_models
from isaaclab_newton.cloner.newton_clone_utils import rename_builder_labels
from isaaclab_newton.cloner.replicate import _build_newton_builder_from_mapping
from isaaclab_tasks.utils import resolve_task_config

_TASK = "Isaac-Lift-KukaAllegro-Camera"
_NUM_ENVS = 4

pytestmark = [pytest.mark.integration, pytest.mark.rendering]


def _make_env(task_name: str, num_envs: int):
    """Create the task environment with the newton_mjwarp physics preset."""
    sim_utils.create_new_stage()
    # resolve_task_config reads preset tokens from sys.argv (same mechanism as the CLI).
    argv = sys.argv
    sys.argv = [argv[0], "physics=newton_mjwarp"]
    try:
        env_cfg, _ = resolve_task_config(task_name, None)
    finally:
        sys.argv = argv
    env_cfg.sim.device = "cuda"
    env_cfg.scene.num_envs = num_envs
    return gym.make(task_name, cfg=env_cfg)


def test_batched_builder_matches_legacy_on_kuka_allegro_camera():
    env = _make_env(_TASK, _NUM_ENVS)
    try:
        env.unwrapped.sim._app_control_on_stop_handle = None

        plan: ClonePlan = env.unwrapped.scene.clone_plan
        assert plan is not None and len(plan.sources) > 0, "clone plan was not published"
        stage = sim_utils.get_current_stage()

        mapping = plan.clone_mask.detach().cpu()
        env_ids = plan.env_ids.detach().cpu()
        positions = plan.positions.detach().cpu().to(torch.float32)

        def build(use_batched: bool):
            builder, _, site_index_map, world_xforms, _, bindings = _build_newton_builder_from_mapping(
                stage=stage,
                sources=tuple(plan.sources),
                destinations=tuple(plan.destinations),
                env_ids=env_ids,
                mapping=mapping,
                positions=positions,
                quaternions=None,
                use_batched_builder=use_batched,
            )
            if bindings is None:
                bindings = rename_builder_labels(builder, tuple(plan.sources), tuple(plan.destinations), env_ids, mapping)
            return builder, site_index_map, world_xforms, bindings

        legacy, legacy_sites, legacy_xforms, legacy_bindings = build(use_batched=False)
        batched, batched_sites, batched_xforms, batched_bindings = build(use_batched=True)

        errors = compare_builder_states(legacy, batched)
        assert not errors, f"builder state mismatch ({len(errors)} fields):\n" + "\n".join(errors[:40])

        assert legacy_sites == batched_sites
        assert [tuple(x) for x in legacy_xforms] == [tuple(x) for x in batched_xforms]
        assert legacy_bindings == batched_bindings

        model_errors = compare_finalized_models(legacy.finalize(device="cpu"), batched.finalize(device="cpu"))
        assert not model_errors, f"finalized model mismatch ({len(model_errors)} fields):\n" + "\n".join(
            model_errors[:40]
        )
    finally:
        if env is not None:
            env.close()
