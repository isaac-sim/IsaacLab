# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for shoelace runtime configuration and randomized resets."""

from types import SimpleNamespace

import pytest
import torch

from isaaclab_tasks.contrib.shoelace import shoelace_constants as constants
from isaaclab_tasks.contrib.shoelace.mdp.events import reset_shoe_position
from isaaclab_tasks.contrib.shoelace.shoelace_env_cfg import ShoelaceEnvCfg
from isaaclab_tasks.contrib.shoelace.shoelace_physics import create_shoelace_env


@pytest.mark.parametrize("env_ids", [slice(None), slice(1, 4, 2), slice(0, 0), torch.tensor([3, 1])])
def test_shoe_reset_accepts_slices_and_preserves_unselected_poses(env_ids: torch.Tensor | slice) -> None:
    """Apply the same translation to selected shoe and cable poses without expanding slices."""
    defaults = torch.zeros(4, 7)
    defaults[:, 6] = 1.0
    origins = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    shoe_pose = defaults.clone()
    cable_pose = defaults[:, None].expand(-1, 3, -1).clone()
    shoe = SimpleNamespace(
        data=SimpleNamespace(default_root_pose=SimpleNamespace(torch=defaults)),
        write_root_pose_to_sim_index=lambda *, root_pose, env_ids: shoe_pose.__setitem__(env_ids, root_pose),
    )
    cable = SimpleNamespace(
        data=SimpleNamespace(default_segment_pose_w=SimpleNamespace(torch=cable_pose.clone())),
        write_segment_pose_to_sim_index=lambda *, segment_pose, env_ids: cable_pose.__setitem__(env_ids, segment_pose),
    )

    class Scene(dict):
        env_origins = origins

    env = SimpleNamespace(device="cpu", scene=Scene(shoe=shoe, shoelace_left=cable, shoelace_right=cable))
    reset_shoe_position(env, env_ids, {"x": (0.1, 0.1), "y": (-0.2, -0.2)})
    expected_shoe = defaults.clone()
    expected_cable = defaults[:, None].expand(-1, 3, -1).clone()
    offset = torch.tensor([0.1, -0.2, 0.0])
    expected_shoe[env_ids, :3] += origins[env_ids] + offset
    expected_cable[env_ids, :, :3] += offset
    torch.testing.assert_close(shoe_pose, expected_shoe)
    torch.testing.assert_close(cable_pose, expected_cable)


@pytest.mark.parametrize(("num_envs", "override"), [(4, None), (1024, None), (1024, 1_024_000)])
def test_runtime_contact_capacity_scales_and_preserves_overrides(num_envs: int, override: int | None) -> None:
    """Grow collision storage while keeping matching indices encodable and honoring larger overrides."""
    cfg = ShoelaceEnvCfg()
    cfg.scene.num_envs = num_envs
    outer_override = 16_777_216 if override is not None else 0
    if override is not None:
        cfg.sim.physics.collision_cfg.max_triangle_pairs = outer_override
        cfg.sim.physics.solver_cfg.contact_max_triangle_pairs = override
        cfg.sim.physics.solver_cfg.contact_reduction_hashtable_size_factor = 4.0
    cfg.validate()

    collision = cfg.sim.physics.collision_cfg
    solver = cfg.sim.physics.solver_cfg
    assert collision.max_triangle_pairs >= constants.TRIANGLE_PAIRS_PER_ENV * num_envs
    assert 0 < solver.contact_max_triangle_pairs < 2**20
    if override is not None:
        assert collision.max_triangle_pairs == outer_override
        assert solver.contact_max_triangle_pairs == override
        assert solver.contact_reduction_hashtable_size_factor == 4.0
    elif num_envs == 4:
        assert solver.contact_max_triangle_pairs == collision.max_triangle_pairs
    else:
        assert solver.contact_max_triangle_pairs < collision.max_triangle_pairs
        assert solver.contact_reduction_hashtable_size_factor > 0.25


@pytest.mark.parametrize("regularization", [-1.0e-6, float("nan"), float("inf")])
def test_invalid_proxy_inertia_is_rejected_before_environment_construction(
    regularization: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject invalid builder regularization at the factory entry before initializing an environment."""
    cfg = ShoelaceEnvCfg()
    cfg.cable_inertia_regularization = regularization
    monkeypatch.setattr(
        "isaaclab_tasks.contrib.shoelace.shoelace_physics.ManagerBasedRLEnv",
        lambda **kwargs: pytest.fail("Invalid regularization must not initialize an environment"),
    )
    with pytest.raises(ValueError, match="finite and nonnegative"):
        create_shoelace_env(cfg=cfg)
