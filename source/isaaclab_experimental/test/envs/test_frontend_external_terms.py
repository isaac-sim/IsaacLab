# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp terms from an external project run on the Warp frontend unchanged.

The project lives outside the packages the frontend mirrors, so its terms have no twins. Its Warp terms
must be used as-is, even when one shares its name with a built-in term, while a Torch term from the same
project must still be rejected instead of being replaced by a same-named built-in twin.
"""

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

from isaaclab.sim import SimulationCfg
from isaaclab.test.utils import launch_test_simulation

launch_test_simulation(SimulationCfg(physics=NewtonCfg(solver_cfg=MJWarpSolverCfg())))

import importlib
import textwrap

import pytest
import torch
from isaaclab_experimental.envs.frontend import WarpFrontend

import isaaclab.sim as sim_utils
from isaaclab.managers import RewardTermCfg
from isaaclab.utils import configclass

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import resolve_task_config

_NUM_ENVS = 4

_EXTERNAL_MDP = textwrap.dedent('''
    import torch
    import warp as wp

    from isaaclab_experimental.managers import ManagerTermBase
    from isaaclab_experimental.utils.warp import WarpCapturable


    @wp.kernel
    def _fill(value: float, out: wp.array(dtype=wp.float32)):
        out[wp.tid()] = value


    @WarpCapturable(True)
    def joint_vel_l2(env, out) -> None:
        """A Warp reward sharing its name with the built-in ``joint_vel_l2``."""
        wp.launch(_fill, dim=env.num_envs, inputs=[2.0, out], device=env.device)


    class constant_bonus(ManagerTermBase):
        """A Warp class reward."""

        def __call__(self, env, out, value: float) -> None:
            wp.launch(_fill, dim=env.num_envs, inputs=[value, out], device=env.device)


    def joint_vel_l1(env) -> torch.Tensor:
        """A Torch reward sharing its name with the built-in ``joint_vel_l1``."""
        return torch.zeros(env.num_envs, device=env.device)
''')


@pytest.fixture
def external_mdp(tmp_path, monkeypatch):
    """An external project package on the import path."""
    package = tmp_path / "my_warp_project"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "mdp.py").write_text(_EXTERNAL_MDP)
    monkeypatch.syspath_prepend(str(tmp_path))
    return importlib.import_module("my_warp_project.mdp")


def _cartpole_cfg(rewards):
    env_cfg, _ = resolve_task_config("Isaac-Cartpole", "", overrides=("physics=newton_mjwarp",))
    env_cfg.seed = 7
    env_cfg.scene.num_envs = _NUM_ENVS
    env_cfg.rewards = rewards
    return env_cfg


def test_external_warp_terms_run_as_declared(external_mdp):
    @configclass
    class RewardsCfg:
        function_term = RewardTermCfg(func=external_mdp.joint_vel_l2, weight=1.0)
        class_term = RewardTermCfg(func=external_mdp.constant_bonus, weight=0.5, params={"value": 3.0})

    env_cfg = _cartpole_cfg(RewardsCfg())
    sim_utils.create_new_stage()
    env = WarpFrontend.build_env(env_cfg, "Isaac-Cartpole").unwrapped
    try:
        env.reset()
        actions = torch.zeros((_NUM_ENVS, env.action_space.shape[-1]), device=env.device)
        for _ in range(3):
            _, reward, _, _, _ = env.step(actions)

        assert env.reward_manager.get_term_cfg("function_term").func is external_mdp.joint_vel_l2
        expected = (1.0 * 2.0 + 0.5 * 3.0) * env.step_dt
        assert torch.allclose(reward, torch.full_like(reward, expected))
    finally:
        env.close()


def test_external_torch_term_is_rejected(external_mdp):
    @configclass
    class RewardsCfg:
        torch_term = RewardTermCfg(func=external_mdp.joint_vel_l1, weight=1.0)

    reason = WarpFrontend.check_compatibility(_cartpole_cfg(RewardsCfg()))

    assert reason is not None and "'joint_vel_l1' from 'my_warp_project.mdp' is not a warp term" in reason
