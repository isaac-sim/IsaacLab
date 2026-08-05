# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch the simulator
simulation_app = AppLauncher(headless=True, enable_cameras=True).app


"""Rest everything follows."""

import pytest
import torch

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.app.settings_manager import get_settings_manager
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.utils.configclass import configclass

from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def _configure_sim():
    # Prevents a bug where the simulation gets stuck randomly on many environments.
    get_settings_manager().set_bool("/physics/cooking/ujitsoCollisionCooking", False)


def _ee_frame_pos(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
    """Return end-effector position in the base (source) frame from the FrameTransformer sensor."""
    return env.scene[sensor_cfg.name].data.target_pos_source[:, 0, :]


@configclass
class _SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    robot: ArticulationCfg = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
        target_frames=[FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Robot/panda_hand", name="hand")],
    )


@configclass
class _ObsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        eef_pos = ObsTerm(func=_ee_frame_pos)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class _ActionsCfg:
    arm = mdp.JointPositionActionCfg(asset_name="robot", joint_names=["panda_joint.*"], scale=0.5)


@configclass
class _EventsCfg:
    reset_robot = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class _RewardsCfg:
    alive = RewTerm(func=mdp.is_alive, weight=1.0)


@configclass
class _TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class _FrankaFrameTransformerEnvCfg(ManagerBasedRLEnvCfg):
    scene: _SceneCfg = _SceneCfg(num_envs=1, env_spacing=2.5)
    observations: _ObsCfg = _ObsCfg()
    actions: _ActionsCfg = _ActionsCfg()
    events: _EventsCfg = _EventsCfg()
    rewards: _RewardsCfg = _RewardsCfg()
    terminations: _TerminationsCfg = _TerminationsCfg()

    def __post_init__(self) -> None:
        self.decimation = 2
        self.episode_length_s = 5.0
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("num_envs", [1, 2])
@pytest.mark.isaacsim_ci
def test_eef_pos_not_stale_after_reset(device, num_envs):
    """Check that FrameTransformer eef_pos observation is not stale on the first step after reset."""
    sim_utils.create_new_stage()

    env_cfg = _FrankaFrameTransformerEnvCfg()
    env_cfg.sim.device = device
    env_cfg.scene.num_envs = num_envs
    env = ManagerBasedRLEnv(cfg=env_cfg)
    env.sim._app_control_on_stop_handle = None

    obs = env.reset()[0]
    pre_reset_eef_pos = obs["policy"]["eef_pos"].clone()
    print(pre_reset_eef_pos)

    idle_actions = torch.zeros(env.action_space.shape, device=env.device)
    obs = env.step(idle_actions)[0]

    post_reset_eef_pos = obs["policy"]["eef_pos"]
    print(post_reset_eef_pos)

    torch.testing.assert_close(pre_reset_eef_pos, post_reset_eef_pos, atol=1e-5, rtol=1e-3)

    env.close()
