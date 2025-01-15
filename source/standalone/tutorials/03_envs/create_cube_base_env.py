# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script creates a simple environment with a floating cube. The cube is controlled by a PD
controller to track an arbitrary target position.

While going through this tutorial, we recommend you to pay attention to how a custom action term
is defined. The action term is responsible for processing the raw actions and applying them to the
scene entities. The rest of the environment is similar to the previous tutorials.

.. code-block:: bash

    # Run the script
    ./isaaclab.sh -p source/standalone/tutorials/04_envs/floating_cube.py --num_envs 32
"""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""


import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on creating a floating cube environment.")
parser.add_argument("--num_envs", type=int, default=100, help="Number of environments to spawn.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import numpy as np
import torch

from source.utils.data_utils import store_h5_dict

import omni.isaac.lab.envs.mdp as mdp
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import AssetBaseCfg, RigidObject, RigidObjectCfg
from omni.isaac.lab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from omni.isaac.lab.managers import ActionTerm, ActionTermCfg
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Custom action term
##


class CubeActionTerm(ActionTerm):
    """Simple action term that implements a PD controller to track a target position.

    The action term is applied to the cube asset. It involves two steps:

    1. **Process the raw actions**: Typically, this includes any transformations of the raw actions
       that are required to map them to the desired space. This is called once per environment step.
    2. **Apply the processed actions**: This step applies the processed actions to the asset.
       It is called once per simulation step.

    In this case, the action term simply applies the raw actions to the cube asset. The raw actions
    are the desired target positions of the cube in the environment frame. The pre-processing step
    simply copies the raw actions to the processed actions as no additional processing is required.
    The processed actions are then applied to the cube asset by implementing a PD controller to
    track the target position.
    """

    _asset: RigidObject
    """The articulation asset on which the action term is applied."""

    def __init__(self, cfg: CubeActionTermCfg, env: ManagerBasedEnv):
        # call super constructor
        super().__init__(cfg, env)
        # create buffers
        self._raw_actions = torch.zeros(env.num_envs, 6, device=self.device)
        self._processed_actions = torch.zeros(env.num_envs, 6, device=self.device)
        self._vel_command = torch.zeros(self.num_envs, 6, device=self.device)
        # gains of controller
        self.p_gain = cfg.p_gain
        self.d_gain = cfg.d_gain

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return self._raw_actions.shape[1]

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    """
    Operations
    """

    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        self._raw_actions[:] = actions
        # no-processing of actions
        self._processed_actions[:] = self._raw_actions[:]

    def apply_actions(self):
        # implement a PD controller to track the target position
        pos_error = self._processed_actions - (self._asset.data.root_link_pos_w - self._env.scene.env_origins)
        vel_error = -self._asset.data.root_com_lin_vel_w
        # set velocity targets
        self._vel_command[:, :3] = self.p_gain * pos_error + self.d_gain * vel_error
        self._asset.write_root_com_velocity_to_sim(self._vel_command)


@configclass
class CubeActionTermCfg(ActionTermCfg):
    """Configuration for the cube action term."""

    class_type: type = CubeActionTerm
    """The class corresponding to the action term."""

    p_gain: float = 5.0
    """Proportional gain of the PD controller."""
    d_gain: float = 0.5
    """Derivative gain of the PD controller."""


##
# Custom observation term
##


def base_position(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Root linear velocity in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_link_pos_w - env.scene.env_origins


##
# Scene definition
##


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Example scene configuration.

    The scene comprises of a ground plane, light source and floating cubes (gravity disabled).
    """

    # add terrain
    terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane", debug_vis=False)

    # add cube
    # cube: RigidObjectCfg = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/cube",
    #     spawn=sim_utils.CuboidCfg(
    #         size=(0.2, 0.2, 0.2),
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0, disable_gravity=True),
    #         mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
    #         physics_material=sim_utils.RigidBodyMaterialCfg(),
    #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.0, 0.0)),
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 5)),
    # )
    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1, 0, 0, 0)),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            scale=(0.8, 0.8, 0.8),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_depenetration_velocity=1.0,
                disable_gravity=False,
            ),
        ),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# Environment settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = CubeActionTermCfg(asset_name="cube")


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # cube velocity
        position = ObsTerm(func=base_position, params={"asset_cfg": SceneEntityCfg("cube")})
        sys_params = ObsTerm(func=sys_params, params={"asset_cfg": SceneEntityCfg("cube")})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.002, 0.002),
                "y": (-0.002, 0.002),
                "z": (-0.002, 0.002),
            },
            "asset_cfg": SceneEntityCfg("cube"),
        },
    )
    cube_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        min_step_count_between_reset=100,
        params={
            "asset_cfg": SceneEntityCfg("cube", body_names=".*"),
            "static_friction_range": (0.7, 1.3),
            "dynamic_friction_range": (0.7, 1.3),
            "restitution_range": (1.0, 1.0),
            "num_buckets": 250,
        },
    )

    cube_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("cube", body_names=".*"),
            "mass_distribution_params": (0.5, 2),
            "operation": "scale",
        },
    )


##
# Environment configuration
##


@configclass
class CubeEnvCfg(ManagerBasedEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=args_cli.num_envs, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        # simulation settings
        self.sim.dt = 0.01
        self.sim.physics_material = self.scene.terrain.physics_material


def get_random_actions(env: ManagerBasedEnv, episode_length: int) -> torch.Tensor:
    num_envs = env.num_envs
    force_range = (0.1, 1.5)
    torque_range = (0.01, 0.1)
    # generates random force torque trajectories using uniform distribution
    forces = (
        torch.rand(num_envs, episode_length, 3, device=env.device) * (force_range[1] - force_range[0]) + force_range[0]
    )
    torques = (
        torch.rand(num_envs, episode_length, 3, device=env.device) * (torque_range[1] - torque_range[0])
        + torque_range[0]
    )
    torques *= 0.1
    actions = torch.cat([forces, torques], dim=-1)
    # smooth the actions using gaussian filters
    # actions = torch.nn.functional.conv1d(actions, torch.ones(1, 1, 5, device=env.device) / 5, padding=2)
    return actions


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Converts a torch tensor to a numpy array."""
    return tensor.detach().cpu().numpy()


def main():
    """Main function."""
    episode_length = 200
    ds_length = 10000
    # setup base environment
    env = ManagerBasedEnv(cfg=CubeEnvCfg())

    # setup target position commands
    target_position = torch.rand(env.num_envs, 3, device=env.device) * 2
    target_position[:, 2] = 0
    # offset all targets so that they move to the world origin
    # target_position -= env.scene.env_origins

    # simulate physics
    count = 0
    obs, _ = env.reset()
    action_trajs = None
    # [s0, a0, r0, ]
    all_trajs = {"cur_state": [], "action": [], "is_slip": []}
    while simulation_app.is_running():
        with torch.inference_mode():
            # reset
            if count % episode_length == 0:
                count = 0
                obs, _ = env.reset()
                print("-" * 80)
                print("[INFO]: Resetting environment...")
                action_trajs = get_random_actions(env, episode_length)

            # step env
            cur_state = to_numpy(obs["policy"])
            action = action_trajs[:, count]
            obs, _ = env.step(action)
            new_state = to_numpy(obs["policy"])
            is_slip = np.any(new_state[:, 7:13] > 0.03, axis=-1)
            print(f"Step: {count}, Slip: {is_slip.mean()}")
            all_trajs["cur_state"].append(cur_state)
            all_trajs["action"].append(to_numpy(action))
            all_trajs["is_slip"].append(is_slip)

            # update counter
            count += 1
            if len(all_trajs["cur_state"]) == ds_length:
                break
    for k, v in all_trajs.items():
        all_trajs[k] = np.concatenate(v)
    store_h5_dict("data.h5", all_trajs)
    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
