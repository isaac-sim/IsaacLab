# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared MDP2 configuration for flat-ground velocity tracking."""

from __future__ import annotations

from dataclasses import MISSING

from isaaclab_experimental import mdp2
from isaaclab_experimental.envs import MDP2RLEnvCfg
from isaaclab_experimental.mdp2 import mdp
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg, NewtonShapeCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass, replace

from isaaclab_tasks.core.velocity.velocity_env_cfg import MySceneCfg
from isaaclab_tasks.utils import PresetCfg

from .velocity_go2.flat_env_cfg import TASK as GO2_TASK

SIM_DT = 0.005
DECIMATION = 4
GROUND_MATERIAL = sim_utils.PhysxRigidBodyMaterialCfg(
    friction_combine_mode="multiply", restitution_combine_mode="multiply", static_friction=1.0, dynamic_friction=1.0
)

robot = mdp2.Entity("robot")
contact_forces = mdp2.Entity("contact_forces")
scene = mdp2.Entity("scene")
joints = robot.joints(".*")

root_state = {"root_link_pose_w": robot.root_link_pose_w, "root_com_vel_w": robot.root_com_vel_w}
velocity_tracking = {**root_state, "command": mdp2.commands.base_velocity}


@configclass
class FlatVelocitySceneCfg(InteractiveSceneCfg):
    """A robot on a ground plane with contact sensors on all its bodies."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=GROUND_MATERIAL,
    )
    robot: ArticulationCfg = MISSING
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/[^/]*", history_length=3, track_air_time=True, update_period=SIM_DT
    )
    sky_light = AssetBaseCfg(prim_path="/World/skyLight", spawn=sim_utils.DomeLightCfg(intensity=750.0))


@configclass
class FlatVelocityPhysicsCfg(PresetCfg):
    """Newton physics preset for flat velocity tracking."""

    newton_mjwarp: NewtonCfg = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            njmax=65,
            nconmax=35,
            cone="pyramidal",
            impratio=1.0,
            integrator="implicitfast",
            use_mujoco_contacts=False,
        ),
        collision_cfg=NewtonCollisionPipelineCfg(max_triangle_pairs=2_500_000),
        num_substeps=2,
        debug_mode=False,
        default_shape_cfg=NewtonShapeCfg(margin=0.0, ke=160000.0, kd=1100.0),
    )
    default: NewtonCfg = newton_mjwarp


FLAT_VELOCITY_TASK = replace(
    GO2_TASK,
    commands={
        "base_velocity": replace(
            GO2_TASK.commands["base_velocity"], term=mdp.heading_velocity_command_steered_on_resample
        ),
    },
    rewards={**GO2_TASK.rewards, "base_height_l2": None},
)
"""Main's Unitree Go2 task as the base of the other robots, as the manager velocity tasks share their base config: the
command that steers when it resamples, and no base height penalty, which is specific to Go2. Robots whose base body is
not ``base`` set it in the base contact termination and the base mass and center of mass events, and robots set their
action rate weight."""


@configclass
class FlatVelocityEnvCfg(MDP2RLEnvCfg):
    """Configuration for flat-ground velocity tracking."""

    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=SIM_DT,
        render_interval=DECIMATION,
        physics=FlatVelocityPhysicsCfg(),
        physics_material=GROUND_MATERIAL,
        use_newton_actuators=True,
    )
    scene: FlatVelocitySceneCfg = FlatVelocitySceneCfg(num_envs=4096, env_spacing=2.5)
    decimation: int = DECIMATION

    def play_mode(self):
        """Play without the random pushes, as the MDP1 velocity tasks do."""
        super().play_mode()
        tasks = [self.task] if isinstance(self.task, mdp2.TaskCfg) else self.task
        tasks = [replace(task, events={**task.events, "push_robot": None}) for task in tasks]
        self.task = tasks[0] if isinstance(self.task, mdp2.TaskCfg) else tasks


_ROUGH_SCENE = MySceneCfg(num_envs=4096, env_spacing=2.5)
FLAT_TERRAIN_SCENE = replace(
    _ROUGH_SCENE,
    terrain=replace(_ROUGH_SCENE.terrain, terrain_type="plane", terrain_generator=None),
    height_scanner=None,
    contact_forces=replace(_ROUGH_SCENE.contact_forces, update_period=SIM_DT),
)
"""The manager velocity tasks' scene on a plane, with their terrain material and sky, without the height scanner.

Tasks set its robot.
"""
