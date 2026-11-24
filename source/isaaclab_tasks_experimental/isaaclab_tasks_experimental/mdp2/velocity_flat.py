# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared MDP2 configuration for flat-ground velocity tracking."""

from __future__ import annotations

from dataclasses import MISSING
from typing import Any

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
from isaaclab_tasks.mdp2.velocity_go2.flat_env_cfg import TASK as GO2_TASK
from isaaclab_tasks.utils import PresetCfg

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


def flat_velocity_task(base_body_name: str, *, action_rate_weight: float) -> mdp2.TaskCfg:
    """Main's Unitree Go2 task for another robot: its base body and action rate weight, the command that steers when
    it resamples, and no base height penalty, which is specific to Go2."""
    base = robot.bodies(base_body_name)
    base_contact = contact_forces.bodies(base_body_name)
    command = GO2_TASK.commands["base_velocity"]
    rewards = {name: reward for name, reward in GO2_TASK.rewards.items() if name != "base_height_l2"}
    contact = GO2_TASK.terminations["base_contact"]
    mass, com = GO2_TASK.events["add_base_mass"], GO2_TASK.events["base_com"]
    return replace(
        GO2_TASK,
        commands={"base_velocity": replace(command, term=mdp.heading_velocity_command_steered_on_resample)},
        rewards={**rewards, "action_rate_l2": replace(rewards["action_rate_l2"], weight=action_rate_weight)},
        terminations={
            **GO2_TASK.terminations,
            "base_contact": replace(contact, inputs={**contact.inputs, "body_ids": base_contact}),
        },
        events={
            **GO2_TASK.events,
            "add_base_mass": replace(mass, inputs={**mass.inputs, "body_ids": base}),
            "base_com": replace(com, inputs={**com.inputs, "body_ids": base}),
        },
    )


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


def flat_velocity_sim(**solver: Any) -> sim_utils.SimulationCfg:
    """The flat velocity simulation with a robot's MJWarp solver settings, such as its constraint capacities."""
    newton_mjwarp = FlatVelocityPhysicsCfg().newton_mjwarp
    newton_mjwarp = replace(newton_mjwarp, solver_cfg=replace(newton_mjwarp.solver_cfg, **solver))
    return replace(
        FlatVelocityEnvCfg().sim, physics=FlatVelocityPhysicsCfg(newton_mjwarp=newton_mjwarp, default=newton_mjwarp)
    )


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
