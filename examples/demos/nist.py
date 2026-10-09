# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run a ten-part NIST assembly policy without a trainer, curriculum, or reset bank.

    uv run --extra ovrtx isaaclab demo nist --policy <exported-policy.pt>

The exported policy includes normalization and takes robot history [N, 195] and
slot history [N, 5, 10, 40]. At startup, the seed selects ten distinct asset
identities from the original 19-asset catalog. The selection stays fixed across
all environments and episodes. Each reset randomizes the board
and drops five parts in each of two side strips, with all ten parts unfinished
and the robot at home. There is no reset bank or grasp-state initialization.
The default RTX view uses a physical ground plane without a backdrop.
Use --num_envs to run independent boards in parallel, each resetting on completion or timeout.
Physics remains at 100 Hz and control at 25 Hz.
Use --visualizer none --episodes 1 for a headless smoke test.
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
from typing import NamedTuple

import torch
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg
from isaaclab_newton.sim.schemas import MujocoRigidBodyCfg, NewtonArticulationCfg
from isaaclab_newton.sim.spawners.materials import NewtonMaterialCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg, mdp
from isaaclab.managers import EventTermCfg, ManagerTermBase, ObservationGroupCfg, ObservationTermCfg, SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.materials import UsdPhysicsRigidBodyMaterialCfg
from isaaclab.utils import math as math_utils
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass


class Assembly(NamedTuple):
    """Static asset metadata: positions [m], masses [kg], xyzw poses. Local frame rotations are identity."""

    identity: int
    name: str
    plug: str
    socket: str
    board_pose: tuple[float, ...]
    held_align: tuple[float, float, float]
    fixed_tip: tuple[float, float, float]
    seated: tuple[float, float, float]
    plug_mass: float
    socket_mass: float = 0.05


# Identity order is part of checkpoint 6xanrgn0's input contract, even with only ten bodies present.
# fmt: off
ASSEMBLIES = (
    Assembly(0, "nut_thread_m8", "nut_m8", "bolt_m8",
             (0.0473, -0.0407, 0.0172, 1.0, 0.0, 0.0, 0.0),
             (0.0, 0.0, 0.0093), (0.0, 0.0, 0.026), (0.0, 0.0, 0.018), 0.03),
    Assembly(1, "nut_thread_m12", "nut_m12", "bolt_m12",
             (0.3473, -0.2665, 0.0212, 1.0, 0.0, 0.0, 0.0),
             (0.0, 0.0, 0.013), (0.0, 0.0, 0.035), (0.0, 0.0, 0.0218), 0.03),
    Assembly(2, "nut_thread_m16", "nut_m16", "bolt_m16",
             (0.04715, -0.3416, 0.0194, 1.0, 0.0, 0.0, 0.0),
             (0.0, 0.0, 0.01), (0.0, 0.0, 0.035), (0.0, 0.0, 0.022), 0.03),
    Assembly(3, "gear_mesh_small", "gear_small", "gear_base",
             (0.0474, -0.1713, -0.0002, 0.7071, -0.7071, 0.0, 0.0),
             (0.05075, 0.0, 0.005), (0.0508, 0.0, 0.025), (0.05075, 0.0, 0.005), 0.019),
    Assembly(4, "gear_mesh_medium", "gear_medium", "gear_base",
             (0.0474, -0.1713, -0.0002, 0.7071, -0.7071, 0.0, 0.0),
             (0.02025, 0.0, 0.005), (0.02025, 0.0, 0.025), (0.02025, 0.0, 0.005), 0.012),
    Assembly(5, "gear_mesh_large", "gear_large", "gear_base",
             (0.0474, -0.1713, -0.0002, 0.7071, -0.7071, 0.0, 0.0),
             (-0.0303, 0.0, 0.005), (-0.0303, 0.0, 0.025), (-0.0303, 0.0, 0.005), 0.019),
    Assembly(6, "rod_insert_4mm", "round_peg_4mm", "round_hole_4mm",
             (0.3473, -0.1918, -0.0001, 0.7071, -0.7071, 0.0, 0.0),
             (0.0, 0.0, 0.0), (0.0, 0.0, 0.009), (0.0, 0.0, 0.0), 0.019),
    Assembly(7, "rod_insert_8mm", "round_peg_8mm", "round_hole_8mm",
             (0.3473, -0.1164, -0.0001, 0.7071, -0.7071, 0.0, 0.0),
             (0.0, 0.0, 0.0), (0.0, 0.0, 0.009), (0.0, 0.0, 0.0), 0.019),
    Assembly(8, "rod_insert_12mm", "round_peg_12mm", "round_hole_12mm",
             (0.1226, -0.0422, -0.0001, 0.7071, -0.7071, 0.0, 0.0),
             (0.0, 0.0, 0.0), (0.0, 0.0, 0.009), (0.0, 0.0, 0.0), 0.019),
    Assembly(9, "rod_insert_16mm", "round_peg_16mm", "round_hole_16mm",
             (0.1221, -0.2665, -0.0001, 0.7071, -0.7071, 0.0, 0.0),
             (0.0, 0.0, 0.0), (0.0, 0.0, 0.009), (0.0, 0.0, 0.0), 0.019),
    Assembly(10, "peg_insert_4mm", "rectangular_peg_4mm", "rectangular_hole_4mm",
             (0.1971, -0.1915, -0.0003, 0.7071, -0.7071, 0.0, 0.0),
             (0.0, 0.0, 0.0), (0.0, 0.0, 0.009), (0.0, 0.0, 0.0), 0.019),
    Assembly(11, "peg_insert_8mm", "rectangular_peg_8mm", "rectangular_hole_8mm",
             (0.2717, -0.2659, -0.0003, 0.7071, -0.7071, 0.0, 0.0),
             (0.0, 0.0, 0.0), (0.0, 0.0, 0.009), (0.0, 0.0, 0.0), 0.019),
    Assembly(12, "peg_insert_12mm", "rectangular_peg_12mm", "rectangular_hole_12mm",
             (0.1971, -0.0413, -0.0003, 0.0, 1.0, 0.0, 0.0),
             (0.0, 0.0, 0.0), (0.0, 0.0, 0.009), (0.0, 0.0, 0.0), 0.019),
    Assembly(13, "peg_insert_16mm", "rectangular_peg_16mm", "rectangular_hole_16mm",
             (0.3472, -0.0413, -0.0003, 0.7071, 0.7071, 0.0, 0.0),
             (0.0, 0.0, 0.0), (0.0, 0.0, 0.009), (0.0, 0.0, 0.0), 0.019),
    Assembly(14, "usba", "usb_a_plug", "usb_a_socket",
             (0.2721, -0.0415, -0.0001, 0.7071, 0.7071, 0.0, 0.0),
             (0.0, 0.0, 0.0335), (0.0, 0.0, 0.0416), (0.0, 0.0, 0.0335), 0.05, 0.012),
    Assembly(15, "waterproof", "waterproof_plug", "waterproof_socket",
             (0.1981, -0.1166, -0.0002, 0.0, 1.0, 0.0, 0.0),
             (0.0, 0.0, 0.021), (0.0, 0.0, 0.034), (0.0, 0.0, 0.021), 0.05),
    Assembly(16, "bnc", "bnc_plug", "bnc_socket",
             (0.2797, -0.1915, 0.0, 0.7071, 0.7071, 0.0, 0.0),
             (0.0, 0.0, 0.0107), (0.0, 0.0, 0.0107), (0.0, 0.0, 0.0107), 0.05),
    Assembly(17, "dsub", "dsub_plug", "dsub_socket",
             (0.2129, -0.2659, -0.019, 0.7071, -0.7071, 0.0, 0.0),
             (0.0, 0.0, 0.0), (0.0, 0.0, 0.0061), (0.0, 0.0, 0.0), 0.005),
    Assembly(18, "rj45", "rj45_plug", "rj45_socket",
             (0.3473, -0.3415, 0.0, 0.0, 1.0, 0.0, 0.0),
             (0.0, 0.0, 0.015), (0.0, 0.0, 0.028), (0.0, 0.0, 0.015), 0.05),
)
# fmt: on
ASSET_DIR = f"{ISAACLAB_NUCLEUS_DIR}/Factory/NIST"
GRASP = (0.0, 0.0, 0.107, 0.0, -1.0, 0.0, 0.0)
WORKSPACE_LOWER = (0.0, -0.675, -0.05)
WORKSPACE_UPPER = (1.0, 0.675, 1.0)
BOARD_CENTER_LOCAL = (0.19232847446867662, -0.191768517928605, 0.0)
# Simple spatial ranges; board orientation and asset identity do not change their bounds.
BOARD_CENTER_RANGE = ((0.40, 0.50), (-0.03, 0.03), (0.0206, 0.0206))
DROP_POSITION_RANGE = ((0.10, 0.50), (0.30, 0.50), (0.12, 0.14))  # Mirror Y for the other strip.
CONTACT_MATERIAL = [
    UsdPhysicsRigidBodyMaterialCfg(static_friction=0.75, dynamic_friction=0.75),
    NewtonMaterialCfg(contact_stiffness=1.6e5, contact_damping=800.0),
]
SOCKET_COLLISION = sim_utils.NewtonSDFCollisionPropertiesCfg(
    rest_offset=0.0,
    contact_gap=0.005,
    sdf_max_resolution=256,
    sdf_narrow_band_inner=-0.005,
    sdf_narrow_band_outer=0.005,
)


def grasp_pose(env: ManagerBasedEnv, robot_cfg: SceneEntityCfg) -> torch.Tensor:
    """Grasp-point position [m] and xyzw orientation in the robot root frame."""
    robot = env.scene[robot_cfg.name]
    hand = robot.data.body_link_pose_w.torch[:, robot_cfg.body_ids[0]]
    offset = hand.new_tensor(GRASP).expand_as(hand)
    position, rotation = math_utils.combine_frame_transforms(hand[:, :3], hand[:, 3:], offset[:, :3], offset[:, 3:])
    root = robot.data.root_link_pose_w.torch
    return torch.cat(math_utils.subtract_frame_transforms(root[:, :3], root[:, 3:], position, rotation), dim=-1)


def grasp_velocity(env: ManagerBasedEnv, robot_cfg: SceneEntityCfg) -> torch.Tensor:
    """Grasp-point linear [m/s] and angular [rad/s] velocity in the robot root frame."""
    robot = env.scene[robot_cfg.name]
    hand_id = robot_cfg.body_ids[0]
    hand_rotation = robot.data.body_link_quat_w.torch[:, hand_id]
    velocity = robot.data.body_link_vel_w.torch[:, hand_id]
    lever = math_utils.quat_apply(hand_rotation, velocity.new_tensor(GRASP[:3]).expand(env.num_envs, -1))
    linear = velocity[:, :3] + torch.cross(velocity[:, 3:], lever, dim=-1)
    root_rotation = robot.data.root_link_quat_w.torch
    return torch.cat(
        (
            math_utils.quat_apply_inverse(root_rotation, linear),
            math_utils.quat_apply_inverse(root_rotation, velocity[:, 3:]),
        ),
        dim=-1,
    )


def reset_parts(env: ManagerBasedEnv, env_ids: torch.Tensor | slice, assemblies: tuple[Assembly, ...]) -> None:
    """Sample the board centre and two loose-part strips from explicit position ranges [m]."""
    if isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=env.device)[env_ids]
    count = len(env_ids)
    if count == 0:
        return
    robot = env.scene["robot"]
    joints = robot.data.default_joint_pos.torch[env_ids]
    robot.write_joint_state_to_sim_index(position=joints, velocity=torch.zeros_like(joints), env_ids=env_ids)

    board = env.scene["nistboard"]
    board_pose = board.data.default_root_pose.torch[env_ids].clone()
    yaw = torch.rand(count, device=env.device) * 6.28 - 3.14
    board_pose[:, 3:] = math_utils.quat_mul(
        board_pose[:, 3:], math_utils.quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
    )
    bounds = board_pose.new_tensor(BOARD_CENTER_RANGE)
    center = bounds[:, 0] + torch.rand((count, 3), device=env.device) * (bounds[:, 1] - bounds[:, 0])
    board_pose[:, :3] = center - math_utils.quat_apply(
        board_pose[:, 3:], board_pose.new_tensor(BOARD_CENTER_LOCAL).expand(count, -1)
    )
    board_pose[:, :3] += env.scene.env_origins[env_ids]
    board.write_root_link_pose_to_sim_index(root_pose=board_pose, env_ids=env_ids)
    # Compose from raw xyzw poses, preserving the quaternion convention seen during training.
    for socket, part in {part.socket: part for part in assemblies}.items():
        offset = board_pose.new_tensor(part.board_pose)
        position, rotation = math_utils.combine_frame_transforms(
            board_pose[:, :3], board_pose[:, 3:], offset[:3].expand(count, -1), offset[3:].expand(count, -1)
        )
        env.scene[f"fixed_{socket}"].write_root_link_pose_to_sim_index(
            root_pose=torch.cat((position, rotation), dim=1), env_ids=env_ids
        )

    positions = board_pose.new_empty((count, 10, 3))
    bounds = board_pose.new_tensor(DROP_POSITION_RANGE)
    sides = torch.rand((count, 10), device=env.device).argsort(dim=1).remainder(2) * 2 - 1
    for slot in range(10):
        pending = torch.arange(count, device=env.device)
        for _ in range(512):
            if len(pending) == 0:
                break
            candidate = bounds[:, 0] + torch.rand((len(pending), 3), device=env.device) * (bounds[:, 1] - bounds[:, 0])
            candidate[:, 1] *= sides[pending, slot]
            clear = ((candidate[:, None, :2] - positions[pending, :slot, :2]).square().sum(dim=-1) >= 0.1**2).all(dim=1)
            positions[pending[clear], slot] = candidate[clear]
            pending = pending[~clear]
        if len(pending):
            raise RuntimeError("Could not sample separated loose parts inside the drop region.")
        asset = env.scene[f"held_{slot:02d}"]
        pose = asset.data.default_root_pose.torch[env_ids].clone()
        pose[:, :3] = positions[:, slot] + env.scene.env_origins[env_ids]
        angles = (torch.rand((count, 3), device=env.device) * 2.0 - 1.0) * pose.new_tensor((1.57, 1.57, 3.14))
        pose[:, 3:7] = math_utils.quat_from_euler_xyz(*angles.unbind(-1))
        asset.write_root_link_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
        asset.write_root_com_velocity_to_sim_index(root_velocity=pose.new_zeros((len(env_ids), 6)), env_ids=env_ids)


class SlotGeometry(ManagerTermBase):
    """The exported actor's three relative poses and full-catalog one-hot per object."""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        assemblies = cfg.params["assemblies"]
        self.held = tuple(env.scene[f"held_{slot:02d}"] for slot in range(10))
        self.fixed = tuple(env.scene[f"fixed_{part.socket}"] for part in assemblies)
        self.robot = env.scene["robot"]
        self.hand_id = self.robot.find_bodies("panda_hand")[0][0]
        self.held_align = torch.tensor([(*part.held_align, 0, 0, 0, 1) for part in assemblies], device=self.device)
        self.fixed_tip = torch.tensor([(*part.fixed_tip, 0, 0, 0, 1) for part in assemblies], device=self.device)
        self.seated = torch.tensor([(*part.seated, 0, 0, 0, 1) for part in assemblies], device=self.device)
        self.identity = torch.eye(len(ASSEMBLIES), device=self.device)[[part.identity for part in assemblies]]

    def __call__(self, env: ManagerBasedEnv, assemblies: tuple[Assembly, ...]) -> torch.Tensor:
        held = torch.stack([asset.data.root_link_pose_w.torch for asset in self.held], dim=1)
        fixed = torch.stack([asset.data.root_link_pose_w.torch for asset in self.fixed], dim=1)
        hand = self.robot.data.body_link_pose_w.torch[:, self.hand_id]
        grasp = hand.new_tensor(GRASP).expand_as(hand)
        ee_pos, ee_quat = math_utils.combine_frame_transforms(hand[:, :3], hand[:, 3:], grasp[:, :3], grasp[:, 3:])
        ee_pos, ee_quat = ee_pos[:, None].expand(-1, 10, -1), ee_quat[:, None].expand(-1, 10, -1)
        align, tip = self.held_align.expand_as(held), self.fixed_tip.expand_as(fixed)
        held_pos, held_quat = math_utils.combine_frame_transforms(
            held[..., :3], held[..., 3:], align[..., :3], align[..., 3:]
        )
        fixed_pos, fixed_quat = math_utils.combine_frame_transforms(
            fixed[..., :3], fixed[..., 3:], tip[..., :3], tip[..., 3:]
        )
        poses = (
            *math_utils.subtract_frame_transforms(fixed_pos, fixed_quat, held_pos, held_quat),
            *math_utils.subtract_frame_transforms(ee_pos, ee_quat, held_pos, held_quat),
            *math_utils.subtract_frame_transforms(ee_pos, ee_quat, fixed_pos, fixed_quat),
        )
        seated = self.seated.expand_as(fixed)
        goal_pos, goal_quat = math_utils.combine_frame_transforms(
            fixed[..., :3], fixed[..., 3:], seated[..., :3], seated[..., 3:]
        )
        error_pos, error_quat = math_utils.subtract_frame_transforms(goal_pos, goal_quat, held_pos, held_quat)
        x, y, z, w = error_quat.unbind(-1)
        roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x.square() + y.square()))
        pitch = torch.asin((2.0 * (w * y - z * x)).clamp(-1.0, 1.0))
        aligned = roll.abs() + pitch.abs() < 0.025
        self.assembled = aligned & (error_pos[..., :2].norm(dim=-1) < 0.0025) & (error_pos[..., 2] < 0.001)
        position = held[..., :3] - env.scene.env_origins[:, None]
        lower, upper = position.new_tensor(WORKSPACE_LOWER), position.new_tensor(WORKSPACE_UPPER)
        self.invalid = (
            (~torch.isfinite(held)).any(dim=(1, 2))
            | ~torch.isfinite(hand).all(dim=1)
            | ((position < lower) | (position > upper)).any(dim=(1, 2))
        )
        return torch.cat((*poses, self.identity.expand(env.num_envs, -1, -1)), dim=-1)


class EpisodeReset(ManagerTermBase):
    """Reset finished boards after physics, before the single observation-history update."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.age = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.completed = self.successes = 0

    def reset(self, env_ids: torch.Tensor | slice | None = None):
        self.age[slice(None) if env_ids is None else env_ids] = 0

    def __call__(self, env: ManagerBasedEnv, env_ids: torch.Tensor | None, horizon: float):
        self.age += 1
        term = env.observation_manager.cfg["slots"].geometry
        geometry = term.func
        geometry(env, **term.params)
        assembled = geometry.assembled.sum(dim=1)
        robot = env.scene["robot"]
        failed = (
            geometry.invalid
            | ~torch.isfinite(robot.data.joint_pos.torch).all(dim=1)
            | ~torch.isfinite(robot.data.joint_vel.torch).all(dim=1)
        )
        success = (assembled == 10) & ~failed
        finished = torch.nonzero(success | failed | (self.age * env.step_dt >= horizon)).flatten()
        if len(finished) == 0:
            return
        results = torch.stack((finished, assembled[finished], self.age[finished], success[finished]), dim=1).tolist()
        for env_id, count, age, succeeded in results:
            self.completed += 1
            self.successes += succeeded
            print(
                f"Episode {self.completed} (env {env_id}): {count}/10 assembled, {age * env.step_dt:.1f}s; "
                f"complete-board successes {self.successes}/{self.completed}",
                flush=True,
            )
        # Public reset() also appends history globally, duplicating samples in continuing boards.
        env._reset_idx(finished)
        env.scene.write_data_to_sim()
        env.sim.forward()


@configclass
class RobotObservations(ObservationGroupCfg):
    """Term-major robot history expected by the exported actor."""

    enable_corruption = False
    history_length = 5
    end_effector_pose_b = ObservationTermCfg(
        func=grasp_pose, params={"robot_cfg": SceneEntityCfg("robot", body_names=["panda_hand"])}
    )
    end_effector_vel_lin_ang_b = ObservationTermCfg(
        func=grasp_velocity, params={"robot_cfg": SceneEntityCfg("robot", body_names=["panda_hand"])}
    )
    joint_pos = ObservationTermCfg(func=mdp.joint_pos)
    joint_vel = ObservationTermCfg(func=mdp.joint_vel)
    prev_action = ObservationTermCfg(func=mdp.last_action)


@configclass
class SlotObservations(ObservationGroupCfg):
    """Unflattened [history, object, feature] input to the slot encoder."""

    enable_corruption = False
    history_length = 5
    flatten_history_dim = False
    geometry = ObservationTermCfg(func=SlotGeometry)


@configclass
class Actions:
    """Seven relative arm targets and one binary gripper command."""

    arm_action = mdp.RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        scale={"(?!panda_joint7).*": 0.02, "panda_joint7": 0.2},
        use_zero_offset=True,
    )
    gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_finger.*"],
        open_command_expr={"panda_finger_.*": 0.04},
        close_command_expr={"panda_finger_.*": 0.0},
    )


def main() -> None:
    """Load a local TorchScript actor and run complete ten-part assembly episodes."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--policy", type=Path, required=True, help="Exported K10 TorchScript .pt, not a training checkpoint."
    )
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--num_envs", type=int, default=1, help="Independent boards simulated in parallel.")
    parser.add_argument(
        "--episodes",
        type=int,
        default=0,
        help="Total completed episodes across boards before exit; zero runs until closed.",
    )
    parser.add_argument("--horizon", type=float, default=140.0, help="Episode limit [s].")
    parser.add_argument("--real_time", action="store_true", help="Pace the visualizer at real time.")
    parser.add_argument("--max_steps", type=int, default=-1, help="Stop after this many steps; negative runs forever.")
    add_launcher_args(parser)
    parser.set_defaults(visualizer=["newton_rtx"])
    args = parser.parse_args()
    if args.num_envs < 1 or args.episodes < 0 or args.horizon <= 0:
        parser.error("num_envs and horizon must be positive; episodes must be nonnegative.")
    if args.max_steps == 0 or args.max_steps < -1:
        parser.error("max_steps must be positive or -1.")
    policy = torch.jit.load(str(args.policy), map_location=args.device).eval()
    assemblies = tuple(random.Random(args.seed).sample(ASSEMBLIES, 10))
    slot_observations = SlotObservations()
    slot_observations.geometry.params = {"assemblies": assemblies}
    scene = InteractiveSceneCfg(num_envs=args.num_envs, env_spacing=2.0)
    scene.ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(
            color=(0.65, 0.65, 0.65),
            physics_material=[
                UsdPhysicsRigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
                NewtonMaterialCfg(contact_stiffness=2500.0, contact_damping=100.0),
            ],
        ),
    )
    scene.nistboard = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/NistBoard",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ASSET_DIR}/Taskboard/nistboard.usd",
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True),
            physics_material=CONTACT_MATERIAL,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.452824, 0.19145, 0.0206), rot=(0.0, 1.0, 0.0, 0.0)),
    )
    scene.robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        joint_ordering=[f"panda_joint{i}" for i in range(1, 8)] + ["panda_finger_joint1", "panda_finger_joint2"],
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/franka_panda.usda",
            activate_contact_sensors=True,
            rigid_props=MujocoRigidBodyCfg(gravcomp=1.0),
            articulation_props=NewtonArticulationCfg(self_collision_enabled=False),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.005,
                rest_offset=0.0,
                mesh_collision_property=sim_utils.NewtonMeshCollisionPropertiesCfg(
                    mesh_approximation_name="convexHull"
                ),
            ),
            physics_material=[
                UsdPhysicsRigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
                NewtonMaterialCfg(contact_stiffness=1.0e6, contact_damping=2000.0),
            ],
            joint_drive_props=sim_utils.UsdPhysicsDriveCfg(),
            ensure_drives_exist=True,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 0.00871,
                "panda_joint2": -0.10368,
                "panda_joint3": -0.00794,
                "panda_joint4": -1.49139,
                "panda_joint5": -0.00083,
                "panda_joint6": 1.38774,
                "panda_joint7": 0.0,
                "panda_finger_joint2": 0.04,
            }
        ),
        actuators={
            "panda_arm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-7]"],
                joint_effort_limit={"panda_joint[1-4]": 87.0, "panda_joint[5-7]": 12.0},
                actuator_velocity_limit={"panda_joint[1-4]": 2.175, "panda_joint[5-7]": 2.61},
                joint_velocity_limit={"panda_joint[1-4]": 20.0, "panda_joint[5-7]": 25.0},
                stiffness={
                    "panda_joint[1-4]": 600.0,
                    "panda_joint5": 250.0,
                    "panda_joint6": 150.0,
                    "panda_joint7": 50.0,
                },
                damping={"panda_joint[1-4]": 50.0, "panda_joint5": 30.0, "panda_joint6": 25.0, "panda_joint7": 15.0},
                armature={"panda_joint[1-2]": 0.6057, "panda_joint[3-4]": 0.4625, "panda_joint[5-7]": 0.2055},
            ),
            "panda_hand": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint1"],
                joint_effort_limit=70.0,
                actuator_velocity_limit=0.2,
                joint_velocity_limit=2.0,
                stiffness=350.0,
                damping=175.0,
                armature=0.1,
            ),
            "panda_finger2_passive": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint2"],
                joint_effort_limit=1.0,
                actuator_velocity_limit=0.2,
                joint_velocity_limit=2.0,
                stiffness=0.0,
                damping=0.0,
                armature=0.1,
            ),
        },
    )
    board = scene.nistboard.init_state
    board_pos, board_quat = torch.tensor(board.pos), torch.tensor(board.rot)
    for socket, part in {part.socket: part for part in assemblies}.items():
        offset = torch.tensor(part.board_pose)
        pos, quat = math_utils.combine_frame_transforms(board_pos, board_quat, offset[:3], offset[3:])
        setattr(
            scene,
            f"fixed_{socket}",
            RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/fixed_{socket}",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=f"{ASSET_DIR}/{socket}.usd",
                    activate_contact_sensors=True,
                    collision_props=SOCKET_COLLISION,
                    physics_material=CONTACT_MATERIAL,
                    rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True),
                    mass_props=sim_utils.MassCfg(mass=part.socket_mass),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(pos.tolist()), rot=tuple(quat.tolist())),
            ),
        )
    events = {
        "reset_parts": EventTermCfg(func=reset_parts, mode="reset", params={"assemblies": assemblies}),
        "reset_finished": EventTermCfg(
            func=EpisodeReset,
            mode="interval",
            interval_range_s=(0.0, 0.0),
            is_global_time=True,
            params={"horizon": args.horizon},
        ),
        "robot_material": EventTermCfg(
            func=mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "static_friction_range": (0.75, 0.75),
                "dynamic_friction_range": (0.75, 0.75),
                "restitution_range": (0.0, 0.0),
                "num_buckets": 64,
            },
        ),
    }
    for slot, part in enumerate(assemblies):
        offset = torch.tensor(part.board_pose)
        seated = torch.tensor((*part.seated, 0, 0, 0, 1))
        align = torch.tensor((*part.held_align, 0, 0, 0, 1))
        fixed_pos, fixed_quat = math_utils.combine_frame_transforms(board_pos, board_quat, offset[:3], offset[3:])
        goal_pos, goal_quat = math_utils.combine_frame_transforms(fixed_pos, fixed_quat, seated[:3], seated[3:])
        goal_root_quat = math_utils.quat_mul(goal_quat, math_utils.quat_inv(align[3:]))
        goal_root_pos = goal_pos - math_utils.quat_apply(goal_root_quat, align[:3])
        if not (
            (goal_root_pos >= torch.tensor(WORKSPACE_LOWER)) & (goal_root_pos <= torch.tensor(WORKSPACE_UPPER))
        ).all():
            raise ValueError(f"Assembly destination outside workspace: {part.name} at {goal_root_pos.tolist()}")
        name = f"held_{slot:02d}"
        setattr(
            scene,
            name,
            RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/{name}",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=f"{ASSET_DIR}/{part.plug}.usd",
                    activate_contact_sensors=True,
                    collision_props=SOCKET_COLLISION.replace(contact_offset=0.0025),
                    physics_material=CONTACT_MATERIAL,
                    mass_props=sim_utils.MassCfg(mass=part.plug_mass),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 2.0 + slot * 0.1)),
            ),
        )
        events[f"inertia_{slot}"] = EventTermCfg(
            func=mdp.randomize_rigid_body_inertia,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg(name),
                "inertia_distribution_params": (1.0e-5, 1.0e-5),
                "operation": "add",
                "diagonal_only": True,
            },
        )
    cfg = ManagerBasedEnvCfg(
        sim=sim_utils.SimulationCfg(
            dt=0.01,
            device=args.device,
            render_interval=2,
            physics=NewtonCfg(
                num_substeps=8,
                use_cuda_graph=True,
                solver_cfg=MJWarpSolverCfg(
                    solver="newton",
                    integrator="implicitfast",
                    njmax=3200,
                    nconmax=2400,
                    impratio=1.0,
                    cone="pyramidal",
                    update_data_interval=2,
                    ls_parallel=False,
                    use_mujoco_contacts=False,
                ),
                collision_cfg=NewtonCollisionPipelineCfg(
                    broad_phase="sap",
                    max_triangle_pairs=max(1_000_000, args.num_envs * 50_000),
                    rigid_contact_max=args.num_envs * 2400,
                ),
            ),
        ),
        scene=scene,
        actions=Actions(),
        observations={"policy": RobotObservations(), "slots": slot_observations},
        events=events,
        decimation=4,
        seed=args.seed,
    )
    print("Assets:", ", ".join(part.name for part in assemblies), flush=True)
    with launch_simulation(cfg=cfg, launcher_args=args) as physics:
        cfg.sim.physics = physics
        env = ManagerBasedEnv(cfg)
        try:
            center = env.scene.env_origins.mean(dim=0).cpu()
            target = center + torch.tensor((0.35, 0.0, 0.4))
            span = env.scene.env_origins.amax(dim=0) - env.scene.env_origins.amin(dim=0)
            camera_scale = 1.0 + 1.25 * float(span[:2].max()) / scene.env_spacing
            eye = target + camera_scale * torch.tensor((1.2, -0.65, 0.8))
            env.sim.set_camera_view(eye=eye.tolist(), target=target.tolist())
            obs, _ = env.reset()
            episodes = env.event_manager.get_term_cfg("reset_finished").func
            if obs["policy"].shape != (args.num_envs, 195) or obs["slots"].shape != (args.num_envs, 5, 10, 40):
                raise ValueError(f"Policy input mismatch: {obs['policy'].shape}, {obs['slots'].shape}")
            with torch.inference_mode():
                step = 0
                while env.sim.is_running() and (args.max_steps < 0 or step < args.max_steps):
                    start = time.monotonic()
                    actions = policy(obs["policy"], obs["slots"])
                    if not torch.isfinite(actions).all():
                        raise RuntimeError("Exported actor returned nonfinite actions.")
                    obs, _ = env.step(actions)
                    step += 1
                    if args.episodes and episodes.completed >= args.episodes:
                        break
                    if args.real_time:
                        time.sleep(max(0.0, env.step_dt - (time.monotonic() - start)))
        finally:
            env.close()


if __name__ == "__main__":
    main()
