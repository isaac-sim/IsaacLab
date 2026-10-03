# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rigid-body fruit, tap, threaded-lid and blender task with visual scalar filling.

The legacy scene supplies the same authored rigid assets and robot controls.
Liquid objects, particle proxies and the MPM entry are removed before constructing
any simulator. Runtime actions only drive robot joints; filling changes a scalar.
"""

from __future__ import annotations

import newton
import torch
from isaaclab_newton.cloner import newton_builder_world_hook
from isaaclab_newton.physics import NewtonManager

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs import mdp as base_mdp
from isaaclab.managers import EventTermCfg, ObservationGroupCfg, ObservationTermCfg, RewardTermCfg, TerminationTermCfg
from isaaclab.utils import math as math_utils
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.franka_pour.pour_env import _set_mjwarp_force_space_solref_mode

from . import basket_stage
from .basket_grasp_continuation import BasketGraspContinuation
from .cup_contact_spacing import apply_cup_contact_spacing, validate_cup_contact_runtime
from .fruit_receipt_geometry import WholeHullReceipt
from .pad_torsional_contact import apply_pad_torsional_contact, verify_pad_torsional_contact
from .scene_cfg import CUP_POSITION, FRUIT_GROUPS, FRUIT_LAYOUT
from .smoothie_assembly_geometry import AssemblyGeometryState, assembly_geometry
from .smoothie_asset import CUP_STATION_POSITION_M, NOZZLE_POSITION_M
from .smoothie_task import FILL_DURATION_S, SmoothiePhase, SmoothieTaskState, all_fruit_delivered

OBJECTS = ("cup", "blade_cap", "basket", *FRUIT_LAYOUT)


def failure(env: SmoothieBlenderEnv) -> torch.Tensor:
    """Reject nonfinite, dropped or escaped rigid objects and lost fastened lids."""
    invalid = ~torch.isfinite(env.robot.data.joint_pos.torch).all(-1)
    invalid |= ~torch.isfinite(env.robot.data.joint_vel.torch).all(-1)
    for name in (*OBJECTS, "tap", "motor"):
        pose = env.pose(name)
        invalid |= ~torch.isfinite(pose).all(-1)
        invalid |= ~torch.isfinite(env.scene[name].data.root_link_vel_w.torch).all(-1)
        invalid |= (pose[:, 2] - env.scene.env_origins[:, 2]) < -0.12
        invalid |= (pose[:, :3] - env.scene.env_origins).norm(dim=-1) > 1.5
    return invalid | env.task.failed | env.assembly.state["assembly_failed"]


def success(env: SmoothieBlenderEnv) -> torch.Tensor:
    """Advance measured milestones once and return ordered completion."""
    env.update_task()
    return env.task.completed & ~failure(env)


def progress(env: SmoothieBlenderEnv) -> torch.Tensor:
    """Reward each newly measured milestone once."""
    env.update_task()
    return env.progress_reward


def state(env: SmoothieBlenderEnv) -> torch.Tensor:
    """Observe rigid state, controls, button travel [m] and scalar fill fraction."""
    poses = []
    for name in (*OBJECTS, "tap", "motor"):
        pose = env.pose(name).clone()
        pose[:, :3] -= env.scene.env_origins
        poses.extend((pose, env.scene[name].data.root_link_vel_w.torch))
    return torch.cat(
        [
            env.robot.data.joint_pos.torch,
            env.robot.data.joint_vel.torch * 0.1,
            *poses,
            torch.nn.functional.one_hot(env.phase, 6).float(),
            env.task.fill_fraction[:, None],
            env.task.tap_on[:, None].float(),
            env.task.milestones.float(),
            env.scene["tap"].data.joint_pos.torch * 250,
            env.scene["motor"].data.joint_pos.torch * 250,
            env.assembly.observations(),
        ],
        dim=-1,
    )


def reset(env: SmoothieBlenderEnv, env_ids: torch.Tensor) -> None:
    """Reset physical rigid objects and task history at episode boundaries only."""
    env.reset_task(env_ids)


@configclass
class ObservationsCfg:
    """Task observations independent of the removed liquid solver."""

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        state = ObservationTermCfg(func=state)
        previous_action = ObservationTermCfg(func=base_mdp.last_action)
        concatenate_terms = True
        enable_corruption = False

    policy = PolicyCfg()


@configclass
class RewardsCfg:
    """Measured milestone reward and the existing action-rate regularizer."""

    progress = RewardTermCfg(func=progress, weight=1.0)
    action_rate = RewardTermCfg(func=base_mdp.action_rate_l2, weight=-0.002)


@configclass
class TerminationsCfg:
    """Ordered completion, physical failure and episode timeout."""

    success = TerminationTermCfg(func=success)
    failure = TerminationTermCfg(func=failure)
    time_out = TerminationTermCfg(func=base_mdp.time_out, time_out=True)


@configclass
class EventsCfg:
    """Reset only at episode boundaries."""

    reset = EventTermCfg(func=reset, mode="reset")


class SmoothieBlenderEnv(ManagerBasedRLEnv):
    """Physical manipulation with no liquid solver or runtime object attachment."""

    def pose(self, name: str) -> torch.Tensor:
        """Return object root pose in world coordinates [m, XYZW]."""
        return self.scene[name].data.root_link_pose_w.torch

    def local(self, name: str, points: torch.Tensor) -> torch.Tensor:
        """Transform world points [m], shape (N, P, 3), into an object frame."""
        pose = self.pose(name)
        return math_utils.quat_apply_inverse(
            pose[:, None, 3:].expand(-1, points.shape[1], -1), points - pose[:, None, :3]
        )

    def offset(self, name: str, xyz: tuple[float, float, float]) -> torch.Tensor:
        """Return a local grasp or target point [m] in world coordinates."""
        pose = self.pose(name)
        return pose[:, :3] + math_utils.quat_apply(
            pose[:, 3:], torch.tensor(xyz, device=self.device).expand(self.num_envs, -1)
        )

    def tcp(self) -> torch.Tensor:
        """Return the Franka finger midpoint [m] in world coordinates."""
        pose = self.robot.data.body_link_pose_w.torch[:, self.hand_id]
        offset = torch.tensor((0.0, 0.0, 0.107), device=self.device).expand(self.num_envs, -1)
        return pose[:, :3] + math_utils.quat_apply(pose[:, 3:], offset)

    def fruit_inside(self) -> torch.Tensor:
        """Report whether the cup contains at least one fruit of each type."""
        occupied = []
        for names in FRUIT_GROUPS.values():
            centers = torch.stack([self.pose(name)[:, :3] for name in names], 1)
            local = self.local("cup", centers)
            # Margins require the fruit center to be fully below the lip.
            inside = (local[:, :, :2].norm(dim=-1) < 0.035) & (local[:, :, 2] > 0.010) & (local[:, :, 2] < 0.185)
            occupied.append(inside.any(-1))
        return torch.stack(occupied, -1)

    def fruit_fraction(self) -> torch.Tensor:
        """Fraction of the fruit centers safely below the receiving cup's lip."""
        centers = torch.stack([self.pose(name)[:, :3] for name in FRUIT_LAYOUT], 1)
        local = self.local("cup", centers)
        inside = (local[:, :, :2].norm(dim=-1) < 0.035) & (local[:, :, 2] > 0.01) & (local[:, :, 2] < 0.185)
        return inside.float().mean(-1)

    def __init__(self, cfg, render_mode=None, **kwargs):
        if cfg.scene.num_envs != 1:
            raise ValueError("The measured basket controller currently supports one environment.")
        if cfg.sim.physics.solver_cfg.solver_type != "mujoco_warp":
            raise ValueError("Smoothie task requires the direct rigid solver.")
        self._pad_identity = None
        with newton_builder_world_hook(self._configure_world):
            super().__init__(cfg, render_mode, **kwargs)
        model = NewtonManager.get_model()
        if model.particle_count != 0 or any("_MPM" in label for label in model.shape_label):
            raise RuntimeError("Unexpected liquid particles or MPM collision proxies.")
        validate_cup_contact_runtime(NewtonManager._solver)
        verify_pad_torsional_contact(NewtonManager._solver, self._pad_identity)

    def _configure_world(self, builder, env_id, position, quaternion):
        for index, world in enumerate(builder.shape_world):
            if world != env_id:
                continue
            label = builder.shape_label[index]
            builder.shape_flags[index] = int(builder.shape_flags[index]) & ~int(newton.ShapeFlags.COLLIDE_PARTICLES)
            if "/Robot/" not in label:
                builder.shape_margin[index] = 0.00015
                builder.shape_material_mu[index] = 0.25 if "/Thread/" in label else 0.8
                builder.shape_material_ke[index] = 1e5
                builder.shape_material_kd[index] = 500.0
                builder.shape_material_kf[index] = 1000.0
                _set_mjwarp_force_space_solref_mode(builder, index)
        apply_cup_contact_spacing(builder, env_id)
        self._pad_identity = apply_pad_torsional_contact(builder, env_id)

    def load_managers(self) -> None:
        self.robot = self.scene["robot"]
        self.hand_id = self.robot.find_bodies("panda_hand")[0][0]
        self.finger_ids = self.robot.find_joints("panda_finger_joint.*")[0]
        self.task = SmoothieTaskState(self.num_envs, self.device, self.step_dt)
        self.phase = self.task.phase
        self.assembly = AssemblyGeometryState(self.num_envs, self.device, self.step_dt)
        self.basket_state = basket_stage.new_milestones(self.num_envs, self.device)
        self.basket_grasp = BasketGraspContinuation(self.device, self.step_dt)
        self.basket_strict_lift_hold_s = torch.zeros(self.num_envs, device=self.device)
        self.basket_strict_lift_complete = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.hull_receipt = WholeHullReceipt(tuple(FRUIT_LAYOUT), self.device)
        self.progress_reward = torch.zeros(self.num_envs, device=self.device)
        self.motor_on = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._last_update = -1
        super().load_managers()

    def reset_task(self, env_ids: torch.Tensor) -> None:
        base_mdp.reset_scene_to_default(self, env_ids, reset_joint_targets=True)
        self.task.reset(env_ids.to(dtype=torch.long))
        self.assembly.reset(env_ids)
        basket_stage.reset_milestones(self.basket_state, env_ids)
        self.basket_grasp.reset()
        self.basket_strict_lift_hold_s[env_ids] = 0
        self.basket_strict_lift_complete[env_ids] = False
        self.progress_reward[env_ids] = 0
        self.motor_on[env_ids] = False
        positions = self.robot.data.default_joint_pos.torch[env_ids].clone()
        positions[:, self.finger_ids] = 0.015
        self.robot.write_joint_state_to_sim_index(
            position=positions, velocity=torch.zeros_like(positions), env_ids=env_ids
        )
        self.robot.set_joint_position_target_index(target=positions, env_ids=env_ids)
        self.action_manager.get_term("gripper_action").set_reset_position(
            positions[:, self.finger_ids[:1]], env_ids=env_ids
        )
        self._last_update = self.common_step_counter

    def grasp_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the basket grasp point [m] and hand rotation [XYZW]."""
        pose = self.pose("basket")
        offset = pose.new_tensor(self.cfg.basket_grasp_offset).expand(self.num_envs, -1)
        rotation = pose.new_tensor(self.cfg.basket_grasp_rotation).expand(self.num_envs, -1)
        return pose[:, :3] + math_utils.quat_apply(pose[:, 3:], offset), math_utils.quat_mul(pose[:, 3:], rotation)

    def assembly_measurements(self, basket=None):
        """Measure rigid lid/cup geometry with a dimensionless virtual fill fraction."""
        if basket is None:
            basket = self.basket_measurements()
        gripper = self.action_manager.get_term("gripper_action")
        measured = assembly_geometry(
            cup_pose=self.pose("cup"),
            cap_pose=self.pose("blade_cap"),
            motor_pose=self.pose("motor"),
            hand_pose=self.robot.data.body_link_pose_w.torch[:, self.hand_id],
            cup_velocity=self.scene["cup"].data.root_link_vel_w.torch,
            cap_velocity=self.scene["blade_cap"].data.root_link_vel_w.torch,
            hand_velocity=self.robot.data.body_link_vel_w.torch[:, self.hand_id],
            tcp_position=self.tcp(),
            finger_positions=self.robot.data.joint_pos.torch[:, self.finger_ids],
            commanded_positions=gripper.processed_actions,
            contact_deflection=gripper.contact_deflection,
            bilateral_contact=gripper.bilateral_contact,
            origins=self.scene.env_origins,
            fruit_count=basket["fruit_count"],
            all_types=basket["all_types"],
            fill_fraction=self.task.fill_fraction,
            failed=failure(self),
        )
        return {**measured, **self.assembly.state}

    def tap_measurements(self, assembly=None):
        """Measure station support, physical buttons and scalar filling [SI units]."""
        if assembly is None:
            assembly = self.assembly_measurements()
        cup = self.pose("cup")
        local = cup[:, :3] - self.scene.env_origins
        up = math_utils.quat_apply(cup[:, 3:], cup.new_tensor((0.0, 0.0, 1.0)).expand(self.num_envs, -1))
        opening = local + 0.210 * up
        nozzle = cup.new_tensor(NOZZLE_POSITION_M)
        aligned = (opening[:, :2] - nozzle[:2]).norm(dim=-1) < 0.018
        aligned &= ((nozzle[2] - opening[:, 2]) > 0.020) & ((nozzle[2] - opening[:, 2]) < 0.10)
        upright = up[:, 2] > 0.985
        station = (local - cup.new_tensor(CUP_STATION_POSITION_M)).abs()
        station_support = (station[:, :2].norm(dim=-1) < 0.012) & (station[:, 2] < 0.003)
        home = (local - cup.new_tensor(CUP_POSITION)).norm(dim=-1) < 0.006
        cap_local = self.local("cup", self.pose("blade_cap")[:, None, :3])[:, 0]
        cup_open = (cap_local[:, :2].norm(dim=-1) > 0.070) | ((cap_local[:, 2] - 0.220).abs() > 0.055)
        valve = -self.scene["tap"].data.joint_pos.torch[:, 0]
        motor = self.scene["motor"].data.joint_pos.torch[:, 0] >= 0.002
        invalid = failure(self)
        return dict(
            finite=~invalid,
            failed=invalid,
            cup_held=assembly["cup_held"],
            cup_supported=(station_support | home) & upright & assembly["cup_stable"],
            cup_under_tap=aligned & upright,
            cup_upright=upright,
            cup_open=cup_open,
            cup_home=home & upright,
            fingers_open=assembly["fingers_open"],
            tap_on=self.task.tap_on,
            tap_fill_time_s=self.task.fill_fraction * FILL_DURATION_S,
            tap_button_pressed=valve >= 0.002,
            motor_button_pressed=motor,
            motor_on=self.motor_on,
            valve_depression_m=valve,
        )

    def update_task(self) -> None:
        """Advance once per measured policy boundary, preserving the physical scene."""
        if self._last_update == self.common_step_counter:
            return
        self._last_update = self.common_step_counter
        basket = self._update_basket_continuation(self.basket_measurements())
        active = self.phase == SmoothiePhase.FRUIT
        strict_lift = (
            active
            & basket["finite"]
            & ~basket["failed"]
            & basket["strict_held"]
            & (basket["clearance_m"] >= basket_stage.CRITERIA["held_upright_lift_m"])
            & (basket["basket_upright"] >= basket_stage.CRITERIA["held_lift_upright_cosine"])
        )
        self.basket_strict_lift_hold_s.copy_(
            torch.where(strict_lift, self.basket_strict_lift_hold_s + self.step_dt, 0.0)
        )
        self.basket_strict_lift_complete |= (
            self.basket_strict_lift_hold_s >= basket_stage.CRITERIA["held_lift_hold_s"] - 1e-6
        )
        basket_stage.advance_milestones(
            self.basket_state, {**basket, "failed": basket["failed"] | ~active}, self.step_dt
        )
        measured = self.assembly_measurements(basket)
        self.assembly.update(
            measured,
            (self.phase >= SmoothiePhase.LID) & (self.phase <= SmoothiePhase.BUTTON),
            self.task.milestones[:, 1],
            step=int(self.common_step_counter),
        )
        measured.update(self.assembly.state)
        tap = self.tap_measurements(measured)
        before = self.task.milestones.sum(-1)
        fruit = all_fruit_delivered(basket["fruit_hull_inside"]) & self.basket_state["finished"]
        self.task.update(
            fruit_delivered=fruit,
            cup_under_tap=tap["cup_under_tap"],
            cup_upright=tap["cup_upright"],
            cup_open=tap["cup_open"],
            lid_fastened=measured["lid_twist_complete"] & measured["lid_seated"] & measured["lid_release_complete"],
            cup_docked=measured["assembly_complete"] & measured["dock_support_ready"],
            blender_button_pressed=tap["motor_button_pressed"],
            valve_depression_m=tap["valve_depression_m"],
            failed=tap["failed"],
        )
        self.progress_reward.copy_((self.task.milestones.sum(-1) - before).float())
        self.motor_on.copy_(self.task.completed)
        self.extras.setdefault("log", {}).update(
            {"Task/phase": self.phase.float().mean(), "Task/fill_fraction": self.task.fill_fraction.mean()}
        )

    def basket_measurements(self) -> dict[str, torch.Tensor]:
        """Measure basket grasp/receipt/support [SI units] independently of active phase."""
        pose = self.pose("basket")
        hand = self.robot.data.body_link_pose_w.torch[:, self.hand_id]
        hv = self.robot.data.body_link_vel_w.torch[:, self.hand_id]
        velocity = self.scene["basket"].data.root_link_vel_w.torch
        grasp = pose[:, :3] + math_utils.quat_apply(
            pose[:, 3:], pose.new_tensor(self.cfg.basket_grasp_offset).expand(self.num_envs, -1)
        )
        rotation = math_utils.quat_mul(
            pose[:, 3:], pose.new_tensor(self.cfg.basket_grasp_rotation).expand(self.num_envs, -1)
        )
        fingers = self.robot.data.joint_pos.torch[:, self.finger_ids]
        gripper = self.action_manager.get_term("gripper_action")
        distance = (self.tcp() - grasp).norm(dim=-1)
        speed = (
            velocity[:, :3]
            + torch.linalg.cross(velocity[:, 3:], grasp - pose[:, :3])
            - hv[:, :3]
            - torch.linalg.cross(hv[:, 3:], grasp - hand[:, :3])
        ).norm(dim=-1)
        actual_grasp_speed = (
            velocity[:, :3]
            + torch.linalg.cross(velocity[:, 3:], self.tcp() - pose[:, :3])
            - hv[:, :3]
            - torch.linalg.cross(hv[:, 3:], self.tcp() - hand[:, :3])
        ).norm(dim=-1)
        corners = pose.new_tensor([[x, y, z] for x in (-0.0605, 0.0605) for y in (-0.0605, 0.0605) for z in (0.0, 0.1)])
        world = (
            math_utils.quat_apply(pose[:, None, 3:].expand(-1, 8, -1), corners.expand(self.num_envs, -1, -1))
            + pose[:, None, :3]
        )
        result = basket_stage.placement_geometry(
            pose, velocity, self.tcp(), fingers, gripper.commanded_position, self.scene.env_origins, 0.015, grasp
        )
        held = (
            gripper.bilateral_contact
            & (distance < self.cfg.basket_grasp_distance)
            & (fingers.sum(-1) > 0.001)
            & (fingers.sum(-1) < 0.065)
            & (speed < self.cfg.basket_relative_speed)
            & result["finite"]
        )
        inside, violations = self.hull_receipt.measure(
            self.pose("cup"), torch.stack([self.pose(name) for name in FRUIT_LAYOUT], dim=1)
        )
        continuation = self.basket_grasp.current & (self.basket_grasp.last_step == self.common_step_counter)
        result.update(
            strict_held=held,
            grasp_continuation=continuation,
            grasp_acquired=self.basket_grasp.acquired,
            grasp_acquisition_step=self.basket_grasp.acquisition_step,
            held=held | continuation,
            grip_position_b_m=math_utils.quat_apply_inverse(pose[:, 3:], self.tcp() - pose[:, :3]),
            hand_rotation_b_xyzw=math_utils.quat_mul(math_utils.quat_conjugate(pose[:, 3:]), hand[:, 3:]),
            commanded_position_m=gripper.commanded_position,
            contact_deflection_m=gripper.contact_deflection,
            tcp_position_w_m=self.tcp(),
            pose_w=pose,
            grasp_relative_speed_m_s=actual_grasp_speed,
            acquisition_hand_path_m=self.basket_grasp.hand_path_m,
            acquisition_basket_path_m=self.basket_grasp.basket_path_m,
            acquisition_hand_displacement_m=self.basket_grasp.hand_displacement_m,
            acquisition_basket_displacement_m=self.basket_grasp.basket_displacement_m,
            acquisition_motion_correlation=self.basket_grasp.motion_correlation,
            continuation_window_s=self.basket_grasp.window_s,
            continuation_translation_drift_m=self.basket_grasp.translation_drift_m,
            continuation_rotation_drift_rad=self.basket_grasp.rotation_drift_rad,
            reach_m=distance,
            alignment=(hand[:, 3:] * rotation).sum(-1).square().clamp(0, 1),
            relative_speed_m_s=speed,
            finger_width_m=fingers.sum(-1),
            clearance_m=world[:, :, 2].amin(-1) - self.scene.env_origins[:, 2],
            upright=result["basket_upright"],
            target_error_m=(pose[:, :3] - self.offset("cup", (0.0, 0.0, 0.30))).norm(dim=-1),
            fruit_count=inside.sum(-1),
            all_types=self.hull_receipt.all_types(inside),
            fruit_hull_inside=inside,
            fruit_hull_max_plane_violation_m=violations,
            legacy_center_fruit_count=torch.round(self.fruit_fraction() * len(FRUIT_LAYOUT)).long(),
            legacy_center_all_types=self.fruit_inside().all(-1),
            failed=failure(self),
        )
        return result

    def _update_basket_continuation(self, basket):
        """Update grasp evidence once per physical policy boundary, before task gates."""
        continuation = self.basket_grasp.update(
            int(self.common_step_counter),
            valid=basket["finite"] & ~basket["failed"] & (self.phase == 0),
            command_m=basket["commanded_position_m"],
            deflection_m=basket["contact_deflection_m"],
            finger_width_m=basket["finger_width_m"],
            relative_speed_m_s=basket["grasp_relative_speed_m_s"],
            grip_position_b_m=basket["grip_position_b_m"],
            hand_rotation_b_xyzw=basket["hand_rotation_b_xyzw"],
            tcp_position_w_m=basket["tcp_position_w_m"],
            basket_pose_w=basket["pose_w"],
            clearance_m=basket["clearance_m"],
            upright=basket["upright"],
        )
        basket["grasp_continuation"] = continuation
        basket["held"] = basket["strict_held"] | continuation
        return basket
