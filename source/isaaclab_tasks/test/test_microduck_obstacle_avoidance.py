# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sensor ranges/cadence and frozen-controller training contracts for MicroDuck."""

import json
import math
from types import SimpleNamespace

import gymnasium as gym
import pytest
import torch

import isaaclab.sim as sim_utils
from isaaclab.sensors.ray_caster.base_multi_mesh_ray_caster import BaseMultiMeshRayCaster
from isaaclab.sim import SimulationContext
from isaaclab.test.utils import DeviceScope, test_devices
from isaaclab.utils.math import quat_apply, quat_apply_inverse, yaw_quat

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.contrib.microduck.mdp.commands import MicroDuckVelocityCommand
from isaaclab_tasks.contrib.microduck_obstacle.mdp.actions import MicroDuckWalkingAction, MicroDuckWalkingActionCfg
from isaaclab_tasks.contrib.microduck_obstacle.mdp.commands import MicroDuckNavigationCommand
from isaaclab_tasks.contrib.microduck_obstacle.mdp.events import (
    OBSTACLE_CELLS,
    OBSTACLE_LAYOUT_NAMES,
    reset_obstacle_field,
)
from isaaclab_tasks.contrib.microduck_obstacle.mdp.geometry import obstacle_clearances
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg

TASK = "IsaacContrib-ObstacleAvoidance-Flat-MicroDuck"


def test_navigation_scene_combines_public_props_and_primitives():
    """Config composition needs no local manifest or prepared asset bundle."""
    cfg = parse_env_cfg(TASK, device="cpu", overrides=("presets=backlash",))
    for name, filename in (
        ("crate", "sm_box_cardboard_a01_01.usd"),
        ("low_block", "sm_trashCan_office_black_a03_01.usd"),
        ("cylinder", "sm_table_bedside_b01_01.usd"),
        ("obstacle", "S_TrafficCone.usd"),
        ("thin_post", "S_TrafficCone.usd"),
    ):
        asset = getattr(cfg.scene, name)
        assert isinstance(asset.spawn, sim_utils.UsdFileCfg)
        assert asset.spawn.usd_path.endswith("/" + filename)
        assert asset.init_state.pos[2] == 0.0
    for name in ("tall_box", "wall_left", "wall_right"):
        asset = getattr(cfg.scene, name)
        assert isinstance(asset.spawn, sim_utils.CuboidCfg)
        assert tuple(asset.spawn.size[:2]) == tuple(cfg.obstacle_footprints[name].size)
    assert cfg.scene.tall_box.spawn.size == (0.4, 0.3, 0.12)
    assert tuple(cfg.obstacle_footprints["cylinder"].size) == pytest.approx((0.463263705, 0.575002059))
    assert cfg.events.obstacle_field.params["spacing_scale"] == 1.6
    assert cfg.scene.env_spacing == 14.0
    assert cfg.terminations.field_boundary.params["extent"] == 5.0


@pytest.fixture(autouse=True)
def clear_ray_mesh_cache():
    """Scenes reuse prim paths for different geometry; legacy meshes outlive a scene."""
    BaseMultiMeshRayCaster.meshes.clear()
    yield
    BaseMultiMeshRayCaster.meshes.clear()


@pytest.fixture
def walking_policy_path(tmp_path):
    """Portable frozen-controller asset shared by the two distinct integration contracts."""
    walking = torch.nn.Linear(61, 14)
    torch.nn.init.zeros_(walking.weight)
    torch.nn.init.zeros_(walking.bias)
    policy_path = str(tmp_path / "walking.pt")
    torch.jit.trace(walking, torch.zeros(1, 61)).save(policy_path)
    return policy_path


def test_navigation_action_stops_for_missing_stale_and_zero_requests_then_recovers():
    """Unsafe cached-frame states bypass slew limits; a new frame restores bounded acceleration."""
    count = 5
    requested = torch.tensor([[0.2, 0.0, 0.0]] * count)
    requested[3] = 0.0
    cfg = MicroDuckWalkingActionCfg()
    sensor = SimpleNamespace(
        delivered_sample_time=torch.tensor([1.0, 1.0, -1.0, 1.0, 1.0]),
        sample_age=torch.tensor([cfg.maximum_sensor_age, cfg.maximum_sensor_age + 0.01, 0.0, 0.0, 0.0]),
    )
    # Only cached adapter inputs are supplied; execute the real native method and
    # inherited output buffer without constructing or calling a walking policy.
    action = object.__new__(MicroDuckWalkingAction)
    action.cfg = cfg
    action._env = SimpleNamespace(
        device="cpu",
        scene={"head_tof": sensor},
        command_manager=SimpleNamespace(get_command=lambda _name: requested),
    )
    action._velocity_limits = torch.tensor(cfg.velocity_limits)
    action._command_step = torch.tensor(cfg.acceleration_limits) * 0.02
    action._raw_actions = torch.tensor([[0.18, 0.02, 0.10]] * count)
    action.previous_velocity_commands = torch.zeros_like(action._raw_actions)
    previous = action.velocity_commands.clone()
    action.process_actions(torch.zeros(count, 3))
    torch.testing.assert_close(action.previous_velocity_commands, previous, atol=0.0, rtol=0.0)
    torch.testing.assert_close(action.velocity_commands[[0, 4]], torch.tensor([[0.2, 0.0, 0.04]] * 2))
    torch.testing.assert_close(action.velocity_commands[1:4], torch.zeros(3, 3), atol=0.0, rtol=0.0)
    assert torch.isfinite(action.velocity_commands).all()

    sensor.delivered_sample_time.fill_(2.0)
    sensor.sample_age.zero_()
    requested[3, 0] = 0.2
    action.process_actions(torch.zeros(count, 3))
    torch.testing.assert_close(action.velocity_commands[1:4], torch.tensor([[0.02, 0.0, 0.0]] * 3))
    torch.testing.assert_close(action.previous_velocity_commands[1:4], torch.zeros(3, 3), atol=0.0, rtol=0.0)


@pytest.mark.integration
@pytest.mark.kitless
@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_microduck_native_navigation_terminations_and_episode_reset(device, walking_policy_path, record_property):
    """Real contact, fall and timeout propagate through native navigation accounting and autoreset."""
    cfg = parse_env_cfg(TASK, device=device, num_envs=16384, overrides=("presets=backlash",))
    cfg.seed = 42
    cfg.actions.joint_pos.policy_path = walking_policy_path
    sim_utils.create_new_stage()
    env = gym.make(TASK, cfg=cfg).unwrapped
    try:
        env.reset()
        assert isinstance(env.command_manager.get_term("base_velocity"), MicroDuckNavigationCommand)
        robot = env.scene["robot"]
        # Env0 intersects a box surface and also reaches its horizon;
        # env1 is inverted above all obstacles; env2 reaches only its horizon.
        # No native termination, command, sensor or reset event is substituted.
        contact_id = torch.tensor([0], device=env.device)
        pose = env.scene["crate"].data.root_pose_w.torch[contact_id].clone()
        pose[:, :2] = robot.data.root_link_pos_w.torch[contact_id, :2]
        # Put a surface into the robot, rather than teleporting it inside a mesh.
        pose[:, 0] += cfg.obstacle_footprints["crate"].size[0] / 2.0 + 0.02
        pose[:, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=env.device)
        env.scene["crate"].write_root_pose_to_sim_index(root_pose=pose, env_ids=contact_id)
        fall_id = torch.tensor([1], device=env.device)
        pose = torch.cat(
            (robot.data.root_link_pos_w.torch[fall_id], robot.data.root_link_quat_w.torch[fall_id]), dim=-1
        )
        pose[:, 2] = 1.0
        pose[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)
        robot.write_root_pose_to_sim_index(root_pose=pose, env_ids=fall_id)
        robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros(1, 6, device=env.device), env_ids=fall_id)
        env.episode_length_buf[[0, 2]] = env.max_episode_length - 1
        env.sim.forward()
        env.scene.update(0.0)

        actions = torch.zeros(env.num_envs, 3, device=env.device)
        with torch.inference_mode():
            observations, rewards, terminated, truncated, extras = env.step(actions)
        assert terminated.nonzero().flatten().tolist() == [0, 1]
        assert truncated.nonzero().flatten().tolist() == [0, 2]
        assert (env.episode_length_buf[:3] == 0).all()
        assert (env.episode_length_buf[3:] == 1).all()
        for name in ("collision_rate", "fall_rate", "horizon_survival_rate"):
            assert extras["log"]["Metrics/base_velocity/" + name].item() == pytest.approx(1.0 / 3.0)
        assert extras["log"]["Metrics/base_velocity/field_boundary_rate"] == 0.0
        assert torch.isfinite(rewards).all()
        assert all(torch.isfinite(values).all() for values in observations.values())
        record_property("native_terminal_envs", json.dumps({"terminated": [0, 1], "truncated": [0, 2]}))

        # A clean episode after the contact/fall reset must not inherit either
        # failure. The same environment now ends solely at the native horizon.
        env.episode_length_buf[0] = env.max_episode_length - 1
        with torch.inference_mode():
            _, rewards, terminated, truncated, extras = env.step(actions)
        assert not terminated.any()
        assert truncated.nonzero().flatten().tolist() == [0]
        assert env.episode_length_buf[0] == 0
        for name in ("collision_rate", "fall_rate", "field_boundary_rate"):
            assert extras["log"]["Metrics/base_velocity/" + name] == 0.0
        assert extras["log"]["Metrics/base_velocity/horizon_survival_rate"] == 1.0
        assert torch.isfinite(rewards).all()
    finally:
        env.close()
        SimulationContext.clear_instance()


@pytest.mark.integration
@pytest.mark.kitless
@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
@pytest.mark.parametrize("overrides", [(), ("presets=backlash",)], ids=["standard", "backlash"])
def test_microduck_sensors_and_frozen_controller(device, overrides, walking_policy_path, record_property, monkeypatch):
    """Check independent obstacle ranges, finite range limits, acquisition rate and one PPO update."""
    cfg = parse_env_cfg(TASK, device=device, num_envs=2, overrides=overrides)
    cfg.seed = 42
    # This portable sensor/controller fixture suppresses navigation episode accounting.
    cfg.commands.base_velocity.class_type = MicroDuckVelocityCommand
    cfg.rewards.goal_arrival = None
    cfg.rewards.barrier_bypass = None
    cfg.actions.joint_pos.policy_path = walking_policy_path
    cfg.events.reset_base.params["pose_range"] = {}
    cfg.events.push_robot = None
    cfg.events.obstacle_field = None
    cfg.scene.head_tof.randomization.enabled = False
    cfg.terminations.obstacle_collision = None
    cfg.terminations.fell_over = None
    # Analytic ray-distance checks use one known box; public-mesh coverage is below.
    cfg.scene.obstacle.spawn = sim_utils.CuboidCfg(
        size=(0.1, 0.6, 10.0),
        rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True),
        collision_props=sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True),
    )
    cfg.scene.obstacle.init_state.pos = (0.6, 0.0, 0.15)
    cfg.obstacle_footprints["obstacle"].size = (0.1, 0.6)
    sim_utils.create_new_stage()
    env = gym.make(TASK, cfg=cfg).unwrapped
    try:
        env.reset()
        # The standing hardware recording constrains the nominal pose, not just its noise.
        sensor = env.scene["head_tof"]
        robot = env.scene["robot"]
        for name in ("neck_pitch", "head_pitch"):
            joint = robot.joint_names.index(name)
            torch.testing.assert_close(
                robot.data.joint_pos.torch[:, joint], torch.full((2,), 0.3491, device=env.device), atol=1e-6, rtol=0.0
            )
        torch.testing.assert_close(
            sensor.data.pos_w.torch[:, 2], torch.full((2,), 0.22, device=env.device), atol=0.001, rtol=0.0
        )
        forward = quat_apply(
            sensor.data.quat_w_world.torch, torch.tensor([[1.0, 0.0, 0.0]], device=env.device).expand(2, -1)
        )
        elevation = torch.rad2deg(torch.asin(forward[:, 2]))
        torch.testing.assert_close(elevation, torch.full((2,), -20.0, device=env.device), atol=0.1, rtol=0.0)
        rows = sensor.observation[..., 0].mean(dim=2)[:, [7, 6, 5]]
        torch.testing.assert_close(
            rows, torch.tensor([[0.32, 0.38, 0.45]], device=env.device).expand(2, -1), atol=0.012, rtol=0.0
        )
        record_property(
            "reference_pose",
            json.dumps(
                {
                    "height_m": sensor.data.pos_w.torch[:, 2].tolist(),
                    "elevation_deg": elevation.tolist(),
                    "rows_7_6_5_m": rows.tolist(),
                }
            ),
        )
        # Moving either serial pitch joint must move the sensor, without simulating a gait.
        home_q = robot.data.joint_pos.torch.clone()
        for name in ("neck_pitch", "head_pitch"):
            q = home_q.clone()
            q[:, robot.joint_names.index(name)] += 0.1
            robot.write_joint_state_to_sim_index(position=q, velocity=torch.zeros_like(q))
            env.sim.forward()
            env.scene.update(0.0)
            sensor.reset()
            changed_forward = quat_apply(
                sensor.data.quat_w_world.torch, torch.tensor([[1.0, 0.0, 0.0]], device=env.device).expand(2, -1)
            )
            angle_change = torch.acos((forward * changed_forward).sum(-1).clamp(-1.0, 1.0))
            record_property(name + "_sensor_angle_change_rad", json.dumps(angle_change.tolist()))
            torch.testing.assert_close(angle_change, torch.full((2,), 0.1, device=env.device), atol=1e-4, rtol=0.0)
        robot.write_joint_state_to_sim_index(position=home_q, velocity=torch.zeros_like(home_q))
        env.sim.forward()
        env.scene.update(0.0)
        sensor.reset()

        # Native open-floor acquisitions keep near floor visible and stochastically lose
        # distant floor returns; repeatedly reading a delivered frame never redraws it.
        obstacle_poses = {name: env.scene[name].data.root_pose_w.torch.clone() for name in OBSTACLE_CELLS}
        for name, obstacle_pose in obstacle_poses.items():
            outside_view = obstacle_pose.clone()
            outside_view[:, 1] += 4.0
            env.scene[name].write_root_pose_to_sim_index(root_pose=outside_view)
        env.sim.forward()
        noise = sensor.cfg.randomization
        noise.enabled = True
        noise.mount_pitch_degrees = (0.0, 0.0)
        noise.mount_height_m = (0.0, 0.0)
        noise.latency_frames = (0, 0)
        noise.frame_dropout_probability = 0.0
        noise.distance_bias = (0.0, 0.0)
        noise.zone_bias = (0.0, 0.0)
        noise.distance_scale = (1.0, 1.0)
        noise.noise_std = (0.0, 0.0)
        noise.distance_noise_fraction = 0.0
        noise.floor_validity_midpoint_offset_m = (0.0, 0.0)
        noise.near_floor_edge_impairment_probability = 0.0
        sensor.reset()
        masks = []
        for _ in range(256):
            sensor.update(sensor.sample_period)
            masks.append(sensor.observation[..., 0] > 0.0)
        masks = torch.stack(masks)
        floor = sensor.ray_hits_w.torch[..., 2].reshape(2, 8, 8).abs() < 1e-3
        depths = sensor.data.output[sensor.output_name].torch[..., 0]
        near = floor & (depths >= 0.12) & (depths <= 0.55)
        far = floor & (depths > 0.65)
        record_property("open_floor_valid_fraction_by_row", json.dumps(masks.float().mean(dim=(0, 1, 3)).tolist()))
        assert near.any() and far.any()
        assert masks[:, near].all()
        assert masks[:, far].float().mean() < 0.01
        assert (masks[1:] != masks[:-1]).any()  # Transition-zone validity flickers.
        # More reflective floor conditions must change far returns, not remove near floor.
        transition = floor & (depths > 0.55) & (depths < 0.70)
        assert transition.any()
        noise.floor_validity_midpoint_offset_m = (0.13, 0.13)
        sensor.reset()
        reflective_masks = []
        for _ in range(256):
            sensor.update(sensor.sample_period)
            reflective_masks.append(sensor.observation[..., 0] > 0.0)
        reflective_masks = torch.stack(reflective_masks)
        assert reflective_masks[:, near].all()
        assert reflective_masks[:, transition].float().mean() > masks[:, transition].float().mean() + 0.3

        # Force opposite impaired edges in the existing native scene; only actual near-floor
        # hits may disappear, and every other range must match the healthy acquisition.
        near_edge = torch.zeros_like(near)
        near_edge[0, :, 0] = near[0, :, 0]
        near_edge[1, :, -1] = near[1, :, -1]
        assert near_edge.flatten(1).any(dim=1).all()
        with monkeypatch.context() as patch:
            patch.setattr(torch, "rand_like", lambda values: torch.full_like(values, 0.5))
            sensor.update(sensor.sample_period)
            healthy = sensor.observation[..., 0].clone()
            assert (healthy[transition] > 0.0).any()
            noise.near_floor_edge_impairment_probability = 1.0
            for survival, retained in ((0.55, True), (0.02, False)):
                noise.near_floor_edge_survival_probability = (survival, survival)
                sensor.reset()
                sensor._near_floor_edge_column[:, 0, 0, 0] = torch.tensor([0, 7], device=env.device)
                sensor.update(sensor.sample_period)
                actual = sensor.observation[..., 0]
                torch.testing.assert_close(actual[~near_edge], healthy[~near_edge], atol=0.0, rtol=0.0)
                assert ((actual[near_edge] > 0.0) == retained).all()
            observation = cfg.observations.tof.ranges_and_validity.func(env)
            assert torch.equal(observation[:, 64:].bool(), observation[:, :64] > 0.0)
            assert (observation[:, :64][near_edge.flatten(1)] == 0.0).all()

            # New settings apply only on reset; env1 retains its sampled impaired episode.
            noise.near_floor_edge_impairment_probability = 0.0
            sensor.reset(torch.tensor([0], device=env.device))
            sensor.update(sensor.sample_period)
            assert (sensor.observation[0, ..., 0][near[0]] > 0.0).all()
            assert (sensor.observation[1, ..., 0][near_edge[1]] == 0.0).all()
            # A fresh acquisition can retain these same edge zones: this is stochastic loss.
            patch.setattr(torch, "rand_like", lambda values: torch.full_like(values, 0.99))
            sensor.update(sensor.sample_period)
            assert (sensor.observation[..., 0][near] > 0.0).all()

        noise.near_floor_edge_impairment_probability = 1.0
        noise.near_floor_edge_survival_probability = (0.25, 0.25)
        sensor.reset()
        sensor._near_floor_edge_column[:, 0, 0, 0] = torch.tensor([0, 7], device=env.device)
        edge_masks = []
        for _ in range(64):
            sensor.update(sensor.sample_period)
            edge_masks.append(sensor.observation[..., 0] > 0.0)
        edge_masks = torch.stack(edge_masks)
        assert edge_masks[:, near & ~near_edge].all()
        for index in range(2):
            edge_validity = edge_masks[:, index, near_edge[index]].float().mean()
            assert 0.15 < edge_validity < 0.35
        assert (edge_masks[1:, near_edge] != edge_masks[:-1, near_edge]).any()
        noise.enabled = False
        sensor.update(sensor.sample_period)
        torch.testing.assert_close(sensor.observation, sensor.data.output[sensor.output_name].torch)
        assert (sensor.observation[..., 0][near] > 0.0).all()
        sensor.reset()
        for name, obstacle_pose in obstacle_poses.items():
            env.scene[name].write_root_pose_to_sim_index(root_pose=obstacle_pose)
        env.sim.forward()
        sensor.reset()
        action = env.action_manager.get_term("joint_pos")
        frozen = {name: value.clone() for name, value in action.policy.state_dict().items()}
        assert env.action_manager.total_action_dim == 3
        assert action.low_level_actions.shape == (2, 14)
        assert not any(parameter.requires_grad for parameter in action.policy.parameters())
        box = env.scene["obstacle"]
        pose = box.data.root_pose_w.torch.clone()
        pose[1, 0] -= 0.3
        box.write_root_pose_to_sim_index(root_pose=pose)
        with torch.inference_mode():
            for _ in range(10):
                obs, *_ = env.step(torch.zeros(2, 3, device=env.device))

        assert obs["tof"].shape == (2, 1290)
        sensor = env.scene["head_tof"]
        ranges = sensor.data.output[sensor.output_name].torch[..., 0]
        box_face_x = pose[:, 0] - cfg.scene.obstacle.spawn.size[0] / 2.0
        optical_forward = quat_apply(
            sensor.data.quat_w_world.torch, torch.tensor([[1.0, 0.0, 0.0]], device=env.device).expand(2, -1)
        )
        # Upper central rays hit the wall; lower rays can hit the closer floor at this tilt.
        hits = sensor.ray_hits_w.torch.reshape(2, 8, 8, 3)
        torch.testing.assert_close(
            hits[:, :2, 3:5, 0], box_face_x[:, None, None].expand(-1, 2, 2), atol=0.001, rtol=0.0
        )
        projected = ((hits - sensor.data.pos_w.torch[:, None, None]) * optical_forward[:, None, None]).sum(-1)
        torch.testing.assert_close(ranges[:, :2, 3:5], projected[:, :2, 3:5], atol=1e-5, rtol=0.0)
        assert (ranges[1] > 0.0).all()
        # Non-floor returns, including close outer-column walls, keep the generic prior.
        near_wall_edge = (hits[..., 2].abs() >= 1e-3) & (ranges >= 0.12) & (ranges <= 0.55)
        near_wall_edge[:, :, 1:-1] = False
        assert near_wall_edge.any()
        noise.enabled = True
        noise.near_floor_edge_survival_probability = (0.02, 0.02)
        sensor.reset()
        sensor._near_floor_edge_column[:, 0, 0, 0] = torch.tensor([0, 7], device=env.device)
        masks = []
        for _ in range(512):
            sensor.update(sensor.sample_period)
            masks.append(sensor.observation[..., 0] > 0.0)
        masks = torch.stack(masks)
        wall_validity = masks[:, :, :2, 3:5].float().mean(dim=(0, 1, 3))
        record_property("wall_top_row_valid_fractions", json.dumps(wall_validity.tolist()))
        torch.testing.assert_close(wall_validity, torch.tensor([0.97, 0.97], device=env.device), atol=0.04, rtol=0.0)
        assert (masks[1:, :, :2, 3:5] != masks[:-1, :, :2, 3:5]).any()
        torch.testing.assert_close(
            masks[:, near_wall_edge].float().mean(), torch.tensor(0.97, device=env.device), atol=0.04, rtol=0.0
        )
        noise.near_floor_edge_impairment_probability = 0.0
        observation = cfg.observations.tof.ranges_and_validity.func(env)
        assert torch.equal(observation[:, 64:].bool(), observation[:, :64] > 0.0)
        assert (observation[:, :64][observation[:, 64:] == 0.0] == 0.0).all()
        noise.enabled = False
        sensor.reset()

        # A known finite box return must become invalid when outside the configured range.
        # Lower the cutoff instead of moving the box: downward rays otherwise hit the nearer floor.
        sensor.cfg.max_distance = 0.15
        with torch.inference_mode():
            for _ in range(16):
                obs, *_ = env.step(torch.zeros(2, 3, device=env.device))
        # Close floor returns may remain valid; every retained range must obey the cutoff.
        frames = obs["tof"][:, :1280].reshape(2, 10, 128)
        distances, valid = frames[..., :64], frames[..., 64:]
        assert (distances <= sensor.cfg.max_distance).all()
        assert torch.equal(valid.bool(), distances > 0.0)
        assert (sensor.data.output[sensor.output_name].torch <= sensor.cfg.max_distance).all()
        sensor.cfg.max_distance = 4.0

        # Timestamp transitions detect real captures, independent of repeated observation reads.
        previous_time = sensor.sample_time.clone()
        captures = 0
        walking_policy = action.policy
        walking_calls = []

        def counted_walker(observations):
            walking_calls.append(None)
            return walking_policy(observations)

        action.policy = counted_walker
        with torch.inference_mode():
            for _ in range(100):
                env.step(torch.zeros(2, 3, device=env.device))
                timestamp = sensor.sample_time.clone()
                captures += int(timestamp[0] > previous_time[0] + 1e-6)
                previous_time = timestamp
        action.policy = walking_policy
        assert len(walking_calls) == 100  # Two simulated seconds at the native walking rate of 50 Hz.
        assert captures == 30, captures

        # Partial reset clears only that environment's low-level history.
        action.low_level_actions.fill_(0.1)
        with torch.inference_mode():
            action.reset(torch.tensor([0], device=env.device))
        assert (action.low_level_actions[0] == 0).all()
        assert (action.low_level_actions[1] == 0.1).all()
        requested = env.command_manager.get_command("base_velocity")
        requested.zero_()
        action.process_actions(torch.ones(2, 3, device=env.device))
        assert (action.processed_actions == 0).all()

        # Deterministic corruption makes calibration, frame delay and reset isolation measurable.
        tof_noise = sensor.cfg.randomization
        tof_noise.enabled = True
        tof_noise.latency_frames = (1, 1)
        tof_noise.frame_dropout_probability = 0.0
        tof_noise.distance_bias = (0.05, 0.05)
        tof_noise.zone_bias = (0.0, 0.0)
        tof_noise.distance_scale = (1.0, 1.0)
        tof_noise.noise_std = (0.0, 0.0)
        tof_noise.distance_noise_fraction = 0.0
        tof_noise.invalid_zone_probability = 0.0
        tof_noise.floor_validity_transition_m = None
        previous_frame = None
        previous_time = None
        checked = False
        with torch.inference_mode():
            sensor.reset()
            for _ in range(15):
                env.step(torch.zeros(2, 3, device=env.device))
                raw = sensor.data.output[sensor.output_name].torch.clone()
                current_time = sensor.sample_time.clone()
                if previous_time is not None and torch.equal(current_time, previous_time):
                    continue
                if previous_frame is not None:
                    torch.testing.assert_close(sensor.delivered_sample_time, previous_time)
                    valid = (previous_frame >= 0.12) & (previous_frame <= 3.95)
                    expected_frame = torch.where(valid, previous_frame + 0.05, 0.0)
                    torch.testing.assert_close(sensor.observation, expected_frame, atol=1e-6, rtol=0.0)
                    if previous_time[0] >= sensor.sample_period:
                        assert (sensor.sample_age >= sensor.sample_period - env.step_dt - 1e-6).all()
                    checked = True
                previous_frame = raw
                previous_time = current_time
            assert checked
            frame = sensor.observation.clone()
            timestamp = sensor.delivered_sample_time.clone()
            # Re-reading the cached observation must neither redraw noise nor advance time.
            torch.testing.assert_close(sensor.observation, frame)
            torch.testing.assert_close(sensor.delivered_sample_time, timestamp)
            sensor.reset(torch.tensor([0], device=env.device))
            assert (sensor.observation[0] == 0).all()
            assert sensor.delivered_sample_time[0] == -1.0
            torch.testing.assert_close(sensor.observation[1], frame[1])
            torch.testing.assert_close(sensor.delivered_sample_time[1], timestamp[1])

        # Mount offsets must change physical rays, stay head-relative, and reset per environment.
        tof_noise.enabled = False
        sensor.reset()
        nominal_pos = sensor.data.pos_w.torch.clone()
        nominal_quat = sensor.data.quat_w_world.torch.clone()
        tof_noise.enabled = True
        tof_noise.mount_pitch_degrees = (10.0, 10.0)
        tof_noise.mount_height_m = (0.02, 0.02)
        sensor.reset(torch.tensor([0], device=env.device))
        actual_pos = sensor.data.pos_w.torch.clone()
        actual_quat = sensor.data.quat_w_world.torch.clone()
        vertical = torch.tensor([[0.0, 0.0, 0.02]], device=env.device)
        expected_pos = nominal_pos[:1] + quat_apply(nominal_quat[:1], vertical)
        torch.testing.assert_close(actual_pos[:1], expected_pos, atol=1e-6, rtol=0.0)
        pitch = math.radians(10.0)
        expected_forward = quat_apply(
            nominal_quat[:1],
            torch.tensor([[math.cos(pitch), 0.0, -math.sin(pitch)]], device=env.device),
        )
        forward = torch.tensor([[1.0, 0.0, 0.0]], device=env.device)
        torch.testing.assert_close(quat_apply(actual_quat[:1], forward), expected_forward, atol=2e-6, rtol=0.0)
        torch.testing.assert_close(actual_pos[1], nominal_pos[1], atol=0.0, rtol=0.0)
        torch.testing.assert_close(actual_quat[1], nominal_quat[1], atol=0.0, rtol=0.0)
        # Acquisitions without physical movement retain the sampled mount instead of redrawing it.
        for _ in range(8):
            sensor.update(env.step_dt)
            torch.testing.assert_close(sensor.data.pos_w.torch, actual_pos, atol=0.0, rtol=0.0)
            torch.testing.assert_close(sensor.data.quat_w_world.torch, actual_quat, atol=0.0, rtol=0.0)
        tof_noise.enabled = False
        sensor.reset()
        torch.testing.assert_close(sensor.data.pos_w.torch, nominal_pos, atol=1e-6, rtol=0.0)
        torch.testing.assert_close(sensor.data.quat_w_world.torch, nominal_quat, atol=1e-6, rtol=0.0)

        from rsl_rl.runners import OnPolicyRunner

        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

        agent = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
        agent.num_steps_per_env = 4
        agent.algorithm.num_learning_epochs = 1
        agent.algorithm.num_mini_batches = 1
        with torch.inference_mode():
            vec_env = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
        runner = OnPolicyRunner(vec_env, agent.to_dict(), device=device)
        actor = runner.alg.actor
        trainable_before = {name: p.detach().clone() for name, p in actor.named_parameters()}
        runner.learn(num_learning_iterations=1, init_at_random_ep_len=False)
        assert any(not torch.equal(p, trainable_before[name]) for name, p in actor.named_parameters())
        # Both sampled exploration and deployment means must obey the executed action bounds.
        with torch.inference_mode():
            observations = vec_env.get_observations()
            samples = torch.stack([actor(observations, stochastic_output=True) for _ in range(256)])
            means = actor(observations)
        assert torch.isfinite(samples).all() and torch.isfinite(means).all()
        assert (samples.abs() <= 1.0).all()
        assert (means.abs() <= 1.0).all()
        assert (actor.output_std <= 1.0).all()
        for name, value in action.policy.state_dict().items():
            torch.testing.assert_close(value, frozen[name], rtol=0.0, atol=0.0)
        assert all(p.grad is None for p in action.policy.parameters())
        # A numerically uniform forward Beta must brake, while a healthy symmetric mean
        # remains a valid zero correction. Exercise native inference, export and the adapter.
        with torch.inference_mode():
            head = actor.mlp[-2]
            head.weight.zero_()
            head.bias.zero_()
            healthy = actor(observations)
            torch.testing.assert_close(healthy, torch.zeros_like(healthy))
            assert (sensor.delivered_sample_time >= 0).all()
            assert (sensor.sample_age <= action.cfg.maximum_sensor_age).all()
            requested = env.command_manager.get_command("base_velocity")
            requested.zero_()
            requested[:, 0] = 0.15
            action.reset()
            for _ in range(8):
                action.process_actions(healthy)
            torch.testing.assert_close(action.velocity_commands, requested)

            # Softplus(-100)+1 rounds to1: the old Beta mean passes the request through.
            head.bias.fill_(-100.0)
            braking = actor(observations)
            expected = torch.tensor([[-1.0, 0.0, 0.0]], device=device).expand(2, -1)
            torch.testing.assert_close(braking, expected)
            exported = torch.jit.script(actor.as_jit())
            inputs = torch.cat([observations[group] for group in actor.obs_groups], dim=-1)
            torch.testing.assert_close(exported(inputs), expected)
            for _ in range(8):
                action.process_actions(braking)
            torch.testing.assert_close(action.velocity_commands, torch.zeros_like(requested))

        # Both blocked and corner approaches leave lateral exits around the front obstacle.
        # Reset only env0 and verify the other environment's physical obstacles stay put.
        with torch.inference_mode():
            env.reset()
            unchanged = {name: env.scene[name].data.root_pose_w.torch[1].clone() for name in OBSTACLE_CELLS}
            for layout in ("blocked", "corner"):
                reset_obstacle_field(env, torch.tensor([0], device=env.device), jitter=0.15, layout=layout)
                assert obstacle_clearances(env)[0].amin() > 0.05
                robot = env.scene["robot"]
                heading = yaw_quat(robot.data.root_link_quat_w.torch[:1])
                local = {}
                for name in OBSTACLE_CELLS:
                    torch.testing.assert_close(env.scene[name].data.root_pose_w.torch[1], unchanged[name])
                    local[name] = quat_apply_inverse(
                        heading, (env.scene[name].data.root_pos_w.torch[:1] - robot.data.root_link_pos_w.torch[:1])
                    )[0]
                assert 0.39999 <= local["obstacle"][0] <= 1.20001
                assert abs(float(local["obstacle"][1])) < 1e-5
                if layout == "corner":
                    assert 0.26999 <= abs(float(local["wall_left"][1])) <= 0.34001
                assert abs(float(local["obstacle"][0] - local["wall_left"][0]) - 0.52) < 1e-5
                if layout == "blocked":
                    assert abs(float(local["wall_right"][1] + local["wall_left"][1])) < 1e-5
                    # Each exit exceeds the 0.24 m body diameter plus 0.16 m clearance.
                    side_gap = abs(local["wall_left"][1]) - 0.06 - env.scene["obstacle"].cfg.spawn.size[1] / 2
                    assert side_gap >= 0.40
                else:
                    assert abs(float(local["wall_right"][1])) > 1.0
    finally:
        env.close()
        SimulationContext.clear_instance()


def test_randomized_layout_spawn_clearance_and_partial_reset():
    """Random resets keep spawn clear, vary approaches, and leave other envs untouched."""
    from isaaclab_tasks.contrib.microduck_obstacle.obstacle_avoidance_env_cfg import (
        OBSTACLE_FOOTPRINTS,
        MicroDuckObstacleEventsCfg,
        MicroDuckObstacleSceneCfg,
    )

    count = 8192
    torch.manual_seed(7721)
    cfg = MicroDuckObstacleSceneCfg()

    class Scene(dict):
        pass

    scene = Scene()
    scene.env_origins = torch.zeros(count, 3)
    robot_pose = torch.zeros(count, 7)
    robot_pose[:, 6] = 1.0
    scene["robot"] = SimpleNamespace(
        data=SimpleNamespace(
            root_link_pos_w=SimpleNamespace(torch=robot_pose[:, :3]),
            root_link_quat_w=SimpleNamespace(torch=robot_pose[:, 3:7]),
        )
    )
    for name in OBSTACLE_CELLS:
        state = torch.zeros(count, 13)
        state[:, 6] = 1.0
        scene[name] = SimpleNamespace(
            cfg=getattr(cfg, name),
            data=SimpleNamespace(
                default_root_state=SimpleNamespace(torch=state.clone()),
                root_pos_w=SimpleNamespace(torch=state[:, :3]),
                root_quat_w=SimpleNamespace(torch=state[:, 3:7]),
            ),
            write_root_pose_to_sim_index=lambda root_pose, env_ids, state=state: state.__setitem__(
                (env_ids, slice(0, 7)), root_pose
            ),
            write_root_velocity_to_sim_index=lambda root_velocity, env_ids: None,
        )
    env = SimpleNamespace(
        num_envs=count, device="cpu", scene=scene, cfg=SimpleNamespace(obstacle_footprints=OBSTACLE_FOOTPRINTS)
    )
    reset_obstacle_field(env, None, layout="balanced", spacing_scale=1.6)
    clearance = obstacle_clearances(env)
    assert torch.isfinite(clearance).all()
    assert clearance.amin() > 0.05
    # Geometry changes within each family, not merely between different templates.
    x = scene["obstacle"].data.root_pos_w.torch[:, 0]
    for family in (2, 6, 7):
        values = x[env.microduck_obstacle_layout == family]
        assert values.max() - values.min() > 0.5
    passage = env.microduck_obstacle_layout == 4
    left = scene["wall_left"].data.root_pos_w.torch[passage]
    right = scene["wall_right"].data.root_pos_w.torch[passage]
    gap = (left - right).norm(dim=-1) - 0.12
    occupied = gap > 1.9
    assert 0.4 < occupied.float().mean() < 0.6
    assert gap[~occupied].min() >= 0.71199 and gap[~occupied].max() <= 1.83201
    assert gap[occupied].min() >= 1.99999 and gap[occupied].max() <= 2.30001
    before = {name: scene[name].data.root_pos_w.torch.clone() for name in OBSTACLE_CELLS}
    layout_before = env.microduck_obstacle_layout.clone()
    ids = torch.arange(0, count, 2)
    reset_obstacle_field(env, ids, layout="mixed", spacing_scale=1.6)
    assert obstacle_clearances(env).amin() > 0.05
    torch.testing.assert_close(env.microduck_obstacle_layout[1::2], layout_before[1::2])
    for name in OBSTACLE_CELLS:
        torch.testing.assert_close(scene[name].data.root_pos_w.torch[1::2], before[name][1::2])

    # Custom blocked distances preserve every other family and the RNG stream.
    torch.manual_seed(6173)
    reset_obstacle_field(env, None, layout="balanced", jitter=0.0)
    before = {name: scene[name].data.root_pos_w.torch.clone() for name in OBSTACLE_CELLS}
    rng_before = torch.get_rng_state()
    torch.manual_seed(6173)
    reset_obstacle_field(env, None, layout="balanced", jitter=0.0, blocked_front_distance_range=(0.30, 0.34))
    assert torch.equal(torch.get_rng_state(), rng_before)
    blocked = env.microduck_obstacle_layout == 6
    for name in OBSTACLE_CELLS:
        torch.testing.assert_close(scene[name].data.root_pos_w.torch[~blocked], before[name][~blocked], rtol=0, atol=0)
    front = scene["obstacle"].data.root_pos_w.torch[blocked]
    assert front[:, 0].min() >= 0.30 and front[:, 0].max() <= 0.34
    assert front[:, 0].max() - front[:, 0].min() > 0.039
    torch.testing.assert_close(front[:, 1], torch.zeros_like(front[:, 1]))
    for name in ("wall_left", "wall_right"):
        wall = scene[name].data.root_pos_w.torch[blocked]
        torch.testing.assert_close(front[:, 0] - wall[:, 0], torch.full_like(front[:, 0], 0.52))
        torch.testing.assert_close(wall[:, 1:], before[name][blocked, 1:], rtol=0, atol=0)

    # Fixed-distance partial resets still leave unselected environments untouched.
    before = {name: scene[name].data.root_pos_w.torch.clone() for name in OBSTACLE_CELLS}
    reset_obstacle_field(env, ids, layout="blocked", jitter=0.0, blocked_front_distance_range=(0.32, 0.32))
    torch.testing.assert_close(scene["obstacle"].data.root_pos_w.torch[ids, 0], torch.full((len(ids),), 0.32))
    for name in OBSTACLE_CELLS:
        torch.testing.assert_close(scene[name].data.root_pos_w.torch[1::2], before[name][1::2], rtol=0, atol=0)
    for invalid in ((0.30,), (0.0, 0.34), (0.34, 0.30), (math.nan, 0.34), (0.30, math.inf)):
        with pytest.raises(ValueError, match="finite, positive, ordered pair"):
            reset_obstacle_field(env, None, blocked_front_distance_range=invalid)

    # Omitted and explicit None preserve the same full, partial and empty reset stream.
    for selected in (None, ids, torch.empty(0, dtype=torch.long)):
        torch.manual_seed(1843)
        reset_obstacle_field(env, selected, layout="mixed")
        before = {
            name: torch.cat((scene[name].data.root_pos_w.torch, scene[name].data.root_quat_w.torch), -1).clone()
            for name in OBSTACLE_CELLS
        }
        layout_before = env.microduck_obstacle_layout.clone()
        rng_before = torch.get_rng_state()
        torch.manual_seed(1843)
        reset_obstacle_field(env, selected, layout="mixed", layout_weights=None)
        assert torch.equal(torch.get_rng_state(), rng_before)
        assert torch.equal(env.microduck_obstacle_layout, layout_before)
        for name in OBSTACLE_CELLS:
            pose = torch.cat((scene[name].data.root_pos_w.torch, scene[name].data.root_quat_w.torch), -1)
            torch.testing.assert_close(pose, before[name], rtol=0, atol=0)

    # Exercise the configured training distribution, including rare difficult layouts.
    weights = MicroDuckObstacleEventsCfg().obstacle_field.params["layout_weights"]
    assert set(weights) == set(OBSTACLE_LAYOUT_NAMES)
    assert {name for name, weight in weights.items() if weight > 0} == {
        "open",
        "barrier",
        "passage",
        "thin_low",
        "blocked",
        "corner",
        "scattered",
    }
    torch.manual_seed(2817)
    reset_obstacle_field(env, None, layout="mixed", layout_weights=weights)
    frequencies = env.microduck_obstacle_layout.bincount(minlength=len(OBSTACLE_LAYOUT_NAMES)).float() / count
    for i, name in enumerate(OBSTACLE_LAYOUT_NAMES):
        assert bool(frequencies[i] > 0) == (weights[name] > 0)
        assert abs(float(frequencies[i]) - weights[name]) < 0.02
    before = {name: scene[name].data.root_pos_w.torch.clone() for name in OBSTACLE_CELLS}
    layout_before = env.microduck_obstacle_layout.clone()
    reset_obstacle_field(env, ids, layout="mixed", layout_weights=weights)
    assert torch.equal(env.microduck_obstacle_layout[1::2], layout_before[1::2])
    for name in OBSTACLE_CELLS:
        torch.testing.assert_close(scene[name].data.root_pos_w.torch[1::2], before[name][1::2])
    rng_before = torch.get_rng_state()
    for invalid in ({}, {"unknown": 1.0}, {"open": -1}, {"open": math.nan}, {"open": math.inf}, {"open": 0.0}):
        with pytest.raises(ValueError, match="Layout weights require"):
            reset_obstacle_field(env, None, layout="mixed", layout_weights=invalid)
        assert torch.equal(torch.get_rng_state(), rng_before)
    with pytest.raises(ValueError, match="Layout weights require"):
        reset_obstacle_field(env, None, layout="blocked", layout_weights=weights)
    assert torch.equal(torch.get_rng_state(), rng_before)

    # Shuffling changes asset roles, not the set of layout positions or untouched rows.
    groups = (("obstacle", "low_block", "thin_post", "tall_box"), ("crate", "cylinder"))
    for layout in ("ring", "balanced"):
        torch.manual_seed(42)
        reset_obstacle_field(env, None, layout=layout, jitter=0.0, spacing_scale=1.6)
        before = {name: scene[name].data.root_pos_w.torch.clone() for name in OBSTACLE_CELLS}
        torch.manual_seed(42)
        reset_obstacle_field(env, None, layout=layout, jitter=0.0, spacing_scale=1.6, shuffle_groups=groups)
        positions = torch.stack([scene[name].data.root_pos_w.torch for name in OBSTACLE_CELLS], dim=1)
        front = positions[torch.arange(count), env.microduck_front_obstacle_index, :2]
        torch.testing.assert_close(front, before["obstacle"][:, :2])
        front_before = env.microduck_front_obstacle_index.clone()
        for group in groups:
            original = torch.stack([before[name][:, :2] for name in group], dim=1)
            shuffled = torch.stack([scene[name].data.root_pos_w.torch[:, :2] for name in group], dim=1)
            distances = (shuffled[:, :, None] - original[:, None]).norm(dim=-1)
            assert distances.amin(-1).max() < 1e-6
            assert distances.amin(-2).max() < 1e-6
            # Each asset reaches every role, including the frontal blocker position.
            for roles in distances.argmin(-1).T:
                assert roles.unique().numel() == len(group)
        for name in ("wall_left", "wall_right"):
            torch.testing.assert_close(scene[name].data.root_pos_w.torch, before[name])
        before = {name: scene[name].data.root_pos_w.torch.clone() for name in OBSTACLE_CELLS}
        reset_obstacle_field(env, ids, layout=layout, spacing_scale=1.6, shuffle_groups=groups)
        torch.testing.assert_close(env.microduck_front_obstacle_index[1::2], front_before[1::2])
        for name in OBSTACLE_CELLS:
            torch.testing.assert_close(scene[name].data.root_pos_w.torch[1::2], before[name][1::2])
        reset_obstacle_field(env, torch.empty(0, dtype=torch.long), layout=layout, shuffle_groups=groups)
    for groups in ((("crate",),), (("crate", "absent"),), (("crate", "crate"),)):
        with pytest.raises(ValueError, match="known, disjoint"):
            reset_obstacle_field(env, None, shuffle_groups=groups)

    # Check scattered geometry with the actual mixed-scene footprint contract, not primitive sizes.
    cfg = parse_env_cfg(TASK, device="cpu", overrides=("presets=backlash",))
    env.cfg = cfg
    for name in OBSTACLE_CELLS:
        scene[name].cfg = getattr(cfg.scene, name)
    torch.manual_seed(117)
    reset_obstacle_field(
        env,
        None,
        layout="scattered",
        spacing_scale=1.6,
        shuffle_groups=cfg.events.obstacle_field.params["shuffle_groups"],
    )
    centers = torch.stack([scene[name].data.root_pos_w.torch[:, :2] for name in OBSTACLE_CELLS], dim=1)
    quats = torch.stack([scene[name].data.root_quat_w.torch for name in OBSTACLE_CELLS], dim=1)
    yaw = 2 * torch.atan2(quats[..., 2], quats[..., 3])
    axes = torch.stack((torch.stack((yaw.cos(), yaw.sin()), -1), torch.stack((-yaw.sin(), yaw.cos()), -1)), -2)
    half_sizes = torch.tensor([cfg.obstacle_footprints[name].size for name in OBSTACLE_CELLS]) * 0.5
    # Independent separating-axis check: no pair of rotated footprint boxes overlaps.
    for left in range(len(OBSTACLE_CELLS)):
        for right in range(left + 1, len(OBSTACLE_CELLS)):
            directions = torch.cat((axes[:, left], axes[:, right]), dim=1)
            distance = ((centers[:, right] - centers[:, left])[:, None] * directions).sum(-1).abs()
            left_radius = ((directions[:, :, None] * axes[:, left, None]).sum(-1).abs() * half_sizes[left]).sum(-1)
            right_radius = ((directions[:, :, None] * axes[:, right, None]).sum(-1).abs() * half_sizes[right]).sum(-1)
            assert (distance > left_radius + right_radius).any(-1).all()
    assert (centers[..., 0] > 0).all()  # Every object begins ahead, including the lateral walls.
    assert obstacle_clearances(env).amin() > 0.1
    # Both sides of each row provide a body-width opening, even with the largest rotated props.
    extent_y = (axes[..., 1].abs() * half_sizes).sum(-1)
    order = centers[:, :6, 0].argsort(-1)
    middle_props = []
    for row in (order[:, :3], order[:, 3:]):
        ys = centers[:, :6, 1].gather(1, row)
        extents = extent_y[:, :6].gather(1, row)
        indices = ys.argsort(-1)
        middle_props.append(row.gather(1, indices[:, 1:2]).squeeze(1))
        ys, extents = ys.gather(1, indices), extents.gather(1, indices)
        gaps = ys[:, 1:] - ys[:, :-1] - extents[:, 1:] - extents[:, :-1]
        assert gaps.amin() > 0.35  # Robot shaping diameter is 0.24 m.
    # Cones must occupy both encounter roles, not only lateral filler positions.
    middle_props = torch.stack(middle_props, dim=1)
    middle_y = centers[:, :6, 1].gather(1, middle_props)
    two_encounters = (middle_y.abs() <= 0.20).all(-1)
    assert two_encounters.float().mean() > 0.45
    cone_indices = [list(OBSTACLE_CELLS).index(name) for name in ("obstacle", "thin_post")]
    center_is_cone = torch.isin(middle_props, torch.tensor(cone_indices))
    for encounter in range(2):
        assert (two_encounters & center_is_cone[:, encounter]).float().mean() > 0.1
    assert (two_encounters & center_is_cone.all(-1)).sum() > 100
    robot_pose[:, 0] = cfg.commands.base_velocity.destination_distance
    assert obstacle_clearances(env).amin() > 0.1  # The requested endpoint is free.
    robot_pose[:, 0] = 0.0

    # Corrected courses leave room for the body at the endpoint and at both exits.
    for layout in ("thin_low", "blocked"):
        reset_params = dict(cfg.events.obstacle_field.params)
        reset_params.update(layout=layout, layout_weights=None)
        reset_obstacle_field(env, None, **reset_params)
        assert obstacle_clearances(env).amin() > 0.1
        robot_pose[:, 0] = cfg.commands.base_velocity.destination_distance
        assert obstacle_clearances(env).amin() > 0.2
        robot_pose[:, 0] = 0.0
        if layout == "blocked":
            front = env.microduck_front_obstacle_index
            centers_y = torch.stack([scene[name].data.root_pos_w.torch[:, 1] for name in OBSTACLE_CELLS], -1)
            front_y = centers_y.gather(1, front[:, None]).squeeze(1)
            front_half_width = half_sizes[front, 1]
            for wall in ("wall_left", "wall_right"):
                gap = (scene[wall].data.root_pos_w.torch[:, 1] - front_y).abs() - 0.06 - front_half_width
                assert gap.amin() > 0.40

    # Occupied passages keep a wide route on BOTH sides of every shuffled prop.
    params = dict(cfg.events.obstacle_field.params)
    params.update(layout="passage", layout_weights=None)
    reset_obstacle_field(env, None, **params)
    left = scene["wall_left"].data.root_pos_w.torch
    right = scene["wall_right"].data.root_pos_w.torch
    wall_midpoint = (left + right) * 0.5
    wall_gap = (left[:, :2] - right[:, :2]).norm(dim=-1) - 0.12
    occupied = wall_gap > 1.9
    assert 0.45 < occupied.float().mean() < 0.55
    assert env.microduck_front_obstacle_index[occupied].unique().numel() == 6
    positions = torch.stack([scene[name].data.root_pos_w.torch for name in OBSTACLE_CELLS], dim=1)
    front_index = env.microduck_front_obstacle_index
    prop_position = positions[torch.arange(count), front_index]
    wall_rotation = scene["wall_left"].data.root_quat_w.torch
    # A wall's local X points across the corridor; local Y runs along it.
    local_prop = quat_apply_inverse(wall_rotation, prop_position - wall_midpoint)
    assert local_prop[occupied, 1].abs().max() < 1e-5
    prop_rotations = torch.stack([scene[name].data.root_quat_w.torch for name in OBSTACLE_CELLS], dim=1)
    prop_rotation = prop_rotations[torch.arange(count), front_index]
    x_axis = torch.zeros(count, 3)
    x_axis[:, 0] = 1.0
    prop_axis = quat_apply_inverse(wall_rotation, quat_apply(prop_rotation, x_axis))
    prop_half_sizes = half_sizes[front_index]
    lateral_extent = prop_axis[:, 0].abs() * prop_half_sizes[:, 0] + prop_axis[:, 1].abs() * prop_half_sizes[:, 1]
    smaller_side_gap = wall_gap * 0.5 - local_prop[:, 0].abs() - lateral_extent
    assert smaller_side_gap[occupied].min() > 0.4  # 24 cm body plus maneuvering space.
    assert obstacle_clearances(env).amin() > 0.1

    # Native Hydra transport can opt in, then explicitly clear it for benchmarks.
    for value, expected in (("null", None), (str(weights), weights)):
        cfg = parse_env_cfg(
            TASK,
            device="cpu",
            overrides=(
                "env.events.obstacle_field.params.layout=mixed",
                f"env.events.obstacle_field.params.layout_weights={value}",
            ),
        )
        assert cfg.events.obstacle_field.params["layout_weights"] == expected
        restored = cfg.events.obstacle_field.copy()
        restored.from_dict(cfg.events.obstacle_field.to_dict())
        assert restored.params["layout_weights"] == expected
    benchmark = parse_env_cfg(
        TASK,
        device="cpu",
        overrides=(
            "env.events.obstacle_field.params.layout=blocked",
            "env.events.obstacle_field.params.layout_weights=null",
        ),
    )
    assert benchmark.events.obstacle_field.params["layout_weights"] is None


@pytest.mark.integration
@pytest.mark.kitless
@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_microduck_simready_props_contacts_and_tof(device, walking_policy_path, record_property):
    """Every public prop is visible to the real ToF and reaches native contact termination."""
    task = TASK
    names = ("obstacle", "crate", "tall_box", "low_block", "cylinder", "thin_post")
    count = len(names) + 1  # Last environment is an open-floor control.
    cfg = parse_env_cfg(task, device=device, num_envs=count, overrides=("presets=backlash",))
    cfg.seed = 42
    cfg.actions.joint_pos.policy_path = walking_policy_path
    cfg.events.reset_base.params["pose_range"] = {}
    cfg.events.push_robot = None
    cfg.scene.head_tof.randomization.enabled = False
    sim_utils.create_new_stage()
    env = gym.make(task, cfg=cfg).unwrapped
    try:
        with torch.inference_mode():
            obs, _ = env.reset()
            assert obs["policy"].shape == (count, 53)
            assert obs["tof"].shape == (count, 1290)
            poses = {}
            for index, name in enumerate(OBSTACLE_CELLS):
                asset = env.scene[name]
                pose = asset.data.root_pose_w.torch.clone()
                pose[:, :3] = env.scene.env_origins + torch.tensor([-4.0, index * 1.0, 0.0], device=env.device)
                pose[:, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=env.device)
                if name in names:
                    row = names.index(name)
                    # Place the near face at 0.5 m, without scaling any mesh.
                    pose[row, :3] = env.scene.env_origins[row]
                    footprint = cfg.obstacle_footprints[name]
                    half_x = footprint.size[0] / 2 if footprint.size is not None else footprint.radius
                    pose[row, 0] += 0.5 + half_x
                pose[:, 2] = env.scene.env_origins[:, 2] + asset.data.default_root_state.torch[:, 2]
                asset.write_root_pose_to_sim_index(root_pose=pose)
                poses[name] = pose
            env.sim.forward()
            env.scene.update(0.0)
            sensor = env.scene["head_tof"]
            sensor.reset()
            assert torch.isfinite(sensor.data.pos_w.torch).all()  # Acquire the reset frame.
            hits = sensor.ray_hits_w.torch - env.scene.env_origins[:, None, :]
            obstacle_hits = (
                torch.isfinite(hits).all(-1) & (hits[..., 0] > 0.4) & (hits[..., 0] < 1.5) & (hits[..., 2] > 0.03)
            )
            hit_counts = obstacle_hits.sum(-1)
            record_property("prop_tof_hits", json.dumps(dict(zip((*names, "open_control"), hit_counts.tolist()))))
            assert (hit_counts[:-1] > 0).all(), hit_counts
            assert hit_counts[-1] == 0, hit_counts
            # Move each prop surface into the robot; native contact/failure must see it.
            for row, name in enumerate(names):
                pose = poses[name].clone()
                pose[row, 0] -= 0.48
                env.scene[name].write_root_pose_to_sim_index(root_pose=pose)
            env.sim.forward()
            env.scene.update(0.0)
            observed = torch.zeros(count, dtype=torch.bool, device=env.device)
            active = torch.ones_like(observed)
            for _ in range(4):
                obs, rewards, terminated, truncated, _ = env.step(torch.zeros(count, 3, device=env.device))
                observed |= active & env.termination_manager.get_term("obstacle_collision")
                active &= ~(terminated | truncated)
                assert torch.isfinite(rewards).all()
                assert all(torch.isfinite(value).all() for value in obs.values())
            record_property(
                "prop_collision_detected", json.dumps(dict(zip((*names, "open_control"), observed.tolist())))
            )
            assert observed[:-1].all(), observed
            assert not observed[-1], observed
    finally:
        env.close()
        SimulationContext.clear_instance()
