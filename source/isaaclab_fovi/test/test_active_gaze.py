# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The gaze action, its observation and its teacher, without a simulator."""

import math
from types import SimpleNamespace

import pytest
import torch
import warp as wp
from fovi.sensing.projection import CameraModel
from isaaclab_fovi.mdp.gaze import (
    DUAL,
    HYBRID,
    PHYSICAL,
    GazeAction,
    gaze_reset_kernel,
    gaze_state,
    gaze_step_kernel,
    gaze_teacher_labels,
    make_rig_gaze_step,
    pending_head_turn_kernel,
)
from isaaclab_fovi.mdp.gaze_cfg import GazeActionCfg, GazeRigCfg
from isaaclab_fovi.sensors.gaze_kernels import camera_params, torch_stream

from isaaclab.managers import ObservationTermCfg, SceneEntityCfg
from isaaclab.test.utils import DeviceScope, test_devices
from isaaclab.utils.math import quat_apply, quat_apply_inverse
from isaaclab.utils.warp import ProxyArray

pytestmark = pytest.mark.kitless

DEVICES = test_devices(DeviceScope.CPU_AND_DEFAULT_CUDA)


def angles_to_direction(angles_deg: torch.Tensor) -> torch.Tensor:
    """Camera-frame unit directions of gaze angles [deg], shape (..., 2) to (..., 3): up and left are positive."""
    tilt, pan = angles_deg.deg2rad().unbind(-1)
    return torch.stack((-tilt.cos() * pan.sin(), -tilt.sin(), tilt.cos() * pan.cos()), -1)


def direction_to_angles(direction: torch.Tensor) -> torch.Tensor:
    """Gaze angles [deg] of camera-frame directions, shape (..., 3) to (..., 2): the reference for the kernels."""
    x, y, z = direction.unbind(-1)
    return torch.stack((-torch.atan2(y, torch.sqrt(x.square() + z.square())), -torch.atan2(x, z)), -1).rad2deg()


def clamp_gaze(gaze: torch.Tensor, limit: float) -> torch.Tensor:
    """Gazes scaled back onto the circle of radius ``limit`` when they leave it: the reference for the step kernel."""
    return gaze * (limit / gaze.norm(dim=-1, keepdim=True).clamp_min(limit))


@pytest.mark.parametrize(("device", "body_id"), [("cpu", None), (DEVICES[-1] if DEVICES else "cpu", 1)])
def test_teacher_labels_and_diagnostics_match_the_torch_reference(device, body_id):
    # The root case checks fixation from rest through a roll over 90 degrees. The body case exercises clamping
    # and a strided pose view, including the CUDA block reduction when a GPU is available.
    camera = CameraModel("pinhole", (480, 640), (400.0, 400.0, 319.5, 239.5), distortion=(-0.1, 0.02, 0.0, 0.0))
    field_radius, step, limit = 30.0, (2.0 if body_id is None else 0.1), 1.0
    roll = math.radians(120.0 if body_id is None else 35.0)
    c, s = math.cos(roll), math.sin(roll)
    frame = torch.tensor([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
    generator = torch.Generator().manual_seed(5)
    count = 4099
    # Each camera has its own mount, as on a moving head.
    mount = torch.nn.functional.normalize(torch.randn(count, 4, generator=generator), dim=-1)
    positions = torch.randn(count, 3, generator=generator)
    lateral = torch.randn(count, 2, generator=generator) * 0.6
    local = torch.cat((lateral, torch.rand(count, 1, generator=generator) + 0.2), -1)
    targets = positions + quat_apply(mount, local)
    orientations = torch.nn.functional.normalize(torch.randn(count, 4, generator=generator), dim=-1)
    offset = torch.tensor([0.0, 0.0, 0.3])
    gaze = torch.zeros(count, 2) if body_id is None else (torch.rand(count, 2, generator=generator) * 2 - 1) * 0.7
    # A head turn already commanded, as physical gaze queues one; the labels cover only what is left.
    pending = torch.zeros(count, 2) if body_id is None else (torch.rand(count, 2, generator=generator) * 2 - 1) * 0.05

    # The torch path the kernel replaces: the point fixed to each target, the teacher, then the labels and logs.
    points = targets + quat_apply(orientations, offset.expand(count, -1))
    direction = torch.nn.functional.normalize(quat_apply_inverse(mount, points - positions), dim=-1)

    def teachable(direction: torch.Tensor) -> torch.Tensor:
        # Gazes are measured in the upright image frame, which the frame carries into the physical camera's.
        desired = direction_to_angles(direction @ frame) / field_radius
        return camera.project(direction)[1] & (desired.norm(dim=-1) <= limit)

    upright = direction @ frame
    desired = direction_to_angles(upright) / field_radius
    valid = teachable(direction)
    # A target within 1e-4 rad of the image edge, the lens's widest angle or the gaze limit may land on either side
    # in float32; every other target must agree exactly.
    nudges = 1e-4 * torch.cat((torch.eye(3), -torch.eye(3)))
    decisive = torch.stack(
        [teachable(torch.nn.functional.normalize(direction + nudge, dim=-1)) == valid for nudge in nudges]
    ).all(0)
    delta = (desired - gaze - pending) / step
    action = torch.where(valid[:, None], delta.clamp(-1, 1), 0.0)
    looking = angles_to_direction(gaze * field_radius)
    error = torch.atan2(torch.cross(looking, upright, dim=-1).norm(dim=-1), (looking * upright).sum(-1)).rad2deg()
    expected = torch.cat((action, valid[:, None].float()), dim=-1)

    poses = torch.cat((targets, orientations), dim=-1).to(device)
    target = SceneEntityCfg("target")
    with torch_stream(device):
        if body_id is None:
            asset_data = SimpleNamespace(root_pose_w=ProxyArray(wp.from_torch(poses, dtype=wp.transformf)))
        else:
            target.body_names, target.body_ids = ["pole"], [body_id]
            bodies = torch.stack((torch.zeros_like(poses), poses), dim=1)
            asset_data = SimpleNamespace(body_pose_w=ProxyArray(wp.from_torch(bodies, dtype=wp.transformf)))
        mounts = ProxyArray(wp.from_torch(mount.to(device), dtype=wp.quatf))
        pending_turn = ProxyArray(wp.from_torch(pending.to(device), dtype=wp.vec2f))
        sensor = SimpleNamespace(
            data=SimpleNamespace(pos_w=ProxyArray(wp.from_torch(positions.to(device), dtype=wp.vec3f))),
            gaze_angles=ProxyArray(wp.from_torch((gaze * field_radius).to(device), dtype=wp.vec2f)),
            image_frame=wp.mat33f(*frame.flatten().tolist()),
            projection_params=camera_params(camera),
        )
    gaze_term = SimpleNamespace(
        camera=sensor,
        teacher_mounts=lambda: mounts,
        pending_head_turn=lambda: pending_turn,
        field_radius_deg=field_radius,
        cfg=GazeActionCfg(asset_name="camera", step_size=step),
    )
    env = SimpleNamespace(
        device=device,
        num_envs=count,
        scene={"target": SimpleNamespace(data=asset_data)},
        action_manager=SimpleNamespace(get_term=lambda name: gaze_term),
        extras={},
    )
    teacher = gaze_teacher_labels(ObservationTermCfg(func=gaze_teacher_labels), env)
    labels = teacher(env, target, offset=tuple(offset.tolist())).cpu()
    assert valid[decisive].any() and (~valid[decisive]).any() and decisive.float().mean() > 0.99
    torch.testing.assert_close(labels[decisive], expected[decisive], atol=1e-4, rtol=0)
    labelled = labels[:, 2] > 0.5
    reachable = labelled & decisive & (delta.abs().amax(dim=-1) < 1.0)
    assert reachable.any()
    fixation = angles_to_direction((gaze + pending + labels[:, :2] * step) * field_radius) @ frame.T
    torch.testing.assert_close(fixation[reachable], direction[reachable], atol=1e-5, rtol=0)
    log = env.extras["log"]
    assert float(log["Gaze/valid_target_fraction"]) == pytest.approx(float(labelled.float().mean()))
    assert float(log["Gaze/target_error_deg"]) == pytest.approx(float(error[labelled].mean()), rel=1e-4)


def test_gaze_config_rejects_bounds_that_divide_by_zero_and_unknown_mounts():
    GazeActionCfg(asset_name="camera").validate_config()
    GazeActionCfg(asset_name="camera", mount="tracked").validate_config()
    for field, value in (
        ("step_size", 0.0),
        ("step_size", float("nan")),
        ("gaze_limit", math.inf),
        ("gaze_limit", "1.0"),
        ("mount", "head"),
    ):
        cfg = GazeActionCfg(asset_name="camera")
        setattr(cfg, field, value)
        with pytest.raises(ValueError, match=field):
            cfg.validate_config()
    # The config names its term lazily, so Isaac Lab's action manager resolves it to the gaze action.
    assert GazeActionCfg(asset_name="camera").class_type._resolve() is GazeAction
    # The modes that turn a rig need one, and a tracked mount since the rig moves the camera.
    rig = GazeRigCfg(asset_name="rig")
    GazeActionCfg(asset_name="camera", mode="hybrid", rig=rig, mount="tracked").validate_config()
    for changes, message in (
        (dict(mode="saccade"), "mode"),
        (dict(mode="physical", mount="tracked"), "rig"),
        (dict(mode="hybrid", rig=rig), "tracked"),
        (dict(mode="dual", rig=rig, mount="tracked", random_reset=True), "random_reset"),
        (dict(mode="hybrid", rig=GazeRigCfg(asset_name="rig", joint_signs=(1.0, 0.5)), mount="tracked"), "signs"),
    ):
        with pytest.raises(ValueError, match=message):
            GazeActionCfg(asset_name="camera", **changes).validate_config()
    # The rig checks its own fields, alone as well as inside the action's config.
    for changes, message in (
        (dict(joint_signs=(1.0,)), "signs"),
        (dict(joint_names=("pan_joint", "pan_joint")), "joint_names"),
        (dict(max_step_deg=0.0), "max_step_deg"),
    ):
        with pytest.raises(ValueError, match=message):
            GazeRigCfg(asset_name="rig", **changes).validate_config()


def test_process_actions_moves_the_gaze_by_the_supplied_actions():
    """The term reads whatever actions it is handed: the manager's slice in place, or another tensor; and its step
    kernel matches the torch update."""
    num_envs = 4
    angles = ProxyArray(wp.zeros(num_envs, dtype=wp.vec2f, device="cpu"))
    set_angles = []
    camera = SimpleNamespace(gaze_angles=angles, set_gaze_angles=lambda value: set_angles.append(value))
    term = object.__new__(GazeAction)
    term._env = SimpleNamespace(num_envs=num_envs, device="cpu", step_dt=0.5)
    term.cfg = GazeActionCfg(asset_name="camera", step_size=0.1, gaze_limit=1.0)
    term._asset = camera
    term.field_radius_deg = 20.0
    term._state = ProxyArray(wp.zeros(num_envs, dtype=wp.vec4f, device="cpu"))
    term._velocity = term._state.torch[:, 2:]
    term._step_launch = term._raw = term.rig = None

    manager_actions = torch.zeros(num_envs, 5)
    manager_actions[:, 2:4] = torch.tensor([[1.0, 0.0], [0.0, -1.0], [0.5, 0.5], [3.0, 0.0]])
    term.process_actions(manager_actions[:, 2:4])
    expected = clamp_gaze(manager_actions[:, 2:4].clamp(-1, 1) * 0.1, 1.0)
    torch.testing.assert_close(term.gaze, expected)
    torch.testing.assert_close(term.velocity, expected / 0.5)
    torch.testing.assert_close(angles.torch, expected * 20.0)
    assert set_angles[-1] is angles.torch
    assert term.raw_actions.data_ptr() == manager_actions[:, 2:4].data_ptr()

    supplied = torch.tensor([[-1.0, 0.0]]).repeat(num_envs, 1)
    term.process_actions(supplied)
    torch.testing.assert_close(term.gaze, clamp_gaze(expected - 0.1 * supplied.abs(), 1.0))
    assert term.raw_actions is supplied
    torch.testing.assert_close(
        gaze_state(SimpleNamespace(action_manager=SimpleNamespace(get_term=lambda name: term))), term.state
    )

    # A gaze set on the camera directly is where the next step starts.
    angles.torch.copy_(torch.tensor([[10.0, 0.0]]).repeat(num_envs, 1))
    term.process_actions(torch.zeros(num_envs, 2))
    torch.testing.assert_close(term.gaze, torch.tensor([[0.5, 0.0]]).repeat(num_envs, 1))

    # The step kernel against the torch update, over many gazes.
    generator = torch.Generator().manual_seed(4)
    count, step, limit, dt, radius = 2048, 0.1, 1.0, 1 / 60, 23.6
    actions = torch.randn(count, 2, generator=generator) * 1.5
    gaze = (torch.rand(count, 2, generator=generator) * 2 - 1) * 0.75
    expected = clamp_gaze(gaze + actions.clamp(-1, 1) * step, limit)
    expected_velocity = (expected - gaze) / dt

    # The step starts from the camera's stored angles, not from the state it records.
    state = torch.randn(count, 4, generator=generator)
    angles = gaze * radius
    with torch_stream("cpu"):
        wp.launch(
            gaze_step_kernel,
            dim=count,
            inputs=[
                wp.from_torch(actions, dtype=wp.vec2f),
                wp.from_torch(state, dtype=wp.vec4f),
                wp.from_torch(angles, dtype=wp.vec2f),
                step,
                limit,
                dt,
                radius,
            ],
        )
    torch.testing.assert_close(state[:, :2], expected, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(state[:, 2:], expected_velocity, atol=2e-3, rtol=1e-5)
    torch.testing.assert_close(angles, expected * radius, atol=1e-5, rtol=1e-6)


@pytest.mark.parametrize(("ids", "random_start"), [(None, False), ([3, 0, 7], True)])
def test_gaze_reset_kernel_matches_the_torch_reset(ids, random_start):
    count, radius = 10, 23.6
    generator = torch.Generator().manual_seed(6)
    angles, state = (torch.randn(count, size, generator=generator) for size in (2, 4))
    selected = torch.arange(count) if ids is None else torch.tensor(ids)
    start = torch.randn(len(selected), 2, generator=generator) if random_start else torch.zeros(len(selected), 2)
    expected_angles, expected_state = angles.clone(), state.clone()
    # The torch reset the kernel replaces: the gaze at its start, at rest.
    expected_state[selected] = torch.cat((start, torch.zeros_like(start)), dim=-1)
    expected_angles[selected] = start * radius

    with torch_stream("cpu"):
        wp.launch(
            gaze_reset_kernel,
            dim=len(selected),
            inputs=[
                wp.from_torch(selected.to(torch.int32)) if ids is not None else wp.zeros(1, dtype=wp.int32),
                int(ids is not None),
                wp.from_torch(start, dtype=wp.vec2f) if random_start else wp.zeros(1, dtype=wp.vec2f),
                int(random_start),
                radius,
            ],
            outputs=[wp.from_torch(angles, dtype=wp.vec2f), wp.from_torch(state, dtype=wp.vec4f)],
        )
    torch.testing.assert_close(angles, expected_angles, atol=0, rtol=0)
    torch.testing.assert_close(state, expected_state, atol=0, rtol=0)


def test_fixed_mount_teacher_rejects_only_turns_after_capture():
    """The fixed-mount teacher aims from the captured mount while the camera has turned its mount as many times as
    when the term captured it, and refuses once a later turn makes the capture stale."""
    term = object.__new__(GazeAction)
    term.tracked_mount = False
    term._mount_turns = 2
    term._fixed_mount = "captured"
    term._asset = SimpleNamespace(mount_turns=2)
    assert GazeAction.teacher_mounts(term) == "captured"
    term._asset.mount_turns = 3
    with pytest.raises(RuntimeError, match="mount='tracked'"):
        GazeAction.teacher_mounts(term)


# -- Gaze modes that turn a rig ----------------------------------------------------------------------------------------

FIELD_RADIUS = 45.0
STEP, LIMIT, HEAD_STEP = 0.1, 1.0, 0.05


def yaw_left(degrees: float) -> torch.Tensor:
    """A camera mount (x, y, z, w), ROS convention, turned left by ``degrees``: about the camera's up axis, -y."""
    half = math.radians(degrees) / 2
    return torch.tensor([[0.0, -math.sin(half), 0.0, math.cos(half)]])


class RigStep:
    """One camera's rig, with pan joint 1 and tilt joint 0 of three, stepped by :func:`make_rig_gaze_step`'s kernel."""

    def __init__(self, mode: int, joint_pos=(0.1, 0.0, 0.0)):
        self.mode = mode
        self.joint_pos = torch.tensor([joint_pos])
        self.limits = torch.tensor([[[-0.5, 0.5], [-0.2, 0.2], [-1.0, 1.0]]])
        self.mount = yaw_left(0.0)
        # Only hybrid gaze holds a fixation; the other modes get empty buffers, as the term gives them.
        held = 1 if mode == HYBRID else 0
        self.held = torch.zeros(held, 3)
        self.written = torch.zeros(held, 2)
        self.held_valid = torch.zeros(held, dtype=torch.int32)
        self.angles = torch.zeros(1, 2)
        self.state = torch.zeros(1, 4)
        self.targets = torch.zeros(1, 2)

    def __call__(self, actions) -> None:
        with torch_stream("cpu"):
            wp.launch(
                make_rig_gaze_step(self.mode),
                dim=1,
                inputs=[
                    wp.from_torch(torch.tensor([actions], dtype=torch.float32)),
                    wp.from_torch(self.joint_pos),
                    wp.from_torch(self.limits, dtype=wp.vec2f),
                    wp.vec2i(1, 0),
                    wp.vec2f(1.0, -1.0),
                    wp.from_torch(self.mount, dtype=wp.quatf),
                    STEP,
                    LIMIT,
                    HEAD_STEP,
                    0.5,
                    FIELD_RADIUS,
                ],
                outputs=[
                    wp.from_torch(self.held, dtype=wp.vec3f),
                    wp.from_torch(self.written, dtype=wp.vec2f),
                    wp.from_torch(self.held_valid),
                    wp.from_torch(self.angles, dtype=wp.vec2f),
                    wp.from_torch(self.state, dtype=wp.vec4f),
                    wp.from_torch(self.targets),
                ],
            )


RADIUS_RAD = math.radians(FIELD_RADIUS)


def test_physical_gaze_steps_the_joint_targets_and_leaves_the_retina_alone():
    rig = RigStep(PHYSICAL, joint_pos=(0.0, 0.0, 0.0))
    # The head lags its last targets (pan, tilt) = (0.05, 0.1): each step adds to the targets, not the measured joints.
    rig.targets[0] = torch.tensor([0.05, 0.1])
    # Tilt up a full step, pan left half a step (pan joint 1, tilt joint 0 with sign -1).
    rig([1.0, 0.5])
    torch.testing.assert_close(rig.targets[0], torch.tensor([0.05 + 0.5 * STEP * RADIUS_RAD, 0.1 - STEP * RADIUS_RAD]))
    assert not rig.angles.any()
    # The state is the measured head's gaze: (tilt, pan) = (-tilt joint, pan joint) in field radii.
    rig.joint_pos[0, :2] = torch.tensor([0.1, 0.05])
    rig([0.0, 0.0])
    torch.testing.assert_close(rig.state[0, :2], torch.tensor([-0.1, 0.05]) / RADIUS_RAD)
    # A large step stops at the joint limits: tilt joint 0 within [-0.5, 0.5], pan joint 1 within [-0.2, 0.2].
    rig.targets[0] = torch.tensor([0.18, -0.45])
    rig([3.0, 3.0])
    torch.testing.assert_close(rig.targets[0], torch.tensor([0.2, -0.5]))


def test_hybrid_gaze_moves_the_retina_at_once_and_holds_the_fixation_while_the_head_follows():
    rig = RigStep(HYBRID, joint_pos=(0.0, 0.0, 0.0))
    rig([0.0, 1.0])  # pan left a full step
    # The retina turns at once; the head steps toward it by at most HEAD_STEP.
    torch.testing.assert_close(rig.angles[0], torch.tensor([0.0, STEP * FIELD_RADIUS]))
    torch.testing.assert_close(rig.targets[0], torch.tensor([HEAD_STEP * RADIUS_RAD, 0.0]))
    # The head turns left by 3 degrees before the next step: the retina comes back by as much, so the fixation holds.
    rig.mount = yaw_left(3.0)
    rig.joint_pos[0, 1] = math.radians(3.0)
    rig([0.0, 0.0])
    torch.testing.assert_close(rig.angles[0], torch.tensor([0.0, STEP * FIELD_RADIUS - 3.0]), atol=1e-4, rtol=0)
    torch.testing.assert_close(rig.state[0, :2], torch.tensor([0.0, STEP]), atol=1e-5, rtol=0)
    # From where it is, the head steps the remaining 1.5 degrees (under HEAD_STEP) onto the fixation.
    pan_target = math.radians(STEP * FIELD_RADIUS)
    torch.testing.assert_close(rig.targets[0, 0], torch.tensor(pan_target), atol=1e-5, rtol=0)
    # A gaze set on the camera between steps starts a new fixation instead of being aimed back at the old one.
    rig.angles[0] = torch.tensor([2.0, -1.0])
    rig([0.0, 0.0])
    torch.testing.assert_close(rig.angles[0], torch.tensor([2.0, -1.0]), atol=1e-5, rtol=0)


def test_dual_gaze_moves_the_head_and_the_retina_independently_and_composes_them():
    rig = RigStep(DUAL, joint_pos=(0.0, 0.0, 0.0))
    rig([0.0, 1.0, 1.0, 0.0])  # the head pans left, the retina tilts up
    torch.testing.assert_close(rig.targets[0], torch.tensor([STEP * RADIUS_RAD, 0.0]))
    torch.testing.assert_close(rig.angles[0], torch.tensor([STEP * FIELD_RADIUS, 0.0]))
    # With the head tilted up 20 degrees, the retina pans left 30: the observed gaze is the direction the turns
    # compose to from the rig's zero pose, not the sum of their angles.
    rig.joint_pos[0, 0] = -math.radians(20.0)  # tilt joint 0 turns the camera up with sign -1
    rig.angles[0] = torch.tensor([0.0, 30.0])
    rig([0.0, 0.0, 0.0, 0.0])
    retina = angles_to_direction(torch.tensor([0.0, 30.0]))
    c, s = math.cos(math.radians(20.0)), math.sin(math.radians(20.0))
    x, y, z = retina.tolist()
    composed = direction_to_angles(torch.tensor([x, c * y - s * z, s * y + c * z])) / FIELD_RADIUS
    torch.testing.assert_close(rig.state[0, :2], composed, atol=1e-5, rtol=0)
    assert not torch.allclose(composed, torch.tensor([20.0, 30.0]) / FIELD_RADIUS, atol=1e-3)


def test_pending_head_turn_is_what_the_joint_targets_still_hold():
    joint_pos = torch.tensor([[0.1, 0.05, 0.0]])  # tilt joint 0, pan joint 1
    targets = torch.tensor([[0.15, -0.1]])  # (pan, tilt)
    pending = torch.zeros(1, 2)
    with torch_stream("cpu"):
        wp.launch(
            pending_head_turn_kernel,
            dim=1,
            inputs=[
                wp.from_torch(joint_pos),
                wp.from_torch(targets),
                wp.vec2i(1, 0),
                wp.vec2f(1.0, -1.0),
                FIELD_RADIUS,
            ],
            outputs=[wp.from_torch(pending, dtype=wp.vec2f)],
        )
    # (tilt, pan): the tilt joint has 0.2 rad to go down, which with sign -1 is up; the pan joint 0.1 rad to the left.
    torch.testing.assert_close(pending[0], torch.tensor([0.2, 0.1]) / RADIUS_RAD)
