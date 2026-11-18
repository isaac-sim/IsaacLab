# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gaze geometry and gaze-supervised PPO, without a simulator."""

import math
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
import warp as wp
from fovi.sensing.projection import CameraModel
from isaaclab_fovi.mdp import angles_to_direction
from isaaclab_fovi.mdp.gaze import (
    GazeAction,
    gaze_reset_kernel,
    gaze_state,
    gaze_step_kernel,
    teacher_block_dim,
    teacher_labels_kernel,
)
from isaaclab_fovi.mdp.gaze_cfg import GazeActionCfg
from isaaclab_fovi.rsl_rl.gaze_distribution import GazeGaussianDistribution
from isaaclab_fovi.rsl_rl.gaze_distribution_cfg import GazeGaussianDistributionCfg
from isaaclab_fovi.rsl_rl.gaze_supervised_ppo import GazeSupervisedPPO, gaze_loss
from isaaclab_fovi.sensors.gaze_kernels import camera_params, torch_stream
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

from isaaclab.test.utils import DeviceScope, test_devices
from isaaclab.utils.math import quat_apply, quat_apply_inverse
from isaaclab.utils.warp import ProxyArray

pytestmark = pytest.mark.kitless

NUM_ENVS = 64
STEPS = 8
DEVICES = test_devices(DeviceScope.CPU_AND_DEFAULT_CUDA)


def direction_to_angles(direction: torch.Tensor) -> torch.Tensor:
    """Gaze angles [deg] of camera-frame directions, shape (..., 3) to (..., 2): the reference for the kernels."""
    x, y, z = direction.unbind(-1)
    return torch.stack((-torch.atan2(y, torch.sqrt(x.square() + z.square())), -torch.atan2(x, z)), -1).rad2deg()


def clamp_gaze(gaze: torch.Tensor, limit: float) -> torch.Tensor:
    """Gazes scaled back onto the circle of radius ``limit`` when they leave it: the reference for the step kernel."""
    return gaze * (limit / gaze.norm(dim=-1, keepdim=True).clamp_min(limit))


def test_gaze_angles_round_trip_and_follow_the_camera_convention():
    angles = torch.tensor([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [-20.0, 35.0]])
    directions = angles_to_direction(angles)
    torch.testing.assert_close(direction_to_angles(directions), angles, atol=1e-4, rtol=0)
    # OpenCV camera frame: z forward, y down, x right; positive tilt looks up, positive pan looks left.
    torch.testing.assert_close(directions[0], torch.tensor([0.0, 0.0, 1.0]))
    assert directions[1, 1] < 0 and directions[2, 0] < 0


def test_gaze_step_kernel_matches_the_torch_update():
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


def mount_frame(rotation_deg: float) -> torch.Tensor:
    """The camera's ``_mask_frame`` for an image rolled by ``rotation_deg``: upright gaze frame to physical frame."""
    angle = torch.tensor(rotation_deg).deg2rad()
    c, s = float(angle.cos()), float(angle.sin())
    return torch.tensor([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("rotation_deg", [0.0, 35.0])
def test_teacher_labels_kernel_matches_the_torch_labels(device, rotation_deg):
    camera = CameraModel("pinhole", (480, 640), (400.0, 400.0, 319.5, 239.5), distortion=(-0.1, 0.02, 0.0, 0.0))
    field_radius, step, limit = 30.0, 0.1, 1.0
    frame = mount_frame(rotation_deg)
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
    gaze = (torch.rand(count, 2, generator=generator) * 2 - 1) * 0.7

    # The torch path the kernel replaces: entity_point, then the teacher, then the labels and logs.
    points = targets + quat_apply(orientations, offset.expand(count, -1))
    direction = torch.nn.functional.normalize(quat_apply_inverse(mount, points - positions), dim=-1)
    # Gazes are measured in the upright image frame, which the frame carries into the physical camera's.
    upright = direction @ frame
    desired = direction_to_angles(upright) / field_radius
    valid = camera.project(direction)[1] & (desired.norm(dim=-1) <= limit)
    action = torch.where(valid[:, None], ((desired - gaze) / step).clamp(-1, 1), 0.0)
    looking = angles_to_direction(gaze * field_radius)
    error = torch.atan2(torch.cross(looking, upright, dim=-1).norm(dim=-1), (looking * upright).sum(-1)).rad2deg()
    expected = torch.cat((action, valid[:, None].float()), dim=-1)

    labels = torch.empty(count, 3, device=device)
    log = torch.full((2,), float("nan"), device=device)
    block = teacher_block_dim(device)
    with torch_stream(device):
        wp.launch(
            teacher_labels_kernel,
            dim=(1, block),
            inputs=[
                wp.from_torch(targets.to(device), dtype=wp.vec3f),
                wp.from_torch(orientations.to(device), dtype=wp.quatf),
                wp.vec3f(*offset.tolist()),
                1,
                wp.from_torch(positions.to(device), dtype=wp.vec3f),
                wp.from_torch(mount.to(device), dtype=wp.quatf),
                wp.from_torch((gaze * field_radius).to(device), dtype=wp.vec2f),
                wp.mat33f(*frame.flatten().tolist()),
                field_radius,
                step,
                limit,
                camera_params(camera),
            ],
            outputs=[wp.from_torch(labels), wp.from_torch(log)],
            block_dim=block,
        )
    labels, log = labels.cpu(), log.cpu()
    agree = (labels[:, 2] > 0.5) == valid
    assert agree.float().mean() > 0.999 and valid.any() and (~valid).any()
    torch.testing.assert_close(labels[agree], expected[agree], atol=1e-4, rtol=0)
    labelled = labels[:, 2] > 0.5
    assert float(log[1]) == pytest.approx(float(labelled.float().mean()))
    assert float(log[0]) == pytest.approx(float(error[labelled].mean()), rel=1e-4)


@pytest.mark.parametrize("rotation_deg", [0.0, -50.0, 120.0])
def test_teacher_gaze_fixates_the_target_through_a_rolled_mount(rotation_deg):
    """Following a teacher label puts the fixation, which reaches the physical camera at frame * gaze * z, on the
    target."""
    camera = CameraModel("pinhole", (480, 640), (300.0, 300.0, 319.5, 239.5))
    field_radius, step = 40.0, 10.0
    frame = mount_frame(rotation_deg)
    direction = torch.nn.functional.normalize(torch.tensor([[0.25, -0.15, 1.0], [-0.3, 0.2, 1.0]]), dim=-1)
    count = len(direction)
    identity = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(count, 1)
    labels, log = torch.empty(count, 3), torch.empty(2)
    with torch_stream("cpu"):
        wp.launch(
            teacher_labels_kernel,
            dim=(1, 1),
            inputs=[
                wp.from_torch(direction, dtype=wp.vec3f),
                wp.zeros(1, dtype=wp.quatf),
                wp.vec3f(0.0, 0.0, 0.0),
                0,
                wp.zeros(count, dtype=wp.vec3f),
                wp.from_torch(identity, dtype=wp.quatf),
                wp.zeros(count, dtype=wp.vec2f),
                wp.mat33f(*frame.flatten().tolist()),
                field_radius,
                step,
                1.0,
                camera_params(camera),
            ],
            outputs=[wp.from_torch(labels), wp.from_torch(log)],
            block_dim=1,
        )
    assert (labels[:, 2] == 1).all()
    gaze = labels[:, :2] * step
    fixation = angles_to_direction(gaze * field_radius) @ frame.T
    torch.testing.assert_close(fixation, direction, atol=1e-5, rtol=0)
    # The error is the gaze's angle from the target before the step: the gaze starts on the optical axis.
    torch.testing.assert_close(log[0], direction[:, 2].acos().rad2deg().mean(), atol=1e-3, rtol=0)


@pytest.mark.parametrize(
    "field, value", [("step_size", 0.0), ("step_size", float("nan")), ("gaze_limit", -1.0), ("gaze_limit", math.inf)]
)
def test_gaze_config_rejects_bounds_that_divide_by_zero(field, value):
    cfg = GazeActionCfg(asset_name="camera")
    cfg.validate_config()
    setattr(cfg, field, value)
    with pytest.raises(ValueError, match=field):
        cfg.validate_config()


def test_gaze_config_rejects_an_unknown_mount():
    GazeActionCfg(asset_name="camera", mount="tracked").validate_config()
    with pytest.raises(ValueError, match="mount"):
        GazeActionCfg(asset_name="camera", mount="head").validate_config()


def test_gaze_config_resolves_its_term_lazily():
    script = (
        "import sys\n"
        "from isaaclab_fovi.mdp.gaze_cfg import GazeActionCfg\n"
        "class_type = GazeActionCfg(asset_name='camera').class_type\n"
        "assert isinstance(class_type, str), type(class_type)\n"
        "assert 'isaaclab_fovi.mdp.gaze' not in sys.modules, 'the config imported its runtime term'\n"
        "term = class_type._resolve()\n"
        "assert 'isaaclab_fovi.mdp.gaze' in sys.modules\n"
        "print(term.__module__, term.__qualname__)\n"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=120, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.split()[-2:] == ["isaaclab_fovi.mdp.gaze", "GazeAction"]
    assert GazeActionCfg(asset_name="camera").class_type._resolve() is GazeAction


def test_process_actions_moves_the_gaze_by_the_supplied_actions():
    """The term reads whatever actions it is handed: the manager's slice in place, or another tensor."""
    num_envs = 4
    angles = ProxyArray(wp.zeros(num_envs, dtype=wp.vec2f, device="cpu"))
    set_angles = []
    camera = SimpleNamespace(gaze_angles=angles, set_gaze_angles=lambda value: set_angles.append(value))
    term = object.__new__(GazeAction)
    term._env = SimpleNamespace(num_envs=num_envs, device="cpu", step_dt=0.5)
    term.cfg = GazeActionCfg(asset_name="camera", step_size=0.1, gaze_limit=1.0)
    term.camera = camera
    term.field_radius_deg = 20.0
    term._state = ProxyArray(wp.zeros(num_envs, dtype=wp.vec4f, device="cpu"))
    term._velocity = term._state.torch[:, 2:]
    term._step_launch = term._raw = None

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


@pytest.mark.parametrize("ids", [None, [3, 0, 7]])
@pytest.mark.parametrize("random_start", [False, True])
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
    """A mount turned before the term captured it is the captured mount; a later turn makes it stale."""
    term = object.__new__(GazeAction)
    term.tracked_mount = False
    term._mount_turns = 2
    term._fixed_mount = "captured"
    term.camera = SimpleNamespace(mount_turns=2)
    assert GazeAction.teacher_mounts(term) == "captured"
    term.camera.mount_turns = 3
    with pytest.raises(RuntimeError, match="mount='tracked'"):
        GazeAction.teacher_mounts(term)


def test_gaze_loss_ignores_invalid_labels():
    mean = torch.tensor([[0.0, 0.0], [1.0, 1.0]], requires_grad=True)
    labels = torch.tensor([[0.3, 0.4, 1.0], [float("nan"), float("nan"), 0.0]])
    loss = gaze_loss(mean, labels)
    loss.backward()
    assert float(loss.detach()) == pytest.approx(0.5)
    assert torch.isfinite(mean.grad).all() and not mean.grad[1].any()


def _algorithm(
    oracle_loss_coef: float = 1.0, with_labels: bool = True, stock: bool = False, gaze_head: bool = False
) -> tuple[PPO | GazeSupervisedPPO, TensorDict]:
    torch.manual_seed(0)
    labels = torch.rand(NUM_ENVS, 3) * 2 - 1
    labels[:, 2] = (torch.arange(NUM_ENVS) % 4 != 0).float()
    labels[labels[:, 2] == 0, :2] = float("nan")
    groups = {"policy": torch.randn(NUM_ENVS, 6)} | ({"oracle": labels} if with_labels else {})
    obs = TensorDict(groups, batch_size=[NUM_ENVS])
    model_groups = {"actor": ["policy"], "critic": ["policy"]}
    distribution = {"class_name": "GaussianDistribution", "init_std": 1.0}
    if gaze_head:
        head = GazeGaussianDistributionCfg(init_std=1.0, gaze_std=0.1)
        distribution = {"class_name": head.class_name, "init_std": 1.0, "gaze_std": head.gaze_std}
    actor = MLPModel(obs, model_groups, "actor", 3, hidden_dims=[32], distribution_cfg=distribution)
    critic = MLPModel(obs, model_groups, "critic", 1, hidden_dims=[32])
    storage = RolloutStorage("rl", NUM_ENVS, STEPS, obs, [3], "cpu")
    settings = dict(num_learning_epochs=4, num_mini_batches=4, learning_rate=3e-3, schedule="fixed", device="cpu")
    if stock:
        return PPO(actor, critic, storage, **settings), obs
    return GazeSupervisedPPO(actor, critic, storage, oracle_loss_coef=oracle_loss_coef, **settings), obs


def _train(algorithm: PPO, obs: TensorDict, iterations: int) -> dict[str, float]:
    for _ in range(iterations):
        for _ in range(STEPS):
            with torch.inference_mode():
                algorithm.act(obs)
                algorithm.process_env_step(obs, torch.zeros(NUM_ENVS), torch.zeros(NUM_ENVS), {})
        algorithm.compute_returns(obs)
        losses = algorithm.update()
    return losses


def test_update_pulls_the_gaze_toward_the_teacher():
    algorithm, obs = _algorithm()
    valid = obs["oracle"][:, 2] > 0.5

    def distance() -> float:
        with torch.no_grad():
            mean = algorithm.actor(obs)[:, -2:]
        return float((mean - obs["oracle"][:, :2])[valid].norm(dim=-1).mean())

    before = distance()
    losses = _train(algorithm, obs, 40)
    assert losses["oracle_valid"] == pytest.approx(0.75)
    assert all(torch.isfinite(p).all() for p in algorithm.actor.parameters())
    assert distance() < 0.7 * before


def test_zero_coefficient_update_matches_stock_ppo_without_labels():
    def updated(stock: bool) -> tuple[list[torch.Tensor], list[torch.Tensor], dict[str, float]]:
        algorithm, obs = _algorithm(oracle_loss_coef=0.0, with_labels=False, stock=stock)
        modules = (algorithm.actor, algorithm.critic)
        initial = [p.detach().clone() for module in modules for p in module.parameters()]
        torch.manual_seed(1)
        losses = _train(algorithm, obs, 1)
        return initial, [p.detach().clone() for module in modules for p in module.parameters()], losses

    initial, gaze_params, losses = updated(stock=False)
    stock_initial, stock_params, stock_losses = updated(stock=True)
    assert losses["oracle"] == 0.0 and losses["oracle_valid"] == 0.0
    for name in ("value", "surrogate", "entropy"):
        assert losses[name] == pytest.approx(stock_losses[name], rel=1e-6, abs=1e-7)
    for start, stock_start in zip(initial, stock_initial, strict=True):
        torch.testing.assert_close(start, stock_start, atol=0, rtol=0)
    for param, stock_param in zip(gaze_params, stock_params, strict=True):
        torch.testing.assert_close(param, stock_param, atol=0, rtol=0)
    assert all(not torch.equal(param, start) for param, start in zip(gaze_params, initial, strict=True))


def test_gaze_head_keeps_the_gaze_noise_fixed_while_ppo_learns_the_rest():
    """The entropy bonus and the surrogate move the other actions' noise, never the gaze's."""
    algorithm, obs = _algorithm(gaze_head=True)
    algorithm.entropy_coef = 0.5
    head = algorithm.actor.distribution
    assert isinstance(head, GazeGaussianDistribution)
    assert head.std_param.shape == (1,)
    _train(algorithm, obs, iterations=3)
    with torch.no_grad():
        head.update(torch.zeros(1, 3))
    std = head.std[0]
    torch.testing.assert_close(std[1:], torch.full((2,), 0.1))
    assert abs(float(std[0]) - 1.0) > 1e-3
    # The gaze noise is part of the saved policy, so a checkpoint restores it.
    assert torch.equal(head.state_dict()["gaze_std"], torch.full((2,), 0.1))


@pytest.mark.parametrize("std_type", ["scalar", "log"])
def test_gaze_head_fixes_the_last_two_actions_noise(std_type):
    head = GazeGaussianDistribution(5, init_std=2.0, std_type=std_type, gaze_std=0.05)
    assert head.input_dim == 5
    head.update(torch.zeros(3, 5))
    torch.testing.assert_close(head.std, torch.tensor([[2.0, 2.0, 2.0, 0.05, 0.05]]).expand(3, -1))


@pytest.mark.parametrize(
    "output_dim, gaze_std", [(5, 0.0), (5, float("inf")), (2, 0.1)], ids=["zero", "infinite", "gaze_only"]
)
def test_gaze_head_rejects_invalid_settings(output_dim, gaze_std):
    with pytest.raises(ValueError, match="gaze"):
        GazeGaussianDistribution(output_dim, gaze_std=gaze_std)
