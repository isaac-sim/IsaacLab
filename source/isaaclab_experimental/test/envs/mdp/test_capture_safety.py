# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CUDA-graph capture safety for warp MDP terms.

The manager call switch runs every manager graph-captured by default
(``ManagerCallSwitch.DEFAULT_CONFIG``), so a warp MDP term that is not capture-safe is not an
edge case — it is the normal execution path. A captured graph records launch topology and
pointer values and replays them, so a term must launch a fixed dim, touch only buffers that
were allocated before capture and never move, and do all of its work in kernels: host-side
Python between launches is simply not replayed.

Capture-and-replay alone cannot prove that. Replaying against unchanged memory re-reads the
same still-valid bytes, so a term that baked a stale pointer or a stale scalar still looks
correct. Every term here is therefore captured, then has its input buffers overwritten
**in place**, then replayed, and the result is compared against the stable implementation on
the *new* data.

Terms are discovered from the warp MDP modules rather than listed, so
:func:`test_every_warp_mdp_term_is_declared` fails as soon as a new term is added without a
:data:`CAPTURE_SPECS` entry.
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import pkgutil
from collections.abc import Callable
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import warp as wp

wp.init()
pytestmark = pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA device required")

import isaaclab_experimental.envs.mdp.rewards as warp_rew
import isaaclab_experimental.envs.mdp.terminations as warp_term
import isaaclab_tasks_experimental.core as tasks_experimental_core
import isaaclab_tasks_experimental.core.locomotion.mdp.rewards as warp_loco_rew
from parity_helpers import (
    DEVICE,
    NUM_BODIES,
    NUM_ENVS,
    MockArticulation,
    MockArticulationData,
    MockPoseCommandManager,
    MockScene,
    MockTerminationManager,
    assert_close,
    assert_equal,
    make_pose_command_term,
    mutate_body_data,
)

import isaaclab.envs.mdp.rewards as stable_rew
import isaaclab.envs.mdp.terminations as stable_term
from isaaclab.managers.manager_term_cfg import RewardTermCfg, TerminationTermCfg

import isaaclab_tasks.core.locomotion.mdp.rewards as stable_loco_rew


@dataclasses.dataclass(frozen=True)
class CaptureCase:
    """One term, wired up and ready to be captured.

    Attributes:
        warp_fn: The warp term, already instantiated when it is a class term.
        stable_fn: The stable counterpart, called as ``stable_fn(stable_env, **params)``.
        warp_env: Environment passed to the warp term.
        stable_env: Environment passed to the stable term; may be the same object.
        params: Term parameters forwarded to both sides.
        mutate: Overwrites every input buffer in place, so replay sees new values.
        after_replay: Asserts the term's side effects landed during replay, given the replayed
            output. A term whose stateful write moved to host code would pass the output
            comparison and fail here.
    """

    warp_fn: Callable
    stable_fn: Callable
    warp_env: object
    stable_env: object
    params: dict
    mutate: Callable[[], None]
    after_replay: Callable[[torch.Tensor], None] | None = None


@dataclasses.dataclass(frozen=True)
class CaptureSpec:
    """Declares how to exercise one warp MDP term under capture.

    Attributes:
        name: The term's ``__name__``, matched against the discovered modules.
        modality: ``"reward"`` (float32 output) or ``"termination"`` (bool output).
        build: Builds a fresh :class:`CaptureCase`; called once per test.
    """

    module: str
    name: str
    modality: str
    build: Callable[[], CaptureCase]
    expect_nonzero: bool = True
    """Whether the mutated inputs should produce a non-zero expectation.

    Guards against a comparison that holds trivially. Set ``False`` only for terms whose
    output is a constant by design, such as a pure-metric term that writes zeros.
    """

    @property
    def qualified(self) -> str:
        """The ``"<module>:<name>"`` identity this spec declares."""
        return f"{self.module}:{self.name}"


# ---------------------------------------------------------------------------
# Case builders
# ---------------------------------------------------------------------------


def _build_is_terminated_term() -> CaptureCase:
    """A reward term reading the termination manager's per-term done columns."""
    manager = MockTerminationManager()
    manager.term_dones[::3] = True
    manager.time_outs[:] = manager.term_dones[:, -1]
    env = SimpleNamespace(termination_manager=manager, num_envs=NUM_ENVS, device=DEVICE)
    params = {"term_keys": ["success", "fell_over"]}

    def mutate() -> None:
        rng = np.random.RandomState(404)
        fresh = torch.tensor(rng.rand(NUM_ENVS, len(manager.active_terms)) < 0.6, device=DEVICE)
        manager.term_dones[:] = fresh
        manager.time_outs[:] = fresh[:, -1]
        wp.synchronize()

    return CaptureCase(
        warp_fn=warp_rew.is_terminated_term(
            RewardTermCfg(func=warp_rew.is_terminated_term, weight=1.0, params=params), env
        ),
        stable_fn=stable_rew.is_terminated_term(
            RewardTermCfg(func=stable_rew.is_terminated_term, weight=1.0, params=params), env
        ),
        warp_env=env,
        stable_env=env,
        params=params,
        mutate=mutate,
    )


def _build_pose_command_success() -> CaptureCase:
    """A termination term reading articulation buffers plus a cached command view."""
    art_data = MockArticulationData(num_bodies=NUM_BODIES)
    scene = MockScene(
        {"robot": MockArticulation(art_data, num_bodies=NUM_BODIES)},
        wp.array(np.zeros((NUM_ENVS, 3), dtype=np.float32), dtype=wp.vec3f, device=DEVICE),
    )
    command = make_pose_command_term(scene["robot"])
    env = SimpleNamespace(
        scene=scene,
        command_manager=MockPoseCommandManager(command),
        num_envs=NUM_ENVS,
        device=DEVICE,
    )
    params = {"command_name": "ee_pose"}

    def mutate() -> None:
        # both the articulation state and the command buffer move, so a stale view of either
        # (the term caches a warp alias of the command tensor at init) shows up on replay
        mutate_body_data(art_data)
        fresh = make_pose_command_term(scene["robot"], seed=505)
        command.pose_command_b[:] = fresh.pose_command_b
        # cleared so the replay has to rewrite it: the sticky write lives in the kernel, and a
        # host-side equivalent would simply not run on replay
        command._succeeded.zero_()
        wp.synchronize()

    def after_replay(out_torch) -> None:
        assert_equal(command._succeeded, out_torch)

    return CaptureCase(
        after_replay=after_replay,
        warp_fn=warp_term.pose_command_success(
            TerminationTermCfg(func=warp_term.pose_command_success, params=params), env
        ),
        stable_fn=stable_term.pose_command_success,
        warp_env=env,
        stable_env=env,
        params=params,
        mutate=mutate,
    )


def _build_terminated_penalty() -> CaptureCase:
    """A reward term reading the termination manager's aggregate terminated flag."""
    manager = MockTerminationManager()
    manager.terminated[::4] = True
    env = SimpleNamespace(termination_manager=manager, num_envs=NUM_ENVS, device=DEVICE, step_dt=0.02)

    def mutate() -> None:
        rng = np.random.RandomState(606)
        manager.terminated[:] = torch.tensor(rng.rand(NUM_ENVS) < 0.5, device=DEVICE)
        wp.synchronize()

    return CaptureCase(
        warp_fn=warp_loco_rew.terminated_penalty,
        stable_fn=stable_loco_rew.terminated_penalty,
        warp_env=env,
        stable_env=env,
        params={},
        mutate=mutate,
    )


def _build_survival_success_rate() -> CaptureCase:
    """A pure-metric reward term: the output is zeros, the work happens on reset."""
    manager = MockTerminationManager()
    env = SimpleNamespace(termination_manager=manager, num_envs=NUM_ENVS, device=DEVICE, step_dt=0.02)
    cfg = RewardTermCfg(func=warp_loco_rew.survival_success_rate, weight=0.0, params={})

    def mutate() -> None:
        manager.time_outs[:] = True
        wp.synchronize()

    return CaptureCase(
        warp_fn=warp_loco_rew.survival_success_rate(cfg, env),
        stable_fn=stable_loco_rew.survival_success_rate(
            RewardTermCfg(func=stable_loco_rew.survival_success_rate, weight=0.0, params={}), env
        ),
        warp_env=env,
        stable_env=env,
        params={},
        mutate=mutate,
    )


_SHARED_REW = "isaaclab_experimental.envs.mdp.rewards"
_SHARED_TERM = "isaaclab_experimental.envs.mdp.terminations"
_LOCO_REW = "isaaclab_tasks_experimental.core.locomotion.mdp.rewards"

CAPTURE_SPECS: list[CaptureSpec] = [
    CaptureSpec(_SHARED_REW, "is_terminated_term", "reward", _build_is_terminated_term),
    CaptureSpec(_SHARED_TERM, "pose_command_success", "termination", _build_pose_command_success),
    CaptureSpec(_LOCO_REW, "terminated_penalty", "reward", _build_terminated_penalty),
    CaptureSpec(_LOCO_REW, "survival_success_rate", "reward", _build_survival_success_rate, expect_nonzero=False),
]


# Warp MDP terms not yet exercised by this harness. Pre-existing terms only: a *new* term must
# arrive with a CAPTURE_SPECS entry instead of a row here. Several are already covered by the
# hand-written ``TestCapturedDataMutation*`` classes in the parity test files; the rest are an
# explicit backlog rather than a silent gap.
def _ids(module: str, *names: str) -> list[str]:
    """Qualified ``"<module>:<name>"`` identities for several terms in one module."""
    return [f"{module}:{name}" for name in names]


_SH = "isaaclab_experimental.envs.mdp"
_TE = "isaaclab_tasks_experimental.core"
_MUT = "covered by the mutate-replay class in the matching parity test file"
_NONE = "capturable (fixed dim=num_envs launch over a mask) but no capture coverage yet"
_MIRROR = "per-task warp mirror; isaaclab_tasks_experimental has no test directory"

# Warp MDP terms not exercised by this harness. Pre-existing terms only: a *new* term must
# arrive with a CAPTURE_SPECS entry instead of a row here.
CAPTURE_UNAUDITED: dict[str, str] = {
    **dict.fromkeys(
        _ids(
            f"{_SH}.rewards",
            "action_l2",
            "action_rate_l2",
            "ang_vel_xy_l2",
            "flat_orientation_l2",
            "is_alive",
            "is_terminated",
            "joint_acc_l2",
            "joint_deviation_l1",
            "joint_pos_limits",
            "joint_torques_l2",
            "joint_vel_l1",
            "joint_vel_l2",
            "lin_vel_z_l2",
            "track_ang_vel_z_exp",
            "track_lin_vel_xy_exp",
            "undesired_contacts",
        )
        + _ids(
            f"{_SH}.terminations",
            "illegal_contact",
            "joint_pos_out_of_manual_limit",
            "root_height_below_minimum",
            "time_out",
        )
        + _ids(
            f"{_SH}.observations",
            "base_ang_vel",
            "base_lin_vel",
            "base_pos_z",
            "generated_commands",
            "joint_pos",
            "joint_pos_limit_normalized",
            "joint_vel",
            "last_action",
            "projected_gravity",
        )
        + _ids(
            f"{_SH}.events",
            "apply_external_force_torque",
            "push_by_setting_velocity",
            "reset_joints_by_offset",
            "reset_joints_by_scale",
        ),
        _MUT,
    ),
    **dict.fromkeys(
        _ids(f"{_SH}.observations", "body_incoming_wrench", "joint_pos_rel", "joint_vel_rel")
        + _ids(
            f"{_SH}.events",
            "randomize_rigid_body_com",
            "randomize_rigid_body_mass",
            "randomize_rigid_body_material",
            "reset_root_state_uniform",
        )
        + _ids(
            f"{_SH}.actions.joint_actions",
            "JointAction",
            "JointPositionAction",
            "JointEffortAction",
        ),
        _NONE,
    ),
    **dict.fromkeys(
        _ids(f"{_TE}.cartpole.mdp.rewards", "joint_pos_target_l2", "survival_success_rate")
        + _ids(
            f"{_TE}.locomotion.mdp.observations",
            "base_angle_to_target",
            "base_heading_proj",
            "base_up_proj",
            "base_yaw_roll",
        )
        + _ids(
            f"{_TE}.locomotion.mdp.rewards",
            "joint_pos_limits_penalty_ratio",
            "move_to_target_bonus",
            "power_consumption",
            "progress_reward",
            "upright_posture_bonus",
        )
        + _ids(
            f"{_TE}.reach.mdp.rewards",
            "orientation_command_error",
            "position_command_error",
            "position_command_error_tanh",
        )
        + _ids(
            f"{_TE}.velocity.mdp.rewards",
            "feet_air_time",
            "feet_air_time_positive_biped",
            "feet_slide",
            "stand_still_joint_deviation_l1",
            "track_ang_vel_z_world_exp",
            "track_lin_vel_xy_yaw_frame_exp",
        )
        + _ids(f"{_TE}.velocity.mdp.terminations", "terrain_out_of_bounds"),
        _MIRROR,
    ),
}


# Every modality a manager can run. ``actions`` is included because ``ActionManager`` is
# graph-captured like the rest, so an unsafe action term would otherwise bypass every check.
_MDP_LEAF_MODULES = ("rewards", "terminations", "observations", "events", "actions.joint_actions")


def _warp_mdp_modules() -> list:
    """The shared warp MDP modules plus every per-task warp MDP mirror."""
    modules = []
    for leaf in _MDP_LEAF_MODULES:
        modules.append(importlib.import_module(f"isaaclab_experimental.envs.mdp.{leaf}"))
    for entry in pkgutil.iter_modules(tasks_experimental_core.__path__):
        for leaf in _MDP_LEAF_MODULES:
            try:
                modules.append(importlib.import_module(f"{tasks_experimental_core.__name__}.{entry.name}.mdp.{leaf}"))
            except ImportError:
                continue  # not every task mirror defines every modality
    return modules


def _discover_warp_mdp_terms() -> set[str]:
    """Return every public warp MDP term as a ``"<module>:<name>"`` identity.

    Qualified rather than bare: the same term name legitimately appears in more than one task
    mirror (``survival_success_rate`` is defined by both cartpole and locomotion), and keying
    by name alone would let a spec for one of them mark the other as declared.
    """
    terms: set[str] = set()
    for module in _warp_mdp_modules():
        for name, obj in vars(module).items():
            if name.startswith("_") or not (inspect.isfunction(obj) or inspect.isclass(obj)):
                continue
            if getattr(obj, "__module__", "") == module.__name__:
                terms.add(f"{module.__name__}:{name}")
    return terms


def test_every_warp_mdp_term_is_declared():
    """Every warp MDP term is either exercised here or listed as an unaudited pre-existing term."""
    declared = {spec.qualified for spec in CAPTURE_SPECS} | set(CAPTURE_UNAUDITED)
    undeclared = sorted(_discover_warp_mdp_terms() - declared)

    assert not undeclared, (
        "warp MDP terms with no capture declaration: "
        + ", ".join(undeclared)
        + ". Add a CAPTURE_SPECS entry; CAPTURE_UNAUDITED is for pre-existing terms only."
    )


def test_no_term_is_both_specified_and_unaudited():
    """A term moved into the harness must lose its unaudited row, so the backlog stays truthful."""
    overlap = sorted({spec.qualified for spec in CAPTURE_SPECS} & set(CAPTURE_UNAUDITED))

    assert not overlap, f"remove from CAPTURE_UNAUDITED, now exercised here: {overlap}"


def test_unaudited_terms_still_exist():
    """Drop rows for terms that no longer exist, so the list cannot rot."""
    stale = sorted(set(CAPTURE_UNAUDITED) - _discover_warp_mdp_terms())

    assert not stale, f"CAPTURE_UNAUDITED lists terms that no longer exist: {stale}"


@pytest.mark.parametrize("spec", CAPTURE_SPECS, ids=lambda spec: spec.name)
def test_term_is_capture_safe(spec: CaptureSpec):
    """Capture the term, overwrite its inputs in place, replay, and match stable on the new data."""
    case = spec.build()
    dtype = wp.float32 if spec.modality == "reward" else wp.bool
    out = wp.zeros((NUM_ENVS,), dtype=dtype, device=DEVICE)

    case.warp_fn(case.warp_env, out, **case.params)  # warm-up outside the capture
    with wp.ScopedCapture() as capture:
        case.warp_fn(case.warp_env, out, **case.params)

    case.mutate()
    wp.capture_launch(capture.graph)
    actual = wp.to_torch(out).clone()

    # Side effects are checked before the stable reference runs. The stable terms write the same
    # shared state they are being compared against — ``compute_success`` ORs into the command's
    # sticky buffer — so calling the reference first would repair exactly what this asserts.
    if case.after_replay is not None:
        case.after_replay(actual)

    expected = case.stable_fn(case.stable_env, **case.params)
    if spec.modality == "reward":
        assert_close(actual, expected)
    else:
        assert_equal(actual, expected)
    if spec.expect_nonzero:
        assert expected.any(), "mutated inputs produced a degenerate expectation; the replay proves little"


def test_survival_success_rate_reset_is_capture_safe():
    """The metric is produced by ``reset()``, not ``__call__``, so capture it separately.

    ``__call__`` writes constant zeros; capturing only that would leave the counting kernels
    and the persistent metric view unexercised, and a host-side ``.item()`` readback — which is
    what the stable term does and what cannot survive capture — would go unnoticed.
    """
    manager = MockTerminationManager()
    env = SimpleNamespace(termination_manager=manager, num_envs=NUM_ENVS, device=DEVICE, step_dt=0.02)
    term = warp_loco_rew.survival_success_rate(
        RewardTermCfg(func=warp_loco_rew.survival_success_rate, weight=0.0, params={}), env
    )
    env_mask = wp.array(np.ones(NUM_ENVS, dtype=bool), dtype=wp.bool, device=DEVICE)
    metric = term.reset(env_mask)["Metrics/success_rate"]

    manager.time_outs[:] = False
    manager.time_outs[: NUM_ENVS // 4] = True
    term.reset(env_mask)  # warm-up outside the capture
    with wp.ScopedCapture() as capture:
        term.reset(env_mask)

    # a different survival fraction must show up through the same persistent view on replay
    manager.time_outs[:] = False
    manager.time_outs[: NUM_ENVS // 2] = True
    wp.synchronize()
    wp.capture_launch(capture.graph)
    wp.synchronize()

    assert metric.item() == pytest.approx(0.5), "replayed reset did not recompute the metric on device"


def test_survival_success_rate_reports_nan_for_an_empty_reset():
    """An empty selection must not publish a real 0% sample; stable's mean of nothing is NaN."""
    manager = MockTerminationManager()
    env = SimpleNamespace(termination_manager=manager, num_envs=NUM_ENVS, device=DEVICE, step_dt=0.02)
    term = warp_loco_rew.survival_success_rate(
        RewardTermCfg(func=warp_loco_rew.survival_success_rate, weight=0.0, params={}), env
    )
    empty = wp.array(np.zeros(NUM_ENVS, dtype=bool), dtype=wp.bool, device=DEVICE)

    metric = term.reset(empty)["Metrics/success_rate"]
    wp.synchronize()

    expected = torch.tensor([], device=DEVICE).mean()
    assert torch.isnan(metric) == torch.isnan(expected), "empty reset must match the stable empty-mean result"
