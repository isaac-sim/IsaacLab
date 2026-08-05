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

import isaaclab_experimental.envs.mdp.events as warp_events
import isaaclab_experimental.envs.mdp.observations as warp_obs
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
    """

    warp_fn: Callable
    stable_fn: Callable
    warp_env: object
    stable_env: object
    params: dict
    mutate: Callable[[], None]


@dataclasses.dataclass(frozen=True)
class CaptureSpec:
    """Declares how to exercise one warp MDP term under capture.

    Attributes:
        name: The term's ``__name__``, matched against the discovered modules.
        modality: ``"reward"`` (float32 output) or ``"termination"`` (bool output).
        build: Builds a fresh :class:`CaptureCase`; called once per test.
    """

    name: str
    modality: str
    build: Callable[[], CaptureCase]
    expect_nonzero: bool = True
    """Whether the mutated inputs should produce a non-zero expectation.

    Guards against a comparison that holds trivially. Set ``False`` only for terms whose
    output is a constant by design, such as a pure-metric term that writes zeros.
    """


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
        wp.synchronize()

    return CaptureCase(
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


CAPTURE_SPECS: list[CaptureSpec] = [
    CaptureSpec("is_terminated_term", "reward", _build_is_terminated_term),
    CaptureSpec("pose_command_success", "termination", _build_pose_command_success),
    CaptureSpec("terminated_penalty", "reward", _build_terminated_penalty),
    CaptureSpec("survival_success_rate", "reward", _build_survival_success_rate, expect_nonzero=False),
]

# Warp MDP terms not yet exercised by this harness. Pre-existing terms only: a *new* term must
# arrive with a CAPTURE_SPECS entry instead of a row here. Several are already covered by the
# hand-written ``TestCapturedDataMutation*`` classes in the parity test files; the rest are an
# explicit backlog rather than a silent gap.
CAPTURE_UNAUDITED: dict[str, str] = {
    "action_l2": "covered by TestCapturedDataMutationRewards",
    "action_rate_l2": "covered by TestCapturedDataMutationRewards",
    "ang_vel_xy_l2": "covered by TestCapturedDataMutationRewards",
    "flat_orientation_l2": "covered by TestCapturedDataMutationRewards",
    "is_alive": "covered by TestCapturedDataMutationRewards",
    "is_terminated": "covered by TestCapturedDataMutationRewards",
    "joint_acc_l2": "covered by TestCapturedDataMutationRewards",
    "joint_deviation_l1": "covered by TestCapturedDataMutationRewards",
    "joint_pos_limits": "covered by TestCapturedDataMutationRewards",
    "joint_torques_l2": "covered by TestCapturedDataMutationRewards",
    "joint_vel_l1": "covered by TestCapturedDataMutationRewards",
    "joint_vel_l2": "covered by TestCapturedDataMutationRewards",
    "lin_vel_z_l2": "covered by TestCapturedDataMutationRewards",
    "track_ang_vel_z_exp": "covered by TestCapturedDataMutationRewardsNewTerms",
    "track_lin_vel_xy_exp": "covered by TestCapturedDataMutationRewardsNewTerms",
    "undesired_contacts": "covered by TestCapturedDataMutationRewardsNewTerms",
    "illegal_contact": "covered by TestCapturedDataMutationTerminationsNewTerms",
    "joint_pos_out_of_manual_limit": "covered by TestCapturedDataMutationTerminations",
    "root_height_below_minimum": "covered by TestCapturedDataMutationTerminations",
    "time_out": "covered by TestCapturedDataMutationTerminationsNewTerms",
    "base_ang_vel": "covered by TestCapturedDataMutationObservations",
    "base_lin_vel": "covered by TestCapturedDataMutationObservations",
    "base_pos_z": "covered by TestCapturedDataMutationObservations",
    "generated_commands": "covered by TestCapturedDataMutationObservations",
    "joint_pos": "covered by TestCapturedDataMutationObservations",
    "joint_pos_limit_normalized": "covered by TestCapturedDataMutationObservations",
    "joint_vel": "covered by TestCapturedDataMutationObservations",
    "last_action": "covered by TestCapturedDataMutationObservations",
    "projected_gravity": "covered by TestCapturedDataMutationObservations",
    # no capture coverage anywhere — these three are the real backlog this gate exposes
    "body_incoming_wrench": "no capture coverage yet",
    "joint_pos_rel": "no capture coverage yet",
    "joint_vel_rel": "no capture coverage yet",
    # Event terms run at reset cadence over an env_ids subset, so their launch dim varies per
    # call and cannot be recorded into a graph. Capture safety does not apply to them as written.
    "apply_external_force_torque": "reset-cadence event term; variable env_ids launch dim",
    "push_by_setting_velocity": "reset-cadence event term; variable env_ids launch dim",
    "randomize_rigid_body_com": "reset-cadence event term; variable env_ids launch dim",
    "randomize_rigid_body_mass": "reset-cadence event term; variable env_ids launch dim",
    "randomize_rigid_body_material": "reset-cadence event term; variable env_ids launch dim",
    "reset_joints_by_offset": "reset-cadence event term; variable env_ids launch dim",
    "reset_joints_by_scale": "reset-cadence event term; variable env_ids launch dim",
    "reset_root_state_uniform": "reset-cadence event term; variable env_ids launch dim",
    # Per-task warp mirrors under isaaclab_tasks_experimental. That package has no test
    # directory, so none of these has parity or capture coverage anywhere in the repo.
    **{
        name: "per-task warp mirror; no test coverage in the repo yet"
        for name in (
            "base_angle_to_target",
            "base_heading_proj",
            "base_up_proj",
            "base_yaw_roll",
            "feet_air_time",
            "feet_air_time_positive_biped",
            "feet_slide",
            "joint_pos_limits_penalty_ratio",
            "joint_pos_target_l2",
            "move_to_target_bonus",
            "orientation_command_error",
            "position_command_error",
            "position_command_error_tanh",
            "power_consumption",
            "progress_reward",
            "stand_still_joint_deviation_l1",
            "terrain_out_of_bounds",
            "track_ang_vel_z_world_exp",
            "track_lin_vel_xy_yaw_frame_exp",
            "upright_posture_bonus",
        )
    },
}


def _warp_mdp_modules() -> list:
    """The shared warp MDP modules plus every per-task warp MDP mirror."""
    modules = [warp_rew, warp_term, warp_obs, warp_events]
    for entry in pkgutil.iter_modules(tasks_experimental_core.__path__):
        for leaf in ("rewards", "terminations", "observations", "events"):
            try:
                modules.append(importlib.import_module(f"{tasks_experimental_core.__name__}.{entry.name}.mdp.{leaf}"))
            except ImportError:
                continue  # not every task mirror defines every modality
    return modules


def _discover_warp_mdp_terms() -> dict[str, str]:
    """Return every public warp MDP term, mapped to its defining module name."""
    terms: dict[str, str] = {}
    for module in _warp_mdp_modules():
        for name, obj in vars(module).items():
            if name.startswith("_") or not (inspect.isfunction(obj) or inspect.isclass(obj)):
                continue
            if getattr(obj, "__module__", "") == module.__name__:
                terms[name] = module.__name__
    return terms


def test_every_warp_mdp_term_is_declared():
    """Every warp MDP term is either exercised here or listed as an unaudited pre-existing term."""
    declared = {spec.name for spec in CAPTURE_SPECS} | set(CAPTURE_UNAUDITED)
    undeclared = sorted(set(_discover_warp_mdp_terms()) - declared)

    assert not undeclared, (
        "warp MDP terms with no capture declaration: "
        + ", ".join(undeclared)
        + ". Add a CAPTURE_SPECS entry; CAPTURE_UNAUDITED is for pre-existing terms only."
    )


def test_no_term_is_both_specified_and_unaudited():
    """A term moved into the harness must lose its unaudited row, so the backlog stays truthful."""
    overlap = sorted({spec.name for spec in CAPTURE_SPECS} & set(CAPTURE_UNAUDITED))

    assert not overlap, f"remove from CAPTURE_UNAUDITED, now exercised here: {overlap}"


def test_unaudited_terms_still_exist():
    """Drop rows for terms that no longer exist, so the list cannot rot."""
    stale = sorted(set(CAPTURE_UNAUDITED) - set(_discover_warp_mdp_terms()))

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

    expected = case.stable_fn(case.stable_env, **case.params)
    actual = wp.to_torch(out).clone()
    if spec.modality == "reward":
        assert_close(actual, expected)
    else:
        assert_equal(actual, expected)
    if spec.expect_nonzero:
        assert expected.any(), "mutated inputs produced a degenerate expectation; the replay proves little"
