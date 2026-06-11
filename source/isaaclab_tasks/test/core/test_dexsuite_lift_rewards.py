# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the Dexsuite-Lift reward terms: ``delivery_progress`` and the time-since-grasp
grip decay (``good_finger_contact_decay`` / ``contact_count_decay`` / ``object_ee_distance_decay``).

These are pure-logic tests on CPU tensors. The terms are built with ``__new__`` (bypassing the
manager/scene wiring) and the environment + the ``contacts`` gate are stubbed, so the tests run
without Isaac Sim."""

import types

import pytest
import torch

from isaaclab.managers import SceneEntityCfg

from isaaclab_tasks.core.dexsuite.mdp import rewards

# --------------------------------------------------------------------------------------------------
# time-since-grasp grip decay (_GraspAgeDecay via good_finger_contact_decay)
# --------------------------------------------------------------------------------------------------


def _stub_env(num_envs: int = 4):
    return types.SimpleNamespace(
        num_envs=num_envs, device="cpu", episode_length_buf=torch.zeros(num_envs, dtype=torch.long)
    )


def _make_decay(num_envs: int = 4) -> rewards.good_finger_contact_decay:
    term = rewards.good_finger_contact_decay.__new__(rewards.good_finger_contact_decay)
    term._t_grasp = torch.full((num_envs,), -1, dtype=torch.long)
    return term


def test_decay_factor_is_one_before_first_grasp():
    env, term = _stub_env(), _make_decay()
    env.episode_length_buf = torch.full((4,), 50, dtype=torch.long)
    factor = term._decay_factor(env, torch.zeros(4, dtype=torch.bool), decay_steps=200, decay_floor=0.3)
    assert torch.allclose(factor, torch.ones(4))


def test_decay_latches_on_first_grasp_then_decays_to_floor():
    env, term = _stub_env(), _make_decay()
    grasped = torch.tensor([True, False, True, False])

    env.episode_length_buf = torch.full((4,), 10, dtype=torch.long)
    f0 = term._decay_factor(env, grasped, decay_steps=100, decay_floor=0.3)
    assert torch.allclose(f0, torch.ones(4))  # age 0 -> 1.0
    assert term._t_grasp.tolist() == [10, -1, 10, -1]  # latched only on grasped envs

    env.episode_length_buf = torch.full((4,), 60, dtype=torch.long)  # age 50 of 100
    f1 = term._decay_factor(env, grasped, decay_steps=100, decay_floor=0.3)
    assert f1[0].item() == pytest.approx(0.65, abs=1e-5)  # 1 - 0.7 * 0.5
    assert f1[1].item() == 1.0  # never grasped -> full

    env.episode_length_buf = torch.full((4,), 10_000, dtype=torch.long)
    f2 = term._decay_factor(env, grasped, decay_steps=100, decay_floor=0.3)
    assert f2[0].item() == pytest.approx(0.3, abs=1e-5)  # clamped at floor


def test_decay_does_not_reset_on_regrip():
    env, term = _stub_env(1), _make_decay(1)
    env.episode_length_buf = torch.tensor([5], dtype=torch.long)
    term._decay_factor(env, torch.tensor([True]), 100, 0.3)
    assert term._t_grasp.item() == 5
    env.episode_length_buf = torch.tensor([40], dtype=torch.long)
    term._decay_factor(env, torch.tensor([False]), 100, 0.3)  # released
    term._decay_factor(env, torch.tensor([True]), 100, 0.3)  # re-grasped
    assert term._t_grasp.item() == 5  # still latched on the FIRST grasp


def test_decay_reset_clears_grasp_time():
    term = _make_decay(2)
    term._t_grasp = torch.tensor([5, 7], dtype=torch.long)
    term.reset(torch.tensor([0]))
    assert term._t_grasp.tolist() == [-1, 7]
    term.reset(None)
    assert term._t_grasp.tolist() == [-1, -1]


def test_good_finger_contact_decay_is_contact_gated(monkeypatch):
    env, term = _stub_env(2), _make_decay(2)
    env.episode_length_buf = torch.zeros(2, dtype=torch.long)
    monkeypatch.setattr(rewards, "contacts", lambda *a, **k: torch.tensor([True, False]))
    out = term(env, threshold=0.1, thumb_name="t", finger_names=["f"])
    assert out.tolist() == [1.0, 0.0]  # grasped env full at age 0; ungrasped env 0


# --------------------------------------------------------------------------------------------------
# delivery_progress
# --------------------------------------------------------------------------------------------------


def _asset(pos: torch.Tensor, n: int):
    quat = torch.zeros(n, 4)
    quat[:, 0] = 1.0  # identity (w, x, y, z)
    data = types.SimpleNamespace(
        root_pos_w=types.SimpleNamespace(torch=pos),
        root_quat_w=types.SimpleNamespace(torch=quat),
    )
    return types.SimpleNamespace(data=data)


def _progress_env(obj_pos: torch.Tensor, goal_b: torch.Tensor, n: int = 1):
    # robot root at origin with identity rotation, so world goal == base-frame goal
    cmd = torch.zeros(n, 7)
    cmd[:, :3] = goal_b
    cmd[:, 3] = 1.0
    return types.SimpleNamespace(
        num_envs=n,
        device="cpu",
        scene={"robot": _asset(torch.zeros(n, 3), n), "object": _asset(obj_pos, n)},
        command_manager=types.SimpleNamespace(get_command=lambda name: cmd),
    )


def _make_progress(n: int = 1) -> rewards.delivery_progress:
    term = rewards.delivery_progress.__new__(rewards.delivery_progress)
    term._d0 = torch.ones(n)
    term._goal_w = torch.zeros(n, 3)
    term._need_capture = torch.ones(n, dtype=torch.bool)
    return term


_R, _O = SceneEntityCfg("robot"), SceneEntityCfg("object")


def _call(term, env):
    return term(env, "object_pose", _R, _O, "t", ["f"])


def test_delivery_progress_captures_d0_and_rises_to_one(monkeypatch):
    monkeypatch.setattr(rewards, "contacts", lambda *a, **k: torch.ones(1, dtype=torch.bool))
    term = _make_progress()
    goal = torch.tensor([[1.0, 0.0, 0.0]])

    p0 = _call(term, _progress_env(torch.zeros(1, 3), goal))  # at start
    assert p0.item() == pytest.approx(0.0, abs=1e-4)
    assert term._d0.item() == pytest.approx(1.0, abs=1e-4)

    p_half = _call(term, _progress_env(torch.tensor([[0.5, 0.0, 0.0]]), goal))
    assert p_half.item() == pytest.approx(0.5, abs=1e-4)

    p_goal = _call(term, _progress_env(goal, goal))
    assert p_goal.item() == pytest.approx(1.0, abs=1e-4)


def test_delivery_progress_zero_for_backward_motion(monkeypatch):
    monkeypatch.setattr(rewards, "contacts", lambda *a, **k: torch.ones(1, dtype=torch.bool))
    term = _make_progress()
    goal = torch.tensor([[1.0, 0.0, 0.0]])
    _call(term, _progress_env(torch.zeros(1, 3), goal))  # capture d0 = 1
    p = _call(term, _progress_env(torch.tensor([[-1.0, 0.0, 0.0]]), goal))  # farther than start
    assert p.item() == 0.0


def test_delivery_progress_is_contact_gated(monkeypatch):
    monkeypatch.setattr(rewards, "contacts", lambda *a, **k: torch.zeros(1, dtype=torch.bool))
    term = _make_progress()
    goal = torch.tensor([[1.0, 0.0, 0.0]])
    _call(term, _progress_env(torch.zeros(1, 3), goal))  # capture d0 = 1
    p = _call(term, _progress_env(torch.tensor([[0.5, 0.0, 0.0]]), goal))  # 0.5 progress, but no grasp
    assert p.item() == 0.0  # gated off despite real progress


def test_delivery_progress_refreshes_d0_on_goal_resample(monkeypatch):
    monkeypatch.setattr(rewards, "contacts", lambda *a, **k: torch.ones(1, dtype=torch.bool))
    term = _make_progress()
    _call(term, _progress_env(torch.zeros(1, 3), torch.tensor([[1.0, 0.0, 0.0]])))  # d0 = 1
    # goal jumps (resample) while object stays put -> d0 must refresh to the new distance
    _call(term, _progress_env(torch.zeros(1, 3), torch.tensor([[2.0, 0.0, 0.0]])))
    assert term._d0.item() == pytest.approx(2.0, abs=1e-4)
    assert term._goal_w[0, 0].item() == pytest.approx(2.0, abs=1e-4)
