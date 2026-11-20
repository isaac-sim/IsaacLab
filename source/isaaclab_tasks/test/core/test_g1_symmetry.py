# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reflection of the velocity-task G1, whose waist and Dex3 thumb joints have symmetric limits."""

from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from isaaclab_tasks.core.velocity.mdp.symmetry import g1

pytestmark = pytest.mark.unit

# (name, lower limit, upper limit): the hip roll and finger limits are flipped between sides, the rest are equal
_JOINTS = [
    ("left_hip_roll_joint", -0.5, 2.9),
    ("right_hip_roll_joint", -2.9, 0.5),
    ("left_knee_joint", -0.1, 2.9),
    ("right_knee_joint", -0.1, 2.9),
    ("torso_joint", -2.6, 2.6),
    ("left_zero_joint", -0.5, 0.5),
    ("right_zero_joint", -0.5, 0.5),
    ("left_one_joint", -1.0, 0.7),
    ("right_one_joint", -0.7, 1.0),
]
_NAMES = [name for name, _, _ in _JOINTS]


def _make_env(device: str = "cpu") -> SimpleNamespace:
    """Robot, action and observation layouts as ``compute_symmetric_states`` reads them, without a scene."""
    limits = torch.tensor([[[lo, hi] for _, lo, hi in _JOINTS]], device=device)
    data = SimpleNamespace(joint_pos_limits=limits, default_joint_pos=torch.zeros(1, len(_JOINTS), device=device))
    robot = SimpleNamespace(
        joint_names=_NAMES, data=data, find_joints=lambda names, preserve_order: (list(range(len(_NAMES))), _NAMES)
    )
    joint_term = SimpleNamespace(params={})
    env = SimpleNamespace(
        device=device,
        scene={"robot": robot},
        action_manager=SimpleNamespace(
            active_terms=["joint_pos"],
            get_term=lambda name: SimpleNamespace(cfg=SimpleNamespace(joint_names=".*", preserve_order=False)),
        ),
        observation_manager=SimpleNamespace(
            active_terms={"policy": ["base_ang_vel", "projected_gravity", "joint_pos"]},
            group_obs_term_dim={"policy": [(3,), (3,), (len(_JOINTS),)]},
            cfg={"policy": {"joint_pos": joint_term}},
        ),
    )
    env.unwrapped = env
    return env


def test_waist_yaw_and_thumb_rotation_resolve():
    """Joints whose limits and names leave the sign open use the explicit signs."""
    perm, sign = g1._build_map(_make_env())
    resolved = dict(zip(_NAMES, sign.tolist()))
    assert resolved["torso_joint"] == -1.0
    assert resolved["left_zero_joint"] == resolved["right_zero_joint"] == 1.0
    assert resolved["left_hip_roll_joint"] == -1.0 and resolved["left_knee_joint"] == 1.0
    assert [_NAMES[i] for i in perm.tolist()][:2] == ["right_hip_roll_joint", "left_hip_roll_joint"]


def test_reflection_is_an_involution_and_reuses_vector_factors():
    """Reflecting twice restores the batch, and repeated calls reuse the cached mirror factors."""
    env = _make_env()
    obs = TensorDict({"policy": torch.randn(4, 6 + len(_JOINTS))}, batch_size=[4])
    first, _ = g1.compute_symmetric_states(env, obs=obs)
    cached = {key: value for key, value in g1._FACTOR_CACHE.items() if key[1] == "cpu"}
    twice, _ = g1.compute_symmetric_states(env, obs=first[4:])
    torch.testing.assert_close(twice[4:]["policy"], obs["policy"])
    assert all(g1._FACTOR_CACHE[key] is value for key, value in cached.items())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph capture needs a GPU")
def test_reflection_can_be_captured_in_a_cuda_graph():
    """After one eager call, the reflection runs inside a captured CUDA graph without host-to-device copies."""
    env = _make_env("cuda")
    obs = TensorDict({"policy": torch.randn(8, 6 + len(_JOINTS), device="cuda")}, batch_size=[8])
    expected, _ = g1.compute_symmetric_states(env, obs=obs)
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        with torch.cuda.graph(graph):
            captured, _ = g1.compute_symmetric_states(env, obs=obs)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(captured["policy"], expected["policy"])
