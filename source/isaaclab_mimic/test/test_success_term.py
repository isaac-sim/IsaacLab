# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pytest
import torch

from isaaclab.managers import ManagerTermBase, TerminationTermCfg

from isaaclab_mimic.datagen.success_term import initialize_success_term, reset_success_term


class _StatefulSuccessTerm(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.reset_env_ids = None

    def __call__(self, env, threshold):
        return env.progress > threshold

    def reset(self, env_ids=None):
        self.reset_env_ids = env_ids


def test_class_based_success_term_lifecycle():
    """A class-based success term is instantiated, invoked, and reset."""
    env = Mock(progress=torch.tensor([0.25, 0.75]), num_envs=2, device="cpu")
    success_term = TerminationTermCfg(func=_StatefulSuccessTerm, params={"threshold": 0.5})

    resolved_success_term = initialize_success_term(success_term, env)

    assert resolved_success_term is success_term
    assert isinstance(success_term.func, _StatefulSuccessTerm)
    assert torch.equal(success_term.func(env, **success_term.params), torch.tensor([False, True]))

    env_ids = torch.tensor([1])
    reset_success_term(success_term, env_ids=env_ids)
    assert success_term.func.reset_env_ids is env_ids


def test_plain_function_success_term_is_unchanged():
    """A function-based success term retains its existing behavior."""

    def success(env, threshold):
        return env.progress > threshold

    env = Mock(progress=torch.tensor([0.75]))
    success_term = TerminationTermCfg(func=success, params={"threshold": 0.5})

    initialize_success_term(success_term, env)
    reset_success_term(success_term, env_ids=torch.tensor([0]))

    assert success_term.func is success
    assert bool(success_term.func(env, **success_term.params)[0])


def test_unrecognized_class_based_success_term_is_rejected():
    """A callable class must follow the manager term lifecycle."""

    class InvalidSuccessTerm:
        def __call__(self, env):
            return torch.ones(env.num_envs, dtype=torch.bool)

    success_term = TerminationTermCfg(func=InvalidSuccessTerm)

    with pytest.raises(TypeError, match="must inherit from ManagerTermBase"):
        initialize_success_term(success_term, Mock())
