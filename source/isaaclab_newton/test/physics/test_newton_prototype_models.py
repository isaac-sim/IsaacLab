# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from types import SimpleNamespace

import pytest
from isaaclab_newton.physics.newton_manager import NewtonManager


class _FakeBuilder:
    def __init__(self):
        self.finalize_calls = 0

    def finalize(self, device=None):
        self.finalize_calls += 1
        return SimpleNamespace(device=device)


@pytest.fixture(autouse=True)
def _restore_newton_manager_state():
    old_model = NewtonManager._model
    old_prototypes = NewtonManager._prototype_models
    try:
        yield
    finally:
        NewtonManager._model = old_model
        NewtonManager._prototype_models = old_prototypes


def test_get_prototype_model_matches_env_regex_path():
    builder = _FakeBuilder()
    NewtonManager._model = SimpleNamespace(device="cuda:0")
    NewtonManager.register_prototype_builders(
        ("/World/envs/env_0",), ("/World/envs/env_{}",), {"/World/envs/env_0": builder}
    )

    info = NewtonManager.get_prototype_model("/World/envs/env_.*/Robot")

    assert info.source_path == "/World/envs/env_0"
    assert info.model.device == "cuda:0"
    assert builder.finalize_calls == 1


def test_get_prototype_model_reuses_cached_model_on_same_device():
    builder = _FakeBuilder()
    NewtonManager._model = SimpleNamespace(device="cpu")
    NewtonManager.register_prototype_builders(
        ("/World/envs/env_0/Robot",), ("/World/envs/env_{}/Robot",), {"/World/envs/env_0/Robot": builder}
    )

    info_0 = NewtonManager.get_prototype_model("/World/envs/env_.*/Robot")
    info_1 = NewtonManager.get_prototype_model("/World/envs/env_.*/Robot")

    assert info_0.model is info_1.model
    assert builder.finalize_calls == 1


def test_register_prototype_builders_accumulates_across_calls():
    robot_builder = _FakeBuilder()
    prop_builder = _FakeBuilder()
    NewtonManager._model = SimpleNamespace(device="cpu")

    NewtonManager.register_prototype_builders(
        ("/World/envs/env_0/Robot",), ("/World/envs/env_{}/Robot",), {"/World/envs/env_0/Robot": robot_builder}
    )
    NewtonManager.register_prototype_builders(
        ("/World/envs/env_0/Prop",), ("/World/envs/env_{}/Prop",), {"/World/envs/env_0/Prop": prop_builder}
    )

    assert NewtonManager.get_prototype_model("/World/envs/env_.*/Robot").builder is robot_builder
    assert NewtonManager.get_prototype_model("/World/envs/env_.*/Prop").builder is prop_builder
