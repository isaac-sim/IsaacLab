# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for :func:`isaaclab_tasks.utils.parse_cfg.parse_env_cfg`."""

import pytest

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def test_parse_env_cfg_rejects_bare_string_overrides():
    """A bare string is itself a ``Sequence[str]`` of characters; catch it explicitly."""
    with pytest.raises(TypeError, match="bare string"):
        parse_env_cfg("Isaac-Cartpole", overrides="physics=isaacsim_physx")


def test_parse_env_cfg_accepts_list_overrides():
    """A properly wrapped override list should apply without error."""
    env_cfg = parse_env_cfg("Isaac-Cartpole", overrides=["physics=isaacsim_physx"])
    assert env_cfg is not None


def test_parse_env_cfg_preserves_task_device_when_omitted():
    """An omitted device should preserve a task-specific simulation requirement."""
    env_cfg = parse_env_cfg("IsaacContrib-Stack-Cube-Galbot-Right-Arm-Suction-RmpFlow")

    assert env_cfg.sim.device == "cpu"
    env_cfg.validate()


def test_parse_env_cfg_applies_explicit_device_override():
    """An explicit device should continue to override the registered task default."""
    env_cfg = parse_env_cfg("Isaac-Cartpole", device="cpu")

    assert env_cfg.sim.device == "cpu"


@pytest.mark.parametrize("missing_module", ["pinocchio", "pink", "qpsolvers", "daqp", "unrelated_dependency"])
def test_task_config_missing_pink_dependency(monkeypatch: pytest.MonkeyPatch, missing_module: str):
    """Missing Pink dependencies should identify the task without hiding unrelated import errors."""
    from isaaclab_tasks.utils import parse_cfg

    task_name = "IsaacContrib-PickPlace-GR1T2-Abs"
    original_error = ModuleNotFoundError(f"No module named '{missing_module}'", name=missing_module)

    def import_missing_dependency(name: str):
        raise original_error

    monkeypatch.setattr(parse_cfg.importlib, "import_module", import_missing_dependency)
    if missing_module == "unrelated_dependency":
        with pytest.raises(ModuleNotFoundError) as exc_info:
            parse_cfg.load_cfg_from_registry(task_name, "env_cfg_entry_point")
        assert exc_info.value is original_error
    else:
        with pytest.raises(ImportError, match=f"{task_name}.*Pink IK requires Linux x86_64 or aarch64") as exc_info:
            parse_cfg.load_cfg_from_registry(task_name, "env_cfg_entry_point")
        assert exc_info.value.__cause__ is original_error


def test_pink_task_checks_lazy_dependencies(monkeypatch: pytest.MonkeyPatch):
    """A real Pink task must report missing dependencies even when its config imports lazily."""
    from isaaclab_tasks.utils import parse_cfg

    task_name = "IsaacContrib-PickPlace-GR1T2-Abs"
    original_find_spec = parse_cfg.importlib.util.find_spec

    def without_pinocchio(name: str, *args, **kwargs):
        return None if name == "pinocchio" else original_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(parse_cfg.importlib.util, "find_spec", without_pinocchio)
    with pytest.raises(ImportError, match=f"{task_name}.*Pink IK requires Linux x86_64 or aarch64"):
        parse_cfg.load_cfg_from_registry(task_name, "env_cfg_entry_point")
    assert parse_cfg.load_cfg_from_registry("Isaac-Cartpole", "env_cfg_entry_point") is not None
