# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
import os
import subprocess
import sys
from collections.abc import Callable
from dataclasses import MISSING, asdict, dataclass, field
from functools import wraps
from typing import Any, ClassVar

import pytest
import torch

from isaaclab.utils import config_field, config_to_dict, copy_config, replace_config, update_config, validate_config
from isaaclab.utils.configclass import configclass
from isaaclab.utils.dict import class_to_dict, dict_to_md5_hash, update_class_from_dict
from isaaclab.utils.io import dump_yaml, load_yaml
from isaaclab.utils.string import ResolvableString

pytestmark = pytest.mark.unit

"""
Mock classes and functions.
"""


def dummy_function1() -> int:
    """Dummy function 1."""
    return 1


def dummy_function2() -> int:
    """Dummy function 2."""
    return 2


def dummy_wrapper(func):
    """Decorator for wrapping function."""

    @wraps(func)
    def wrapper():
        return func() + 1

    return wrapper


@dummy_wrapper
def wrapped_dummy_function3():
    """Dummy function 3."""
    return 3


@dummy_wrapper
def wrapped_dummy_function4():
    """Dummy function 4."""
    return 4


class DummyClass:
    """Dummy class."""

    def __init__(self):
        """Initialize dummy class."""
        self.a = 1
        self.b = 2


"""
Dummy configuration: Basic
"""


def double(x):
    """Dummy function."""
    return 2 * x


@dataclass
class ModifierCfg:
    params: dict[str, Any] = config_field({"A": 1, "B": 2})


@dataclass
class ViewerCfg:
    eye: list = config_field([7.5, 7.5, 7.5])  # field missing on purpose
    lookat: list = field(default_factory=lambda: [0.0, 0.0, 0.0])


@dataclass
class EnvCfg:
    num_envs: int = config_field(double(28))  # uses function for assignment
    episode_length: int = config_field(2000)
    viewer: ViewerCfg = config_field(ViewerCfg())


@dataclass
class RobotDefaultStateCfg:
    pos: Any = config_field((0.0, 0.0, 0.0))  # type annotation missing on purpose (immutable)
    rot: tuple = config_field((0.0, 0.0, 0.0, 1.0))  # xyzw format
    dof_pos: tuple = config_field((0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    dof_vel: Any = config_field([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])  # type annotation missing on purpose (mutable)


@dataclass
class BasicDemoCfg:
    """Dummy configuration class."""

    device_id: int = config_field(0)
    env: EnvCfg = config_field(EnvCfg())
    robot_default_state: RobotDefaultStateCfg = config_field(RobotDefaultStateCfg())
    list_config: Any = config_field([ModifierCfg(), ModifierCfg(params={"A": 3, "B": 4})])


@dataclass
class BasicDemoPostInitCfg:
    """Dummy configuration class."""

    device_id: int = config_field(0)
    env: EnvCfg = config_field(EnvCfg())
    robot_default_state: RobotDefaultStateCfg = config_field(RobotDefaultStateCfg())

    def __post_init__(self):
        self.device_id = 1
        self.add_variable = 3


@dataclass
class BasicDemoTorchCfg:
    """Dummy configuration class with a torch tensor ."""

    some_number: int = config_field(0)
    some_tensor: torch.Tensor = config_field(torch.Tensor([1, 2, 3]))


@dataclass
class BasicActuatorCfg:
    """Dummy configuration class for ActuatorBase config."""

    joint_names_expr: list[str] = config_field(["some_string"])
    joint_parameter_lookup: list[list[float]] = config_field([[1, 2, 3], [4, 5, 6]])
    stiffness: float = config_field(1.0)
    damping: float = config_field(2.0)


"""
Dummy configuration to check type annotations ordering.
"""


@dataclass
class TypeAnnotationOrderingDemoCfg:
    """Config class with type annotations."""

    anymal: RobotDefaultStateCfg = config_field(RobotDefaultStateCfg())
    unitree: RobotDefaultStateCfg = config_field(RobotDefaultStateCfg())
    franka: RobotDefaultStateCfg = config_field(RobotDefaultStateCfg())


@dataclass
class NonTypeAnnotationOrderingDemoCfg:
    """Config class without type annotations."""

    anymal: Any = config_field(RobotDefaultStateCfg())
    unitree: Any = config_field(RobotDefaultStateCfg())
    franka: Any = config_field(RobotDefaultStateCfg())


@dataclass
class InheritedNonTypeAnnotationOrderingDemoCfg(NonTypeAnnotationOrderingDemoCfg):
    """Inherited config class without type annotations."""

    pass


@dataclass
class MixedAnnotationOrderingDemoCfg:
    """Config class with type annotations on only some attributes."""

    plane: Any = config_field(RobotDefaultStateCfg())
    robot: Any = config_field(RobotDefaultStateCfg())
    peg: RobotDefaultStateCfg = config_field(RobotDefaultStateCfg())
    hole: RobotDefaultStateCfg = config_field(RobotDefaultStateCfg())
    camera: Any = config_field(RobotDefaultStateCfg())
    light: Any = config_field(RobotDefaultStateCfg())


@dataclass
class InheritedMixedAnnotationOrderingDemoCfg(MixedAnnotationOrderingDemoCfg):
    """Inherited config class with type annotations on only some attributes."""

    table: Any = config_field(RobotDefaultStateCfg())
    sensor: RobotDefaultStateCfg = config_field(RobotDefaultStateCfg())
    marker: Any = config_field(RobotDefaultStateCfg())


"""
Dummy configuration: Inheritance
"""


@dataclass
class ParentDemoCfg:
    """Dummy parent configuration with missing fields."""

    a: int = config_field(MISSING)  # add new missing field
    b: Any = config_field(2)  # type annotation missing on purpose
    c: RobotDefaultStateCfg = config_field(MISSING)  # add new missing field
    m: RobotDefaultStateCfg = config_field(RobotDefaultStateCfg())  # Add class type with defaults
    j: list[str] = config_field(MISSING)  # add new missing field
    i: list[str] = config_field(MISSING)  # add new missing field
    func: Callable = config_field(MISSING)  # add new missing field


@dataclass
class ChildADemoCfg(ParentDemoCfg):
    """Dummy child configuration with missing fields."""

    func: Any = config_field(dummy_function1)  # set default value for missing field
    c: Any = config_field(RobotDefaultStateCfg())  # set default value for missing field

    func_2: Callable = config_field(MISSING)  # add new missing field
    d: int = config_field(MISSING)  # add new missing field
    k: list[str] = config_field(["c", "d"])
    e: ViewerCfg = config_field(MISSING)  # add new missing field

    dummy_class: Any = config_field(DummyClass)

    def __post_init__(self):
        self.b = 3  # change value of existing field
        self.m.rot = (0.0, 0.0, 0.0, 2.0)  # change value of default (xyzw format)
        self.i = ["a", "b"]  # change value of existing field


@dataclass
class ChildBDemoCfg(ParentDemoCfg):
    """Dummy child configuration to test inheritance across instances."""

    a: Any = config_field(100)  # set default value for missing field
    j: Any = config_field(["3", "4"])  # set default value for missing field

    def __post_init__(self):
        self.b = 8  # change value of existing field
        self.i = ["1", "2"]  # change value of existing field


@dataclass
class ChildChildDemoCfg(ChildADemoCfg):
    """Dummy child configuration with missing fields."""

    func_2: Any = config_field(dummy_function2)
    d: Any = config_field(2)  # set default value for missing field

    def __post_init__(self):
        """Post initialization function."""
        if parent_post_init := getattr(super(), "__post_init__", None):
            parent_post_init()
        self.b = 4  # set default value for missing field
        self.f = "new"  # add new missing field


"""
Configuration with class inside.
"""


@dataclass
class DummyClassCfg:
    """Dummy class configuration with class type."""

    class_name_1: type = config_field(DummyClass)
    class_name_2: type[DummyClass] = config_field(DummyClass)
    class_name_3: Any = config_field(DummyClass)
    class_name_4: ClassVar[type[DummyClass]] = DummyClass

    b: str = config_field("dummy")


"""
Configuration with nested classes.
"""


@dataclass
class OutsideClassCfg:
    """Outermost dummy configuration."""

    @dataclass
    class InsideClassCfg:
        """Inner dummy configuration."""

        @dataclass
        class InsideInsideClassCfg:
            """Dummy configuration with class type."""

            u: list[int] = config_field([1, 2, 3])

        class_type: type = config_field(DummyClass)
        b: str = config_field("dummy")

    inside: InsideClassCfg = config_field(InsideClassCfg())
    x: int = config_field(20)

    def __post_init__(self):
        self.inside.b = "dummy_changed"


"""
Dummy configuration: Functions
"""


@dataclass
class FunctionsDemoCfg:
    """Dummy configuration class with functions as attributes."""

    func: Any = config_field(dummy_function1)
    wrapped_func: Any = config_field(wrapped_dummy_function3)
    func_in_dict: Any = config_field({"func": dummy_function1})


@dataclass
class FunctionImplementedDemoCfg:
    """Dummy configuration class with functions as attributes."""

    func: Any = config_field(dummy_function1)
    a: int = config_field(5)
    k: Any = config_field(100.0)

    def set_a(self, a: int):
        self.a = a


@dataclass
class ClassFunctionImplementedDemoCfg:
    """Dummy configuration class with function members defined in the class."""

    a: int = config_field(5)

    def instance_method(self):
        print("Value of a: ", self.a)

    @classmethod
    def class_method(cls, value: int) -> ClassFunctionImplementedDemoCfg:
        return cls(a=value)

    @property
    def a_proxy(self) -> int:
        return self.a

    @a_proxy.setter
    def a_proxy(self, value: int):
        self.a = value


"""
Dummy configuration: Nested dictionaries
"""


@dataclass
class NestedDictAndListCfg:
    """Dummy configuration class with nested dictionaries and lists."""

    dict_1: dict = config_field({"dict_2": {"func": dummy_function1}})
    list_1: list[EnvCfg] = config_field([EnvCfg(), EnvCfg()])


"""
Dummy configuration: Missing attributes
"""


@dataclass
class MissingParentDemoCfg:
    """Dummy parent configuration with missing fields."""

    a: int = config_field(MISSING)

    @dataclass
    class InsideClassCfg:
        """Inner dummy configuration."""

        @dataclass
        class InsideInsideClassCfg:
            """Inner inner dummy configuration."""

            a: str = config_field(MISSING)

        inside: str = config_field(MISSING)
        inside_dict: Any = config_field({"a": MISSING})
        inside_nested_dict: Any = config_field({"a": {"b": "hello", "c": MISSING, "d": InsideInsideClassCfg()}})
        inside_tuple: Any = config_field((10, MISSING, 20))
        inside_list: Any = config_field([MISSING, MISSING, 2])

    b: InsideClassCfg = config_field(InsideClassCfg())


@dataclass
class MissingChildDemoCfg(MissingParentDemoCfg):
    """Dummy child configuration with missing fields."""

    c: Callable = config_field(MISSING)
    d: int | None = config_field(None)
    e: dict = config_field({})


"""
Test solutions: Basic
"""

basic_demo_cfg_correct = {
    "env": {"num_envs": 56, "episode_length": 2000, "viewer": {"eye": [7.5, 7.5, 7.5], "lookat": [0.0, 0.0, 0.0]}},
    "robot_default_state": {
        "pos": (0.0, 0.0, 0.0),
        "rot": (0.0, 0.0, 0.0, 1.0),
        "dof_pos": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        "dof_vel": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    },
    "device_id": 0,
    "list_config": [{"params": {"A": 1, "B": 2}}, {"params": {"A": 3, "B": 4}}],
}

basic_demo_cfg_change_correct = {
    "env": {"num_envs": 22, "episode_length": 2000, "viewer": {"eye": (2.0, 2.0, 2.0), "lookat": [0.0, 0.0, 0.0]}},
    "robot_default_state": {
        "pos": (0.0, 0.0, 0.0),
        "rot": (0.0, 0.0, 0.0, 1.0),
        "dof_pos": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        "dof_vel": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    },
    "device_id": 0,
    "list_config": [{"params": {"A": 1, "B": 2}}, {"params": {"A": 3, "B": 4}}],
}

basic_demo_cfg_change_with_none_correct = {
    "env": {"num_envs": 22, "episode_length": 2000, "viewer": None},
    "robot_default_state": {
        "pos": (0.0, 0.0, 0.0),
        "rot": (0.0, 0.0, 0.0, 1.0),
        "dof_pos": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        "dof_vel": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    },
    "device_id": 0,
    "list_config": [{"params": {"A": 1, "B": 2}}, {"params": {"A": 3, "B": 4}}],
}

basic_demo_cfg_change_with_tuple_correct = {
    "env": {"num_envs": 56, "episode_length": 2000, "viewer": {"eye": [7.5, 7.5, 7.5], "lookat": [0.0, 0.0, 0.0]}},
    "robot_default_state": {
        "pos": (0.0, 0.0, 0.0),
        "rot": (0.0, 0.0, 0.0, 1.0),
        "dof_pos": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        "dof_vel": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    },
    "device_id": 0,
    "list_config": [{"params": {"A": -1, "B": -2}}, {"params": {"A": -3, "B": -4}}],
}

basic_demo_cfg_nested_dict_and_list = {
    "dict_1": {
        "dict_2": {"func": "test_configclass:dummy_function2"},
    },
    "list_1": [
        {"num_envs": 23, "episode_length": 3000, "viewer": {"eye": [5.0, 5.0, 5.0], "lookat": [0.0, 0.0, 0.0]}},
        {"num_envs": 24, "episode_length": 2000, "viewer": {"eye": [6.0, 6.0, 6.0], "lookat": [0.0, 0.0, 0.0]}},
    ],
}

basic_demo_post_init_cfg_correct = {
    "env": {"num_envs": 56, "episode_length": 2000, "viewer": {"eye": [7.5, 7.5, 7.5], "lookat": [0.0, 0.0, 0.0]}},
    "robot_default_state": {
        "pos": (0.0, 0.0, 0.0),
        "rot": (0.0, 0.0, 0.0, 1.0),
        "dof_pos": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        "dof_vel": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    },
    "device_id": 1,
    "add_variable": 3,
}

"""
Test solutions: Functions
"""

functions_demo_cfg_correct = {
    "func": "test_configclass:dummy_function1",
    "wrapped_func": "test_configclass:wrapped_dummy_function3",
    "func_in_dict": {"func": "test_configclass:dummy_function1"},
}

functions_demo_cfg_for_updating = {
    "func": "test_configclass:dummy_function2",
    "wrapped_func": "test_configclass:wrapped_dummy_function4",
    "func_in_dict": {"func": "test_configclass:dummy_function2"},
}

"""
Test solutions: Missing attributes
"""

validity_expected_fields = [
    "a",
    "b.inside",
    "b.inside_dict.a",
    "b.inside_nested_dict.a.c",
    "b.inside_nested_dict.a.d.a",
    "b.inside_tuple[1]",
    "b.inside_list[0]",
    "b.inside_list[1]",
    "c",
]

"""
Test fixtures.
"""


def test_str():
    """Test printing the configuration."""
    cfg = BasicDemoCfg()
    print()
    print(cfg)


def test_str_dict():
    """Test printing the configuration using dataclass utility."""
    cfg = BasicDemoCfg()
    print()
    print("Using dataclass function: ", asdict(cfg))
    print("Using internal function: ", config_to_dict(cfg))
    assert asdict(cfg) == config_to_dict(cfg)


def test_dict_conversion():
    """Test dictionary conversion of configclass instance."""
    cfg = BasicDemoCfg()
    # dataclass function
    assert asdict(cfg) == basic_demo_cfg_correct
    assert asdict(cfg.env) == basic_demo_cfg_correct["env"]
    # utility function
    assert class_to_dict(cfg) == basic_demo_cfg_correct
    assert class_to_dict(cfg.env) == basic_demo_cfg_correct["env"]
    # internal function
    assert config_to_dict(cfg) == basic_demo_cfg_correct
    assert config_to_dict(cfg.env) == basic_demo_cfg_correct["env"]

    torch_cfg = BasicDemoTorchCfg()
    torch_cfg_dict = config_to_dict(torch_cfg)
    # We have to do a manual check because torch.Tensor does not work with assertDictEqual.
    assert torch_cfg_dict["some_number"] == 0
    assert torch.all(torch_cfg_dict["some_tensor"] == torch.tensor([1, 2, 3]))


def test_actuator_cfg_dict_conversion():
    """Test dict conversion of ActuatorConfig."""
    # create a basic RemotizedPDActuator config
    actuator_cfg = BasicActuatorCfg()
    # return writable attributes of config object
    actuator_cfg_dict_attr = actuator_cfg.__dict__
    # check if __dict__ attribute of config is not empty
    assert len(actuator_cfg_dict_attr) > 0
    # class_to_dict utility function should return a primitive dictionary
    actuator_cfg_dict = class_to_dict(actuator_cfg)
    assert isinstance(actuator_cfg_dict, dict)


def test_dict_conversion_order():
    """Tests that order is conserved when converting to dictionary."""
    true_outer_order = ["device_id", "env", "robot_default_state", "list_config"]
    true_env_order = ["num_envs", "episode_length", "viewer"]
    # create config
    cfg = BasicDemoCfg()
    # check ordering
    for label, parsed_value in zip(true_outer_order, cfg.__dict__.keys()):
        assert label == parsed_value
    for label, parsed_value in zip(true_env_order, cfg.env.__dict__.keys()):
        assert label == parsed_value
    # convert config to dictionary
    cfg_dict = class_to_dict(cfg)
    # check ordering
    for label, parsed_value in zip(true_outer_order, cfg_dict.keys()):
        assert label == parsed_value
    for label, parsed_value in zip(true_env_order, cfg_dict["env"].keys()):
        assert label == parsed_value
    # check ordering when copied
    cfg_dict_copied = copy.deepcopy(cfg_dict)
    cfg_dict_copied.pop("list_config")
    # check ordering
    for label, parsed_value in zip(true_outer_order, cfg_dict_copied.keys()):
        assert label == parsed_value
    for label, parsed_value in zip(true_env_order, cfg_dict_copied["env"].keys()):
        assert label == parsed_value


def test_config_update_via_constructor():
    """Test updating configclass through initialization."""
    cfg = BasicDemoCfg(env=EnvCfg(num_envs=22, viewer=ViewerCfg(eye=(2.0, 2.0, 2.0))))
    assert asdict(cfg) == basic_demo_cfg_change_correct


def test_config_update_after_init():
    """Test updating configclass using instance members."""
    cfg = BasicDemoCfg()
    cfg.env.num_envs = 22
    cfg.env.viewer.eye = (2.0, 2.0, 2.0)  # note: changes from list to tuple
    assert asdict(cfg) == basic_demo_cfg_change_correct


def test_config_update_dict():
    """Test updating configclass using dictionary."""
    cfg = BasicDemoCfg()
    cfg_dict = {"env": {"num_envs": 22, "viewer": {"eye": (2.0, 2.0, 2.0)}}}
    update_class_from_dict(cfg, cfg_dict)
    assert asdict(cfg) == basic_demo_cfg_change_correct

    # check types are also correct
    assert isinstance(cfg.env.viewer, ViewerCfg)
    assert isinstance(cfg.env.viewer.eye, tuple)


def test_config_update_dict_with_none():
    """Test updating configclass using a dictionary that contains None."""
    cfg = BasicDemoCfg()
    cfg_dict = {"env": {"num_envs": 22, "viewer": None}}
    update_class_from_dict(cfg, cfg_dict)
    assert asdict(cfg) == basic_demo_cfg_change_with_none_correct


def test_config_update_dict_tuple():
    """Test updating configclass using a dictionary that modifies a tuple."""
    cfg = BasicDemoCfg()
    cfg_dict = {"list_config": [{"params": {"A": -1, "B": -2}}, {"params": {"A": -3, "B": -4}}]}
    update_class_from_dict(cfg, cfg_dict)
    assert asdict(cfg) == basic_demo_cfg_change_with_tuple_correct


def test_config_update_nested_dict():
    """Test updating configclass with sub-dictionaries."""
    cfg = NestedDictAndListCfg()
    cfg_dict = {
        "dict_1": {"dict_2": {"func": "test_configclass:dummy_function2"}},
        "list_1": [
            {"num_envs": 23, "episode_length": 3000, "viewer": {"eye": [5.0, 5.0, 5.0]}},
            {"num_envs": 24, "viewer": {"eye": [6.0, 6.0, 6.0]}},
        ],
    }
    update_class_from_dict(cfg, cfg_dict)
    assert asdict(cfg) == basic_demo_cfg_nested_dict_and_list

    # check types are also correct
    assert isinstance(cfg.list_1[0], EnvCfg)
    assert isinstance(cfg.list_1[1], EnvCfg)
    assert isinstance(cfg.list_1[0].viewer, ViewerCfg)
    assert isinstance(cfg.list_1[1].viewer, ViewerCfg)


def test_wrap_resolvable_strings_handles_cyclic_containers():
    """Cyclic container graphs in config values should not recurse forever."""

    @dataclass
    class CyclicContainerCfg:
        payload: dict[str, Any] = field(default_factory=dict)

        def __post_init__(self):
            cycle = {}
            cycle["self"] = cycle
            cycle["tuple"] = (cycle, {"back": cycle})
            self.payload = cycle

    cfg = CyclicContainerCfg()

    assert cfg.payload["self"] is cfg.payload
    assert cfg.payload["tuple"][0] is cfg.payload
    assert cfg.payload["tuple"][1]["back"] is cfg.payload


def test_dir_resolution_uses_declaring_class_for_inherited_field():
    """{DIR} expansion should use the field declaring class, not subclass module."""

    @dataclass
    class _BaseCfg:
        class_type: type | str = config_field("{DIR}.base_mod:BaseSymbol")

    @dataclass
    class _ChildCfg(_BaseCfg):
        pass

    cfg = _ChildCfg()

    assert isinstance(cfg.class_type, ResolvableString)
    assert str(cfg.class_type) == "test_configclass.base_mod:BaseSymbol"


def test_standard_field_does_not_apply_config_string_resolution():
    """Standard dataclass fields should remain free of Isaac Lab-specific processing."""

    @dataclass
    class _BaseCfg:
        class_type: str = field(default_factory=lambda: "{DIR}.base_mod:BaseSymbol")

    @dataclass
    class _ChildCfg(_BaseCfg):
        pass

    cfg = _ChildCfg()

    assert cfg.class_type == "{DIR}.base_mod:BaseSymbol"


def test_config_field_supports_frozen_configs():
    """Configuration fields should work with frozen standard dataclasses."""

    @dataclass(frozen=True)
    class FrozenCfg:
        values: list[int] = config_field([1, 2])

    first = FrozenCfg()
    second = copy_config(first)

    assert first.values == second.values
    assert first.values is not second.values


def test_config_field_supports_dataclass_subclasses():
    """Configuration fields should follow standard dataclass inheritance."""

    @dataclass
    class ParentCfg:
        values: list[int] = config_field([1, 2])

    @dataclass
    class ChildCfg(ParentCfg):
        runtime_cache: list[int] | None = config_field(None)

    cfg = ChildCfg()

    assert cfg.values == [1, 2]
    assert cfg.runtime_cache is None


def test_config_field_supports_class_body_customization_and_references():
    """Defaults can be customized and reused while a dataclass body is being declared."""

    @dataclass
    class NestedCfg:
        values: list[int] = config_field([1])

    @dataclass
    class ParentCfg:
        primary: NestedCfg = config_field(NestedCfg())
        primary.values.append(2)
        secondary: NestedCfg = config_field(primary)

    first = ParentCfg()
    second = ParentCfg()

    assert first.primary.values == [1, 2]
    assert first.secondary.values == [1, 2]
    assert first.primary is not first.secondary
    assert first.primary is not second.primary
    assert first.primary.values is not second.primary.values


def test_config_update_different_iterable_lengths():
    """Iterables are whole replaced, even if their lengths are different."""

    # original cfg has length-6 tuple and list
    cfg = RobotDefaultStateCfg()
    assert cfg.dof_pos == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert cfg.dof_vel == [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

    # patch uses different lengths
    patch = {
        "dof_pos": (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0),  # longer tuple
        "dof_vel": [9.0, 8.0, 7.0],  # shorter list
    }

    # should not raise
    update_class_from_dict(cfg, patch)

    # whole sequences are replaced
    assert cfg.dof_pos == (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)
    assert cfg.dof_vel == [9.0, 8.0, 7.0]


def test_config_update_dict_using_internal():
    """Test updating configclass from a dictionary using configclass method."""
    cfg = BasicDemoCfg()
    cfg_dict = {"env": {"num_envs": 22, "viewer": {"eye": (2.0, 2.0, 2.0)}}}
    update_config(cfg, cfg_dict)
    assert config_to_dict(cfg) == basic_demo_cfg_change_correct


def test_config_update_dict_using_post_init():
    cfg = BasicDemoPostInitCfg()
    assert config_to_dict(cfg) == basic_demo_post_init_cfg_correct


def test_invalid_update_key():
    """Test invalid key update."""
    cfg = BasicDemoCfg()
    cfg_dict = {"env": {"num_envs": 22, "viewer": {"pos": (2.0, 2.0, 2.0)}}}
    with pytest.raises(KeyError):
        update_class_from_dict(cfg, cfg_dict)


def test_multiple_instances():
    """Test multiple instances with twice instantiation."""
    # create two config instances
    cfg1 = BasicDemoCfg()
    cfg2 = BasicDemoCfg()

    # check variables
    # mutable -- variables should be different
    assert id(cfg1.env.viewer.eye) != id(cfg2.env.viewer.eye)
    assert id(cfg1.env.viewer.lookat) != id(cfg2.env.viewer.lookat)
    assert id(cfg1.robot_default_state) != id(cfg2.robot_default_state)
    # immutable -- variables are the same
    assert id(cfg1.robot_default_state.dof_pos) == id(cfg2.robot_default_state.dof_pos)
    assert id(cfg1.env.num_envs) == id(cfg2.env.num_envs)
    assert id(cfg1.device_id) == id(cfg2.device_id)

    # check values
    assert config_to_dict(cfg1.env) == config_to_dict(cfg2.env)
    assert config_to_dict(cfg1.robot_default_state) == config_to_dict(cfg2.robot_default_state)


def test_alter_values_multiple_instances():
    """Test alterations in multiple instances of the same configclass."""
    # create two config instances
    cfg1 = BasicDemoCfg()
    cfg2 = BasicDemoCfg()

    # alter configurations
    cfg1.env.num_envs = 22  # immutable data: int
    cfg1.env.viewer.eye[0] = 1.0  # mutable data: list
    cfg1.env.viewer.lookat[2] = 12.0  # mutable data: list

    # check variables
    # values should be different
    assert cfg1.env.num_envs != cfg2.env.num_envs
    assert cfg1.env.viewer.eye != cfg2.env.viewer.eye
    assert cfg1.env.viewer.lookat != cfg2.env.viewer.lookat
    # mutable -- variables are different ids
    assert id(cfg1.env.viewer.eye) != id(cfg2.env.viewer.eye)
    assert id(cfg1.env.viewer.lookat) != id(cfg2.env.viewer.lookat)
    # immutable -- altered variables are different ids
    assert id(cfg1.env.num_envs) != id(cfg2.env.num_envs)


def test_multiple_instances_with_replace():
    """Test multiple instances with creation through replace function."""
    # create two config instances
    cfg1 = BasicDemoCfg()
    cfg2 = replace_config(cfg1)

    # check variable IDs
    # mutable -- variables should be different
    assert id(cfg1.env.viewer.eye) != id(cfg2.env.viewer.eye)
    assert id(cfg1.env.viewer.lookat) != id(cfg2.env.viewer.lookat)
    assert id(cfg1.robot_default_state) != id(cfg2.robot_default_state)
    # immutable -- variables are the same
    assert id(cfg1.robot_default_state.dof_pos) == id(cfg2.robot_default_state.dof_pos)
    assert id(cfg1.env.num_envs) == id(cfg2.env.num_envs)
    assert id(cfg1.device_id) == id(cfg2.device_id)

    # check values
    assert config_to_dict(cfg1) == config_to_dict(cfg2)


def test_alter_values_multiple_instances_wth_replace():
    """Test alterations in multiple instances through replace function."""
    # create two config instances
    cfg1 = BasicDemoCfg()
    cfg2 = replace_config(cfg1, device_id=1)

    # alter configurations
    cfg1.env.num_envs = 22  # immutable data: int
    cfg1.env.viewer.eye[0] = 1.0  # mutable data: list
    cfg1.env.viewer.lookat[2] = 12.0  # mutable data: list

    # check variables
    # values should be different
    assert cfg1.env.num_envs != cfg2.env.num_envs
    assert cfg1.env.viewer.eye != cfg2.env.viewer.eye
    assert cfg1.env.viewer.lookat != cfg2.env.viewer.lookat
    # mutable -- variables are different ids
    assert id(cfg1.env.viewer.eye) != id(cfg2.env.viewer.eye)
    assert id(cfg1.env.viewer.lookat) != id(cfg2.env.viewer.lookat)
    # immutable -- altered variables are different ids
    assert id(cfg1.env.num_envs) != id(cfg2.env.num_envs)
    assert id(cfg1.device_id) != id(cfg2.device_id)


def test_configclass_type_ordering():
    """Checks ordering of config objects when no type annotation is provided."""

    cfg_1 = TypeAnnotationOrderingDemoCfg()
    cfg_2 = NonTypeAnnotationOrderingDemoCfg()
    cfg_3 = InheritedNonTypeAnnotationOrderingDemoCfg()

    # check ordering
    assert list(cfg_1.__dict__.keys()) == list(cfg_2.__dict__.keys())
    assert list(cfg_3.__dict__.keys()) == list(cfg_2.__dict__.keys())
    assert list(cfg_1.__dict__.keys()) == list(cfg_3.__dict__.keys())


def test_configclass_mixed_type_annotations_ordering():
    """Checks that declaration order is preserved when only some attributes have type annotations.

    Reference: https://github.com/isaac-sim/IsaacLab/issues/1949
    """
    cfg = MixedAnnotationOrderingDemoCfg()
    expected_order = ["plane", "robot", "peg", "hole", "camera", "light"]

    # check ordering of attributes and dictionary conversion
    assert list(cfg.__dict__.keys()) == expected_order
    assert list(config_to_dict(cfg).keys()) == expected_order

    # check ordering with inheritance: parent fields first, then child fields in declaration order
    cfg_inherited = InheritedMixedAnnotationOrderingDemoCfg()
    expected_inherited_order = expected_order + ["table", "sensor", "marker"]

    assert list(cfg_inherited.__dict__.keys()) == expected_inherited_order
    assert list(config_to_dict(cfg_inherited).keys()) == expected_inherited_order


def test_functions_config():
    """Tests having functions as values in the configuration instance."""
    cfg = FunctionsDemoCfg()
    # check calling
    assert cfg.func() == 1
    assert cfg.wrapped_func() == 4
    assert cfg.func_in_dict["func"]() == 1


def test_function_impl_config():
    """Tests having function defined in the class instance."""
    cfg = FunctionImplementedDemoCfg()
    # change value
    assert cfg.a == 5
    cfg.set_a(10)
    assert cfg.a == 10


def test_class_function_impl_config():
    """Tests having class function defined in the class instance."""
    cfg = ClassFunctionImplementedDemoCfg()

    # check that the annotations are correct
    assert cfg.__annotations__ == {"a": "int"}

    # check all methods are callable
    cfg.instance_method()
    new_cfg1 = cfg.class_method(20)
    # check value is correct
    assert new_cfg1.a == 20

    # create the same config instance using class method
    new_cfg2 = ClassFunctionImplementedDemoCfg.class_method(20)
    # check value is correct
    assert new_cfg2.a == 20


def test_class_property_impl_config():
    """Tests having class property defined in the class instance."""
    cfg = ClassFunctionImplementedDemoCfg()

    # check that the annotations are correct
    assert cfg.__annotations__ == {"a": "int"}

    # check all methods are callable
    cfg.instance_method()

    # check value is correct
    assert cfg.a == 5
    assert cfg.a_proxy == 5

    # set through property
    cfg.a_proxy = 10
    assert cfg.a == 10
    assert cfg.a_proxy == 10


def test_dict_conversion_functions_config():
    """Tests conversion of config with functions into dictionary."""
    cfg = FunctionsDemoCfg()
    cfg_dict = class_to_dict(cfg)
    assert cfg_dict["func"] == functions_demo_cfg_correct["func"]
    assert cfg_dict["wrapped_func"] == functions_demo_cfg_correct["wrapped_func"]
    assert cfg_dict["func_in_dict"]["func"] == functions_demo_cfg_correct["func_in_dict"]["func"]


def test_update_functions_config_with_functions():
    """Tests updating config with functions."""
    cfg = FunctionsDemoCfg()
    # update config
    update_class_from_dict(cfg, functions_demo_cfg_for_updating)
    # check calling
    assert cfg.func() == 2
    assert cfg.wrapped_func() == 5
    assert cfg.func_in_dict["func"]() == 2


def test_config_field_supports_missing_values():
    """Explicitly annotated fields should retain the validation sentinel."""

    @dataclass
    class MissingValueCfg:
        value: int = config_field(MISSING)

    with pytest.raises(TypeError, match="value"):
        validate_config(MissingValueCfg())


def test_required_argument_for_missing_type_in_config():
    """Tests required positional argument for missing type annotation in config creation."""

    @dataclass
    class MissingTypeDemoCfg:
        a: int = config_field(1)
        b: Any = config_field(2)
        c: int = config_field(MISSING)

    # should complain that 'c' is missed in positional arguments
    # TODO: Uncomment this when we move to 3.10.
    # with self.assertRaises(TypeError):
    #     cfg = MissingTypeDemoCfg(a=1)
    # should not complain
    cfg = MissingTypeDemoCfg(a=1, c=3)

    assert cfg.a == 1
    assert cfg.b == 2


def test_config_inheritance():
    """Tests that inheritance works properly."""
    # check variables
    cfg_a = ChildADemoCfg(a=20, d=3, e=ViewerCfg(), j=["c", "d"])

    assert cfg_a.func == dummy_function1
    assert cfg_a.a == 20
    assert cfg_a.d == 3
    assert cfg_a.j == ["c", "d"]

    # check post init
    assert cfg_a.b == 3
    assert cfg_a.i == ["a", "b"]
    assert cfg_a.m.rot == (0.0, 0.0, 0.0, 2.0)


def test_config_inheritance_independence():
    """Tests that subclass instantions have fully unique members,
    rather than references to members of the parent class"""
    # instantiate two classes which inherit from a shared parent,
    # but which will differently modify their members in their
    # __init__ and  __post_init__
    cfg_a = ChildADemoCfg()
    cfg_b = ChildBDemoCfg()

    # Test various combinations of initialization
    # and defaults across inherited members in
    # instances to verify independence between the subclasses
    assert isinstance(cfg_a.a, type(MISSING))
    assert cfg_b.a == 100
    assert cfg_a.b == 3
    assert cfg_b.b == 8
    assert cfg_a.c == RobotDefaultStateCfg()
    assert isinstance(cfg_b.c, type(MISSING))
    assert cfg_a.m.rot == (0.0, 0.0, 0.0, 2.0)
    assert cfg_b.m.rot == (0.0, 0.0, 0.0, 1.0)
    assert isinstance(cfg_a.j, type(MISSING))
    assert cfg_b.j == ["3", "4"]
    assert cfg_a.i == ["a", "b"]
    assert cfg_b.i == ["1", "2"]
    assert cfg_a.func == dummy_function1
    assert isinstance(cfg_b.func, type(MISSING))

    # Explicitly assert that members are not the same object
    # for different levels and kinds of data types
    assert cfg_a.m != cfg_b.m
    assert cfg_a.m.rot != cfg_b.m.rot
    assert cfg_a.i != cfg_b.i
    assert cfg_a.b != cfg_b.b


def test_config_double_inheritance():
    """Tests that inheritance works properly when inheriting twice."""
    # check variables
    cfg = ChildChildDemoCfg(a=20, d=3, e=ViewerCfg(), j=["c", "d"])

    assert cfg.func == dummy_function1
    assert cfg.func_2 == dummy_function2
    assert cfg.a == 20
    assert cfg.d == 3
    assert cfg.j == ["c", "d"]

    # check post init
    assert cfg.b == 4
    assert cfg.f == "new"
    assert cfg.i == ["a", "b"]


def test_config_with_class_type():
    """Tests that configclass works properly with class type."""

    cfg = DummyClassCfg()

    # since python 3.10, annotations are stored as strings
    annotations = {k: eval(v) if isinstance(v, str) else v for k, v in cfg.__annotations__.items()}
    # check types
    assert annotations["class_name_1"] is type
    assert annotations["class_name_2"] == type[DummyClass]
    assert annotations["class_name_3"] is Any
    assert annotations["class_name_4"] is ClassVar[type[DummyClass]]
    # check values
    assert cfg.class_name_1 == DummyClass
    assert cfg.class_name_2 == DummyClass
    assert cfg.class_name_3 == DummyClass
    assert cfg.class_name_4 == DummyClass
    assert cfg.b == "dummy"


def test_nested_config_class_declarations():
    """Tests that configclass works properly with nested class class declarations."""

    cfg = OutsideClassCfg()

    # check types
    assert "InsideClassCfg" not in cfg.__annotations__
    assert "InsideClassCfg" not in OutsideClassCfg.__annotations__
    assert "InsideInsideClassCfg" not in OutsideClassCfg.InsideClassCfg.__annotations__
    assert "InsideInsideClassCfg" not in cfg.inside.__annotations__
    # check values
    assert cfg.inside.class_type == DummyClass
    assert cfg.inside.b == "dummy_changed"
    assert cfg.x == 20


def test_config_dumping():
    """Check that config dumping works properly."""

    # file for dumping
    dirname = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(dirname, "output", "configclass", "test_config.yaml")

    # create config
    cfg = ChildADemoCfg(a=20, d=3, e=ViewerCfg(), j=["c", "d"])

    # save config
    dump_yaml(filename, cfg)
    # load config
    cfg_loaded = load_yaml(filename)
    # check dictionaries are the same
    assert list(config_to_dict(cfg).keys()) == list(cfg_loaded.keys())
    assert config_to_dict(cfg) == cfg_loaded

    # save config with sorted order won't work!
    # save config
    dump_yaml(filename, cfg, sort_keys=True)
    # load config
    cfg_loaded = load_yaml(filename)
    # check dictionaries are the same
    assert list(config_to_dict(cfg).keys()) != list(cfg_loaded.keys())
    assert config_to_dict(cfg) == cfg_loaded


def test_config_md5_hash():
    """Check that config md5 hash generation works properly."""

    # create config
    cfg = ChildADemoCfg(a=20, d=3, e=ViewerCfg(), j=["c", "d"])

    # generate md5 hash
    md5_hash_1 = dict_to_md5_hash(config_to_dict(cfg))
    md5_hash_2 = dict_to_md5_hash(config_to_dict(cfg))

    assert md5_hash_1 == md5_hash_2


def test_validity():
    """Check that invalid configurations raise errors."""

    cfg = MissingChildDemoCfg()

    with pytest.raises(TypeError) as context:
        validate_config(cfg)

    # check that the expected missing fields are in the error message
    error_message = str(context.value)
    for elem in validity_expected_fields:
        assert elem in error_message

    # check that no more than the expected missing fields are in the error message
    assert len(error_message.split("\n")) - 2 == len(validity_expected_fields)


# =============================================================================
# Tests: checked_apply
# =============================================================================


def test_checked_apply_forwards_all_fields():
    """checked_apply forwards every declared field on src onto target."""
    from dataclasses import dataclass as plain_dataclass

    from isaaclab.utils import checked_apply

    @dataclass
    class WrapperCfg:
        gap: float = config_field(0.01)
        margin: float = config_field(0.0)

    @plain_dataclass
    class UpstreamLike:
        gap: float = 99.0
        margin: float = 99.0
        unrelated: str = "keep me"

    src = WrapperCfg(margin=0.005)
    target = UpstreamLike()
    checked_apply(src, target)

    assert target.gap == 0.01
    assert target.margin == 0.005
    # fields not declared on src are not touched
    assert target.unrelated == "keep me"


def test_checked_apply_raises_on_missing_target_field():
    """checked_apply fails loudly when target lacks a declared field."""
    from dataclasses import dataclass as plain_dataclass

    from isaaclab.utils import checked_apply

    @dataclass
    class WrapperCfg:
        margin: float = config_field(0.01)
        renamed_in_upstream: float = config_field(0.0)

    @plain_dataclass
    class UpstreamMissingField:
        margin: float = 0.0
        # 'renamed_in_upstream' was renamed/removed upstream

    with pytest.raises(AttributeError, match="renamed_in_upstream"):
        checked_apply(WrapperCfg(), UpstreamMissingField())


def test_checked_apply_rejects_non_dataclass_src():
    """checked_apply requires src to be a dataclass."""
    from isaaclab.utils import checked_apply

    class NotADataclass:
        margin = 0.01

    with pytest.raises(TypeError, match="must be a dataclass"):
        checked_apply(NotADataclass(), object())


def test_configclass_emits_deprecation_warning():
    """The compatibility decorator should direct users to standard dataclasses."""
    with pytest.deprecated_call(match="functional configuration utilities"):

        @configclass
        class LegacyCfg:
            value: int = 1

    assert LegacyCfg().value == 1


@pytest.mark.parametrize("import_first", ["sub-module", "decorator"])
def test_configclass_name_works_for_every_import_form(import_first):
    """``isaaclab.utils.configclass`` serves both the decorator and the sub-module, in either order.

    The name belongs to a sub-module and to the decorator that sub-module defines, so whichever was
    imported first used to decide which one the other import forms got. A fresh interpreter is
    required because that is settled on the very first import.
    """
    prologue = {
        "sub-module": "import isaaclab.utils.configclass",
        "decorator": "from isaaclab.utils import configclass",
    }[import_first]
    script = f"""
{prologue}

import isaaclab.utils.configclass as configclass_module
assert configclass_module.checked_apply is not None

import isaaclab.utils
assert isaaclab.utils.configclass._field_module_dir is not None

from isaaclab.utils import configclass

@configclass
class DemoCfg:
    value: int = 1

assert DemoCfg().to_dict() == {{"value": 1}}
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
