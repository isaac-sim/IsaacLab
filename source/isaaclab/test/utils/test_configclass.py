# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

# NOTE: While we don't actually use the simulation app in this test, we still need to launch it
#       because warp is only available in the context of a running simulation
"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher, run_tests

# launch omniverse app
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

"""Rest everything follows."""

import copy
import os
import torch
import unittest
from collections.abc import Callable
from dataclasses import MISSING, asdict, field
from functools import wraps
from typing import Any, ClassVar

from isaaclab.utils.configclass import configclass
from isaaclab.utils.dict import class_to_dict, dict_to_md5_hash, update_class_from_dict
from isaaclab.utils.io import dump_yaml, load_yaml

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


@configclass
class ModifierCfg:
    params: dict[str, Any] = {"A": 1, "B": 2}


@configclass
class ViewerCfg:
    eye: list = [7.5, 7.5, 7.5]  # field missing on purpose
    lookat: list = field(default_factory=lambda: [0.0, 0.0, 0.0])


@configclass
class EnvCfg:
    num_envs: int = double(28)  # uses function for assignment
    episode_length: int = 2000
    viewer: ViewerCfg = ViewerCfg()


@configclass
class RobotDefaultStateCfg:
    pos = (0.0, 0.0, 0.0)  # type annotation missing on purpose (immutable)
    rot: tuple = (1.0, 0.0, 0.0, 0.0)
    dof_pos: tuple = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    dof_vel = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]  # type annotation missing on purpose (mutable)


@configclass
class BasicDemoCfg:
    """Dummy configuration class."""

    device_id: int = 0
    env: EnvCfg = EnvCfg()
    robot_default_state: RobotDefaultStateCfg = RobotDefaultStateCfg()
    list_config = [ModifierCfg(), ModifierCfg(params={"A": 3, "B": 4})]


@configclass
class BasicDemoPostInitCfg:
    """Dummy configuration class."""

    device_id: int = 0
    env: EnvCfg = EnvCfg()
    robot_default_state: RobotDefaultStateCfg = RobotDefaultStateCfg()

    def __post_init__(self):
        self.device_id = 1
        self.add_variable = 3


@configclass
class BasicDemoTorchCfg:
    """Dummy configuration class with a torch tensor ."""

    some_number: int = 0
    some_tensor: torch.Tensor = torch.Tensor([1, 2, 3])


@configclass
class BasicActuatorCfg:
    """Dummy configuration class for ActuatorBase config."""

    joint_names_expr: list[str] = ["some_string"]
    joint_parameter_lookup: list[list[float]] = [[1, 2, 3], [4, 5, 6]]
    stiffness: float = 1.0
    damping: float = 2.0


"""
Dummy configuration to check type annotations ordering.
"""


@configclass
class TypeAnnotationOrderingDemoCfg:
    """Config class with type annotations."""

    anymal: RobotDefaultStateCfg = RobotDefaultStateCfg()
    unitree: RobotDefaultStateCfg = RobotDefaultStateCfg()
    franka: RobotDefaultStateCfg = RobotDefaultStateCfg()


@configclass
class NonTypeAnnotationOrderingDemoCfg:
    """Config class without type annotations."""

    anymal = RobotDefaultStateCfg()
    unitree = RobotDefaultStateCfg()
    franka = RobotDefaultStateCfg()


@configclass
class InheritedNonTypeAnnotationOrderingDemoCfg(NonTypeAnnotationOrderingDemoCfg):
    """Inherited config class without type annotations."""

    pass


"""
Dummy configuration: Inheritance
"""


@configclass
class ParentDemoCfg:
    """Dummy parent configuration with missing fields."""

    a: int = MISSING  # add new missing field
    b = 2  # type annotation missing on purpose
    c: RobotDefaultStateCfg = MISSING  # add new missing field
    m: RobotDefaultStateCfg = RobotDefaultStateCfg()  # Add class type with defaults
    j: list[str] = MISSING  # add new missing field
    i: list[str] = MISSING  # add new missing field
    func: Callable = MISSING  # add new missing field


@configclass
class ChildADemoCfg(ParentDemoCfg):
    """Dummy child configuration with missing fields."""

    func = dummy_function1  # set default value for missing field
    c = RobotDefaultStateCfg()  # set default value for missing field

    func_2: Callable = MISSING  # add new missing field
    d: int = MISSING  # add new missing field
    k: list[str] = ["c", "d"]
    e: ViewerCfg = MISSING  # add new missing field

    dummy_class = DummyClass

    def __post_init__(self):
        self.b = 3  # change value of existing field
        self.m.rot = (2.0, 0.0, 0.0, 0.0)  # change value of default
        self.i = ["a", "b"]  # change value of existing field


@configclass
class ChildBDemoCfg(ParentDemoCfg):
    """Dummy child configuration to test inheritance across instances."""

    a = 100  # set default value for missing field
    j = ["3", "4"]  # set default value for missing field

    def __post_init__(self):
        self.b = 8  # change value of existing field
        self.i = ["1", "2"]  # change value of existing field


@configclass
class ChildChildDemoCfg(ChildADemoCfg):
    """Dummy child configuration with missing fields."""

    func_2 = dummy_function2
    d = 2  # set default value for missing field

    def __post_init__(self):
        """Post initialization function."""
        super().__post_init__()
        self.b = 4  # set default value for missing field
        self.f = "new"  # add new missing field


"""
Configuration with class inside.
"""


@configclass
class DummyClassCfg:
    """Dummy class configuration with class type."""

    class_name_1: type = DummyClass
    class_name_2: type[DummyClass] = DummyClass
    class_name_3 = DummyClass
    class_name_4: ClassVar[type[DummyClass]] = DummyClass

    b: str = "dummy"


"""
Configuration with nested classes.
"""


@configclass
class OutsideClassCfg:
    """Outermost dummy configuration."""

    @configclass
    class InsideClassCfg:
        """Inner dummy configuration."""

        @configclass
        class InsideInsideClassCfg:
            """Dummy configuration with class type."""

            u: list[int] = [1, 2, 3]

        class_type: type = DummyClass
        b: str = "dummy"

    inside: InsideClassCfg = InsideClassCfg()
    x: int = 20

    def __post_init__(self):
        self.inside.b = "dummy_changed"


"""
Dummy configuration: Functions
"""


@configclass
class FunctionsDemoCfg:
    """Dummy configuration class with functions as attributes."""

    func = dummy_function1
    wrapped_func = wrapped_dummy_function3
    func_in_dict = {"func": dummy_function1}


@configclass
class FunctionImplementedDemoCfg:
    """Dummy configuration class with functions as attributes."""

    func = dummy_function1
    a: int = 5
    k = 100.0

    def set_a(self, a: int):
        self.a = a


@configclass
class ClassFunctionImplementedDemoCfg:
    """Dummy configuration class with function members defined in the class."""

    a: int = 5

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


@configclass
class NestedDictAndListCfg:
    """Dummy configuration class with nested dictionaries and lists."""

    dict_1: dict = {"dict_2": {"func": dummy_function1}}
    list_1: list[EnvCfg] = [EnvCfg(), EnvCfg()]


"""
Dummy configuration: Missing attributes
"""


@configclass
class MissingParentDemoCfg:
    """Dummy parent configuration with missing fields."""

    a: int = MISSING

    @configclass
    class InsideClassCfg:
        """Inner dummy configuration."""

        @configclass
        class InsideInsideClassCfg:
            """Inner inner dummy configuration."""

            a: str = MISSING

        inside: str = MISSING
        inside_dict = {"a": MISSING}
        inside_nested_dict = {"a": {"b": "hello", "c": MISSING, "d": InsideInsideClassCfg()}}
        inside_tuple = (10, MISSING, 20)
        inside_list = [MISSING, MISSING, 2]

    b: InsideClassCfg = InsideClassCfg()


@configclass
class MissingChildDemoCfg(MissingParentDemoCfg):
    """Dummy child configuration with missing fields."""

    c: Callable = MISSING
    d: int | None = None
    e: dict = {}


"""
Test solutions: Basic
"""

basic_demo_cfg_correct = {
    "env": {"num_envs": 56, "episode_length": 2000, "viewer": {"eye": [7.5, 7.5, 7.5], "lookat": [0.0, 0.0, 0.0]}},
    "robot_default_state": {
        "pos": (0.0, 0.0, 0.0),
        "rot": (1.0, 0.0, 0.0, 0.0),
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
        "rot": (1.0, 0.0, 0.0, 0.0),
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
        "rot": (1.0, 0.0, 0.0, 0.0),
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
        "rot": (1.0, 0.0, 0.0, 0.0),
        "dof_pos": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        "dof_vel": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    },
    "device_id": 0,
    "list_config": [{"params": {"A": -1, "B": -2}}, {"params": {"A": -3, "B": -4}}],
}

basic_demo_cfg_nested_dict_and_list = {
    "dict_1": {
        "dict_2": {"func": dummy_function2},
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
        "rot": (1.0, 0.0, 0.0, 0.0),
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
    "func": "__main__:dummy_function1",
    "wrapped_func": "__main__:wrapped_dummy_function3",
    "func_in_dict": {"func": "__main__:dummy_function1"},
}

functions_demo_cfg_for_updating = {
    "func": "__main__:dummy_function2",
    "wrapped_func": "__main__:wrapped_dummy_function4",
    "func_in_dict": {"func": "__main__:dummy_function2"},
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


class TestConfigClass(unittest.TestCase):
    """Test cases for various situations with configclass decorator for configuration."""

    def test_str(self):
        """Test printing the configuration."""
        cfg = BasicDemoCfg()
        print()
        print(cfg)

    def test_str_dict(self):
        """Test printing the configuration using dataclass utility."""
        cfg = BasicDemoCfg()
        print()
        print("Using dataclass function: ", asdict(cfg))
        print("Using internal function: ", cfg.to_dict())
        self.assertDictEqual(asdict(cfg), cfg.to_dict())

    def test_dict_conversion(self):
        """Test dictionary conversion of configclass instance."""
        cfg = BasicDemoCfg()
        # dataclass function
        self.assertDictEqual(asdict(cfg), basic_demo_cfg_correct)
        self.assertDictEqual(asdict(cfg.env), basic_demo_cfg_correct["env"])
        # utility function
        self.assertDictEqual(class_to_dict(cfg), basic_demo_cfg_correct)
        self.assertDictEqual(class_to_dict(cfg.env), basic_demo_cfg_correct["env"])
        # internal function
        self.assertDictEqual(cfg.to_dict(), basic_demo_cfg_correct)
        self.assertDictEqual(cfg.env.to_dict(), basic_demo_cfg_correct["env"])

        torch_cfg = BasicDemoTorchCfg()
        torch_cfg_dict = torch_cfg.to_dict()
        # We have to do a manual check because torch.Tensor does not work with assertDictEqual.
        self.assertEqual(torch_cfg_dict["some_number"], 0)
        self.assertTrue(torch.all(torch_cfg_dict["some_tensor"] == torch.tensor([1, 2, 3])))

    def test_actuator_cfg_dict_conversion(self):
        """Test dict conversion of ActuatorConfig."""
        # create a basic RemotizedPDActuator config
        actuator_cfg = BasicActuatorCfg()
        # return writable attributes of config object
        actuator_cfg_dict_attr = actuator_cfg.__dict__
        # check if __dict__ attribute of config is not empty
        self.assertTrue(len(actuator_cfg_dict_attr) > 0)
        # class_to_dict utility function should return a primitive dictionary
        actuator_cfg_dict = class_to_dict(actuator_cfg)
        self.assertTrue(isinstance(actuator_cfg_dict, dict))

    def test_dict_conversion_order(self):
        """Tests that order is conserved when converting to dictionary."""
        true_outer_order = ["device_id", "env", "robot_default_state", "list_config"]
        true_env_order = ["num_envs", "episode_length", "viewer"]
        # create config
        cfg = BasicDemoCfg()
        # check ordering
        for label, parsed_value in zip(true_outer_order, cfg.__dict__.keys()):
            self.assertEqual(label, parsed_value)
        for label, parsed_value in zip(true_env_order, cfg.env.__dict__.keys()):
            self.assertEqual(label, parsed_value)
        # convert config to dictionary
        cfg_dict = class_to_dict(cfg)
        # check ordering
        for label, parsed_value in zip(true_outer_order, cfg_dict.keys()):
            self.assertEqual(label, parsed_value)
        for label, parsed_value in zip(true_env_order, cfg_dict["env"].keys()):
            self.assertEqual(label, parsed_value)
        # check ordering when copied
        cfg_dict_copied = copy.deepcopy(cfg_dict)
        cfg_dict_copied.pop("list_config")
        # check ordering
        for label, parsed_value in zip(true_outer_order, cfg_dict_copied.keys()):
            self.assertEqual(label, parsed_value)
        for label, parsed_value in zip(true_env_order, cfg_dict_copied["env"].keys()):
            self.assertEqual(label, parsed_value)

    def test_config_update_via_constructor(self):
        """Test updating configclass through initialization."""
        cfg = BasicDemoCfg(env=EnvCfg(num_envs=22, viewer=ViewerCfg(eye=(2.0, 2.0, 2.0))))
        self.assertDictEqual(asdict(cfg), basic_demo_cfg_change_correct)

    def test_config_update_after_init(self):
        """Test updating configclass using instance members."""
        cfg = BasicDemoCfg()
        cfg.env.num_envs = 22
        cfg.env.viewer.eye = (2.0, 2.0, 2.0)  # note: changes from list to tuple
        self.assertDictEqual(asdict(cfg), basic_demo_cfg_change_correct)

    def test_config_update_dict(self):
        """Test updating configclass using dictionary."""
        cfg = BasicDemoCfg()
        cfg_dict = {"env": {"num_envs": 22, "viewer": {"eye": (2.0, 2.0, 2.0)}}}
        update_class_from_dict(cfg, cfg_dict)
        self.assertDictEqual(asdict(cfg), basic_demo_cfg_change_correct)

        # check types are also correct
        self.assertIsInstance(cfg.env.viewer, ViewerCfg)
        self.assertIsInstance(cfg.env.viewer.eye, tuple)

    def test_config_update_dict_with_none(self):
        """Test updating configclass using a dictionary that contains None."""
        cfg = BasicDemoCfg()
        cfg_dict = {"env": {"num_envs": 22, "viewer": None}}
        update_class_from_dict(cfg, cfg_dict)
        self.assertDictEqual(asdict(cfg), basic_demo_cfg_change_with_none_correct)

    def test_config_update_dict_tuple(self):
        """Test updating configclass using a dictionary that modifies a tuple."""
        cfg = BasicDemoCfg()
        cfg_dict = {"list_config": [{"params": {"A": -1, "B": -2}}, {"params": {"A": -3, "B": -4}}]}
        update_class_from_dict(cfg, cfg_dict)
        self.assertDictEqual(asdict(cfg), basic_demo_cfg_change_with_tuple_correct)

    def test_config_update_nested_dict(self):
        """Test updating configclass with sub-dictionaries."""
        cfg = NestedDictAndListCfg()
        cfg_dict = {
            "dict_1": {"dict_2": {"func": "__main__:dummy_function2"}},
            "list_1": [
                {"num_envs": 23, "episode_length": 3000, "viewer": {"eye": [5.0, 5.0, 5.0]}},
                {"num_envs": 24, "viewer": {"eye": [6.0, 6.0, 6.0]}},
            ],
        }
        update_class_from_dict(cfg, cfg_dict)
        self.assertDictEqual(asdict(cfg), basic_demo_cfg_nested_dict_and_list)

        # check types are also correct
        self.assertIsInstance(cfg.list_1[0], EnvCfg)
        self.assertIsInstance(cfg.list_1[1], EnvCfg)
        self.assertIsInstance(cfg.list_1[0].viewer, ViewerCfg)
        self.assertIsInstance(cfg.list_1[1].viewer, ViewerCfg)

    def test_config_update_dict_using_internal(self):
        """Test updating configclass from a dictionary using configclass method."""
        cfg = BasicDemoCfg()
        cfg_dict = {"env": {"num_envs": 22, "viewer": {"eye": (2.0, 2.0, 2.0)}}}
        cfg.from_dict(cfg_dict)
        self.assertDictEqual(cfg.to_dict(), basic_demo_cfg_change_correct)

    def test_config_update_dict_using_post_init(self):
        cfg = BasicDemoPostInitCfg()
        self.assertDictEqual(cfg.to_dict(), basic_demo_post_init_cfg_correct)

    def test_invalid_update_key(self):
        """Test invalid key update."""
        cfg = BasicDemoCfg()
        cfg_dict = {"env": {"num_envs": 22, "viewer": {"pos": (2.0, 2.0, 2.0)}}}
        with self.assertRaises(KeyError):
            update_class_from_dict(cfg, cfg_dict)

    def test_multiple_instances(self):
        """Test multiple instances with twice instantiation."""
        # create two config instances
        cfg1 = BasicDemoCfg()
        cfg2 = BasicDemoCfg()

        # check variables
        # mutable -- variables should be different
        self.assertNotEqual(id(cfg1.env.viewer.eye), id(cfg2.env.viewer.eye))
        self.assertNotEqual(id(cfg1.env.viewer.lookat), id(cfg2.env.viewer.lookat))
        self.assertNotEqual(id(cfg1.robot_default_state), id(cfg2.robot_default_state))
        # immutable -- variables are the same
        self.assertEqual(id(cfg1.robot_default_state.dof_pos), id(cfg2.robot_default_state.dof_pos))
        self.assertEqual(id(cfg1.env.num_envs), id(cfg2.env.num_envs))
        self.assertEqual(id(cfg1.device_id), id(cfg2.device_id))

        # check values
        self.assertDictEqual(cfg1.env.to_dict(), cfg2.env.to_dict())
        self.assertDictEqual(cfg1.robot_default_state.to_dict(), cfg2.robot_default_state.to_dict())

    def test_alter_values_multiple_instances(self):
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
        self.assertNotEqual(cfg1.env.num_envs, cfg2.env.num_envs)
        self.assertNotEqual(cfg1.env.viewer.eye, cfg2.env.viewer.eye)
        self.assertNotEqual(cfg1.env.viewer.lookat, cfg2.env.viewer.lookat)
        # mutable -- variables are different ids
        self.assertNotEqual(id(cfg1.env.viewer.eye), id(cfg2.env.viewer.eye))
        self.assertNotEqual(id(cfg1.env.viewer.lookat), id(cfg2.env.viewer.lookat))
        # immutable -- altered variables are different ids
        self.assertNotEqual(id(cfg1.env.num_envs), id(cfg2.env.num_envs))

    def test_multiple_instances_with_replace(self):
        """Test multiple instances with creation through replace function."""
        # create two config instances
        cfg1 = BasicDemoCfg()
        cfg2 = cfg1.replace()

        # check variable IDs
        # mutable -- variables should be different
        self.assertNotEqual(id(cfg1.env.viewer.eye), id(cfg2.env.viewer.eye))
        self.assertNotEqual(id(cfg1.env.viewer.lookat), id(cfg2.env.viewer.lookat))
        self.assertNotEqual(id(cfg1.robot_default_state), id(cfg2.robot_default_state))
        # immutable -- variables are the same
        self.assertEqual(id(cfg1.robot_default_state.dof_pos), id(cfg2.robot_default_state.dof_pos))
        self.assertEqual(id(cfg1.env.num_envs), id(cfg2.env.num_envs))
        self.assertEqual(id(cfg1.device_id), id(cfg2.device_id))

        # check values
        self.assertDictEqual(cfg1.to_dict(), cfg2.to_dict())

    def test_alter_values_multiple_instances_wth_replace(self):
        """Test alterations in multiple instances through replace function."""
        # create two config instances
        cfg1 = BasicDemoCfg()
        cfg2 = cfg1.replace(device_id=1)

        # alter configurations
        cfg1.env.num_envs = 22  # immutable data: int
        cfg1.env.viewer.eye[0] = 1.0  # mutable data: list
        cfg1.env.viewer.lookat[2] = 12.0  # mutable data: list

        # check variables
        # values should be different
        self.assertNotEqual(cfg1.env.num_envs, cfg2.env.num_envs)
        self.assertNotEqual(cfg1.env.viewer.eye, cfg2.env.viewer.eye)
        self.assertNotEqual(cfg1.env.viewer.lookat, cfg2.env.viewer.lookat)
        # mutable -- variables are different ids
        self.assertNotEqual(id(cfg1.env.viewer.eye), id(cfg2.env.viewer.eye))
        self.assertNotEqual(id(cfg1.env.viewer.lookat), id(cfg2.env.viewer.lookat))
        # immutable -- altered variables are different ids
        self.assertNotEqual(id(cfg1.env.num_envs), id(cfg2.env.num_envs))
        self.assertNotEqual(id(cfg1.device_id), id(cfg2.device_id))

    def test_configclass_type_ordering(self):
        """Checks ordering of config objects when no type annotation is provided."""

        cfg_1 = TypeAnnotationOrderingDemoCfg()
        cfg_2 = NonTypeAnnotationOrderingDemoCfg()
        cfg_3 = InheritedNonTypeAnnotationOrderingDemoCfg()

        # check ordering
        self.assertEqual(list(cfg_1.__dict__.keys()), list(cfg_2.__dict__.keys()))
        self.assertEqual(list(cfg_3.__dict__.keys()), list(cfg_2.__dict__.keys()))
        self.assertEqual(list(cfg_1.__dict__.keys()), list(cfg_3.__dict__.keys()))

    def test_functions_config(self):
        """Tests having functions as values in the configuration instance."""
        cfg = FunctionsDemoCfg()
        # check types
        self.assertEqual(cfg.__annotations__["func"], type(dummy_function1))
        self.assertEqual(cfg.__annotations__["wrapped_func"], type(wrapped_dummy_function3))
        self.assertEqual(cfg.__annotations__["func_in_dict"], dict)
        # check calling
        self.assertEqual(cfg.func(), 1)
        self.assertEqual(cfg.wrapped_func(), 4)
        self.assertEqual(cfg.func_in_dict["func"](), 1)

    def test_function_impl_config(self):
        """Tests having function defined in the class instance."""
        cfg = FunctionImplementedDemoCfg()
        # change value
        self.assertEqual(cfg.a, 5)
        cfg.set_a(10)
        self.assertEqual(cfg.a, 10)

    def test_class_function_impl_config(self):
        """Tests having class function defined in the class instance."""
        cfg = ClassFunctionImplementedDemoCfg()

        # check that the annotations are correct
        self.assertDictEqual(cfg.__annotations__, {"a": "int"})

        # check all methods are callable
        cfg.instance_method()
        new_cfg1 = cfg.class_method(20)
        # check value is correct
        self.assertEqual(new_cfg1.a, 20)

        # create the same config instance using class method
        new_cfg2 = ClassFunctionImplementedDemoCfg.class_method(20)
        # check value is correct
        self.assertEqual(new_cfg2.a, 20)

    def test_class_property_impl_config(self):
        """Tests having class property defined in the class instance."""
        cfg = ClassFunctionImplementedDemoCfg()

        # check that the annotations are correct
        self.assertDictEqual(cfg.__annotations__, {"a": "int"})

        # check all methods are callable
        cfg.instance_method()

        # check value is correct
        self.assertEqual(cfg.a, 5)
        self.assertEqual(cfg.a_proxy, 5)

        # set through property
        cfg.a_proxy = 10
        self.assertEqual(cfg.a, 10)
        self.assertEqual(cfg.a_proxy, 10)

    def test_dict_conversion_functions_config(self):
        """Tests conversion of config with functions into dictionary."""
        cfg = FunctionsDemoCfg()
        cfg_dict = class_to_dict(cfg)
        self.assertEqual(cfg_dict["func"], functions_demo_cfg_correct["func"])
        self.assertEqual(cfg_dict["wrapped_func"], functions_demo_cfg_correct["wrapped_func"])
        self.assertEqual(cfg_dict["func_in_dict"]["func"], functions_demo_cfg_correct["func_in_dict"]["func"])

    def test_update_functions_config_with_functions(self):
        """Tests updating config with functions."""
        cfg = FunctionsDemoCfg()
        # update config
        update_class_from_dict(cfg, functions_demo_cfg_for_updating)
        # check calling
        self.assertEqual(cfg.func(), 2)
        self.assertEqual(cfg.wrapped_func(), 5)
        self.assertEqual(cfg.func_in_dict["func"](), 2)

    def test_missing_type_in_config(self):
        """Tests missing type annotation in config.

        Should complain that 'c' is missing type annotation since it cannot be inferred
        from 'MISSING' value.
        """
        with self.assertRaises(TypeError):

            @configclass
            class MissingTypeDemoCfg:
                a: int = 1
                b = 2
                c = MISSING

    def test_missing_default_value_in_config(self):
        """Tests missing default value in config.

        Should complain that 'a' is missing default value since it cannot be inferred
        from type annotation.
        """
        with self.assertRaises(ValueError):

            @configclass
            class MissingTypeDemoCfg:
                a: int
                b = 2

    def test_required_argument_for_missing_type_in_config(self):
        """Tests required positional argument for missing type annotation in config creation."""

        @configclass
        class MissingTypeDemoCfg:
            a: int = 1
            b = 2
            c: int = MISSING

        # should complain that 'c' is missed in positional arguments
        # TODO: Uncomment this when we move to 3.10.
        # with self.assertRaises(TypeError):
        #     cfg = MissingTypeDemoCfg(a=1)
        # should not complain
        cfg = MissingTypeDemoCfg(a=1, c=3)

        self.assertEqual(cfg.a, 1)
        self.assertEqual(cfg.b, 2)

    def test_config_inheritance(self):
        """Tests that inheritance works properly."""
        # check variables
        cfg_a = ChildADemoCfg(a=20, d=3, e=ViewerCfg(), j=["c", "d"])

        self.assertEqual(cfg_a.func, dummy_function1)
        self.assertEqual(cfg_a.a, 20)
        self.assertEqual(cfg_a.d, 3)
        self.assertEqual(cfg_a.j, ["c", "d"])

        # check post init
        self.assertEqual(cfg_a.b, 3)
        self.assertEqual(cfg_a.i, ["a", "b"])
        self.assertEqual(cfg_a.m.rot, (2.0, 0.0, 0.0, 0.0))

    def test_config_inheritance_independence(self):
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
        self.assertIsInstance(cfg_a.a, type(MISSING))
        self.assertEqual(cfg_b.a, 100)
        self.assertEqual(cfg_a.b, 3)
        self.assertEqual(cfg_b.b, 8)
        self.assertEqual(cfg_a.c, RobotDefaultStateCfg())
        self.assertIsInstance(cfg_b.c, type(MISSING))
        self.assertEqual(cfg_a.m.rot, (2.0, 0.0, 0.0, 0.0))
        self.assertEqual(cfg_b.m.rot, (1.0, 0.0, 0.0, 0.0))
        self.assertIsInstance(cfg_a.j, type(MISSING))
        self.assertEqual(cfg_b.j, ["3", "4"])
        self.assertEqual(cfg_a.i, ["a", "b"])
        self.assertEqual(cfg_b.i, ["1", "2"])
        self.assertEqual(cfg_a.func, dummy_function1)
        self.assertIsInstance(cfg_b.func, type(MISSING))

        # Explicitly assert that members are not the same object
        # for different levels and kinds of data types
        self.assertIsNot(cfg_a.m, cfg_b.m)
        self.assertIsNot(cfg_a.m.rot, cfg_b.m.rot)
        self.assertIsNot(cfg_a.i, cfg_b.i)
        self.assertIsNot(cfg_a.b, cfg_b.b)

    def test_config_double_inheritance(self):
        """Tests that inheritance works properly when inheriting twice."""
        # check variables
        cfg = ChildChildDemoCfg(a=20, d=3, e=ViewerCfg(), j=["c", "d"])

        self.assertEqual(cfg.func, dummy_function1)
        self.assertEqual(cfg.func_2, dummy_function2)
        self.assertEqual(cfg.a, 20)
        self.assertEqual(cfg.d, 3)
        self.assertEqual(cfg.j, ["c", "d"])

        # check post init
        self.assertEqual(cfg.b, 4)
        self.assertEqual(cfg.f, "new")
        self.assertEqual(cfg.i, ["a", "b"])

    def test_config_with_class_type(self):
        """Tests that configclass works properly with class type."""

        cfg = DummyClassCfg()

        # since python 3.10, annotations are stored as strings
        annotations = {k: eval(v) if isinstance(v, str) else v for k, v in cfg.__annotations__.items()}
        # check types
        self.assertEqual(annotations["class_name_1"], type)
        self.assertEqual(annotations["class_name_2"], type[DummyClass])
        self.assertEqual(annotations["class_name_3"], type[DummyClass])
        self.assertEqual(annotations["class_name_4"], ClassVar[type[DummyClass]])
        # check values
        self.assertEqual(cfg.class_name_1, DummyClass)
        self.assertEqual(cfg.class_name_2, DummyClass)
        self.assertEqual(cfg.class_name_3, DummyClass)
        self.assertEqual(cfg.class_name_4, DummyClass)
        self.assertEqual(cfg.b, "dummy")

    def test_nested_config_class_declarations(self):
        """Tests that configclass works properly with nested class class declarations."""

        cfg = OutsideClassCfg()

        # check types
        self.assertNotIn("InsideClassCfg", cfg.__annotations__)
        self.assertNotIn("InsideClassCfg", OutsideClassCfg.__annotations__)
        self.assertNotIn("InsideInsideClassCfg", OutsideClassCfg.InsideClassCfg.__annotations__)
        self.assertNotIn("InsideInsideClassCfg", cfg.inside.__annotations__)
        # check values
        self.assertEqual(cfg.inside.class_type, DummyClass)
        self.assertEqual(cfg.inside.b, "dummy_changed")
        self.assertEqual(cfg.x, 20)

    def test_config_dumping(self):
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
        self.assertEqual(list(cfg.to_dict().keys()), list(cfg_loaded.keys()))
        self.assertDictEqual(cfg.to_dict(), cfg_loaded)

        # save config with sorted order won't work!
        # save config
        dump_yaml(filename, cfg, sort_keys=True)
        # load config
        cfg_loaded = load_yaml(filename)
        # check dictionaries are the same
        self.assertNotEqual(list(cfg.to_dict().keys()), list(cfg_loaded.keys()))
        self.assertDictEqual(cfg.to_dict(), cfg_loaded)

    def test_config_md5_hash(self):
        """Check that config md5 hash generation works properly."""

        # create config
        cfg = ChildADemoCfg(a=20, d=3, e=ViewerCfg(), j=["c", "d"])

        # generate md5 hash
        md5_hash_1 = dict_to_md5_hash(cfg.to_dict())
        md5_hash_2 = dict_to_md5_hash(cfg.to_dict())

        self.assertEqual(md5_hash_1, md5_hash_2)

    def test_validity(self):
        """Check that invalid configurations raise errors."""

        cfg = MissingChildDemoCfg()

        with self.assertRaises(TypeError) as context:
            cfg.validate()

        # check that the expected missing fields are in the error message
        error_message = str(context.exception)
        for elem in validity_expected_fields:
            self.assertIn(elem, error_message)

        # check that no more than the expected missing fields are in the error message
        self.assertEqual(len(error_message.split("\n")) - 2, len(validity_expected_fields))


if __name__ == "__main__":
    run_tests()
