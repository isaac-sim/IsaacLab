# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Wrapper around the Python 3.7 onwards `dataclasses` module."""


from copy import deepcopy
from dataclasses import Field, dataclass, field
from typing import Any, Callable, ClassVar, Dict

from .dict import class_to_dict, update_class_from_dict

# List of all methods provided by sub-module.
__all__ = ["configclass"]


"""
Wrapper around dataclass.
"""


def __dataclass_transform__():
    """Add annotations decorator for PyLance."""
    return lambda a: a


@__dataclass_transform__()
def configclass(cls, **kwargs):
    """Wrapper around `dataclass` functionality to add extra checks and utilities.

    As of Python3.8, the standard dataclasses have two main issues which makes them non-generic for configuration use-cases.
    These include:

    1. Requiring a type annotation for all its members.
    2. Requiring explicit usage of :meth:`field(default_factory=...)` to reinitialize mutable variables.

    This function wraps around :class:`dataclass` utility to deal with the above two issues.

    Usage:
        .. code-block:: python

            from dataclasses import MISSING

            from omni.isaac.orbit.utils.configclass import configclass


            @configclass
            class ViewerCfg:
                eye: list = [7.5, 7.5, 7.5]  # field missing on purpose
                lookat: list = field(default_factory=[0.0, 0.0, 0.0])


            @configclass
            class EnvCfg:
                num_envs: int = MISSING
                episode_length: int = 2000
                viewer: ViewerCfg = ViewerCfg()

            # create configuration instance
            env_cfg = EnvCfg(num_envs=24)
            # print information
            print(env_cfg.to_dict())

    Reference:
        https://docs.python.org/3/library/dataclasses.html#dataclasses.Field
    """
    # add type annotations
    _add_annotation_types(cls)
    # add field factory
    _process_mutable_types(cls)
    # copy mutable members
    setattr(cls, "__post_init__", _custom_post_init)
    # add helper functions for dictionary conversion
    setattr(cls, "to_dict", _class_to_dict)
    setattr(cls, "from_dict", _update_class_from_dict)
    # wrap around dataclass
    cls = dataclass(cls, **kwargs)
    # return wrapped class
    return cls


"""
Dictionary <-> Class operations.

These are redefined here to add new docstrings.
"""


def _class_to_dict(obj: object) -> Dict[str, Any]:
    """Convert an object into dictionary recursively.

    Returns:
        Dict[str, Any]: Converted dictionary mapping.
    """
    return class_to_dict(obj)


def _update_class_from_dict(obj, data: Dict[str, Any]) -> None:
    """Reads a dictionary and sets object variables recursively.

    This function performs in-place update of the class member attributes.

    Args:
        data (Dict[str, Any]): Input (nested) dictionary to update from.

    Raises:
        TypeError: When input is not a dictionary.
        ValueError: When dictionary has a value that does not match default config type.
        KeyError: When dictionary has a key that does not exist in the default config type.
    """
    return update_class_from_dict(obj, data, _ns="")


"""
Private helper functions.
"""


def _add_annotation_types(cls):
    """Add annotations to all elements in the dataclass.

    By definition in Python, a field is defined as a class variable that has a type annotation.

    In case type annotations are not provided, dataclass ignores those members when :func:`__dict__()` is called.
    This function adds these annotations to the class variable to prevent any issues in case the user forgets to
    specify the type annotation.

    This makes the following a feasible operation:

    @dataclass
    class State:
        pos = (0.0, 0.0, 0.0)
           ^^
           If the function is NOT used, the following type-error is returned:
           TypeError: 'pos' is a field but has no type annotation
    """
    # Note: Do not change this line. `cls.__dict__.get("__annotations__", {})` is different from `cls.__annotations__` because of inheritance.
    cls.__annotations__ = cls.__dict__.get("__annotations__", {})
    # cls.__annotations__ = dict()
    for key in dir(cls):
        # skip dunder members
        if key.startswith("__"):
            continue
        # skip class functions
        if key in ["from_dict", "to_dict"]:
            continue
        # add type annotations for members that are not functions
        var = getattr(cls, key)
        if not isinstance(var, type):
            if key not in cls.__annotations__:
                cls.__annotations__[key] = type(var)


def _process_mutable_types(cls):
    """Initialize all mutable elements through :obj:`dataclasses.Field` to avoid unnecessary complaints.

    By default, dataclass requires usage of :obj:`field(default_factory=...)` to reinitialize mutable objects every time a new
    class instance is created. If a member has a mutable type and it is created without specifying the `field(default_factory=...)`,
    then Python throws an error requiring the usage of `default_factory`.

    Additionally, Python only explicitly checks for field specification when the type is a list, set or dict. This misses the
    use-case where the type is class itself. Thus, the code silently carries a bug with it which can lead to undesirable effects.

    This function deals with this issue

    This makes the following a feasible operation:

    @dataclass
    class State:
        pos: list = [0.0, 0.0, 0.0]
           ^^
           If the function is NOT used, the following value-error is returned:
           ValueError: mutable default <class 'list'> for field pos is not allowed: use default_factory
    """

    def _return_f(f: Any) -> Callable[[], Any]:
        """Returns default function for creating mutable/immutable variables."""

        def _wrap():
            if isinstance(f, Field):
                return f.default_factory
            else:
                return f

        return _wrap

    for key in dir(cls):
        # skip dunder members
        if key.startswith("__"):
            continue
        # skip class functions
        if key in ["from_dict", "to_dict"]:
            continue
        # do not create field for class variables
        if key in cls.__annotations__:
            origin = getattr(cls.__annotations__[key], "__origin__", None)
            if origin is ClassVar:
                continue
        # define explicit field for data members
        f = getattr(cls, key)
        if not isinstance(f, type):
            f = field(default_factory=_return_f(f))
            setattr(cls, key, f)


def _custom_post_init(obj):
    """Deepcopy all elements to avoid shared memory issues for mutable objects in dataclasses initialization.

    This function is called explicitly instead of as a part of :func:`_process_mutable_types()` to prevent mapping
    proxy type i.e. a read only proxy for mapping objects. The error is thrown when using hierarchical data-classes
    for configuration.
    """
    for key in dir(obj):
        # skip dunder members
        if key.startswith("__"):
            continue
        # duplicate data members
        var = getattr(obj, key)
        if not callable(var):
            setattr(obj, key, deepcopy(var))
