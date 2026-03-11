# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module that provides a wrapper around the Python 3.7 onwards ``dataclasses`` module."""

import inspect
import re
import types
from collections.abc import Callable
from copy import deepcopy
from dataclasses import MISSING, Field, dataclass, field, replace
from typing import Any, ClassVar

from .dict import class_to_dict, update_class_from_dict
from .string import ResolvableString

_CALLABLE_STR_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_\\.]*:[A-Za-z_][A-Za-z0-9_]*$")
_CALLABLE_STR_WITH_DIR_RE = re.compile(r"^\{DIR\}(?:\.[A-Za-z_][A-Za-z0-9_]*)*:[A-Za-z_][A-Za-z0-9_]*$")

_CONFIGCLASS_METHODS = ["to_dict", "from_dict", "replace", "copy", "validate"]
"""List of class methods added at runtime to dataclass."""

"""
Wrapper around dataclass.
"""


def __dataclass_transform__():
    """Add annotations decorator for PyLance."""
    return lambda a: a


@__dataclass_transform__()
def configclass(cls, **kwargs):
    """Wrapper around `dataclass` functionality to add extra checks and utilities.

    As of Python 3.7, the standard dataclasses have two main issues which makes them non-generic for
    configuration use-cases. These include:

    1. Requiring a type annotation for all its members.
    2. Requiring explicit usage of :meth:`field(default_factory=...)` to reinitialize mutable variables.

    This function provides a decorator that wraps around Python's `dataclass`_ utility to deal with
    the above two issues. It also provides additional helper functions for dictionary <-> class
    conversion and easily copying class instances.

    Usage:

    .. code-block:: python

        from dataclasses import MISSING

        from isaaclab.utils.configclass import configclass


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

        # print information as a dictionary
        print(env_cfg.to_dict())

        # create a copy of the configuration
        env_cfg_copy = env_cfg.copy()

        # replace arbitrary fields using keyword arguments
        env_cfg_copy = env_cfg_copy.replace(num_envs=32)

    Args:
        cls: The class to wrap around.
        **kwargs: Additional arguments to pass to :func:`dataclass`.

    Returns:
        The wrapped class.

    .. _dataclass: https://docs.python.org/3/library/dataclasses.html
    """
    # snapshot field names declared in *this* class body before configclass
    # merges parent fields — used by _field_module_dir to resolve {DIR} correctly.
    _own_ann = set(cls.__dict__.get("__annotations__", {}).keys())
    _own_body = {k for k in cls.__dict__ if not k.startswith("__")}
    cls.__configclass_own_fields__ = frozenset(_own_ann | _own_body)
    # add type annotations
    _add_annotation_types(cls)
    # add field factory
    _process_mutable_types(cls)
    # copy mutable members
    # note: we check if user defined __post_init__ function exists and augment it with our own
    if hasattr(cls, "__post_init__"):
        setattr(cls, "__post_init__", _combined_function(cls.__post_init__, _custom_post_init))
    else:
        setattr(cls, "__post_init__", _custom_post_init)
    # add helper functions for dictionary conversion
    setattr(cls, "to_dict", _class_to_dict)
    setattr(cls, "from_dict", _update_class_from_dict)
    setattr(cls, "replace", _replace_class_with_kwargs)
    setattr(cls, "copy", _copy_class)
    setattr(cls, "validate", _validate)
    # wrap around dataclass
    cls = dataclass(cls, **kwargs)
    # return wrapped class
    return cls


"""
Dictionary <-> Class operations.

These are redefined here to add new docstrings.
"""


def _class_to_dict(obj: object) -> dict[str, Any]:
    """Convert an object into dictionary recursively.

    Args:
        obj: The object to convert.

    Returns:
        Converted dictionary mapping.
    """
    return class_to_dict(obj)


def _update_class_from_dict(obj, data: dict[str, Any]) -> None:
    """Reads a dictionary and sets object variables recursively.

    This function performs in-place update of the class member attributes.

    Args:
        obj: The object to update.
        data: Input (nested) dictionary to update from.

    Raises:
        TypeError: When input is not a dictionary.
        ValueError: When dictionary has a value that does not match default config type.
        KeyError: When dictionary has a key that does not exist in the default config type.
    """
    update_class_from_dict(obj, data, _ns="")


def _replace_class_with_kwargs(obj: object, **kwargs) -> object:
    """Return a new object replacing specified fields with new values.

    This is especially useful for frozen classes.  Example usage:

    .. code-block:: python

        @configclass(frozen=True)
        class C:
            x: int
            y: int


        c = C(1, 2)
        c1 = c.replace(x=3)
        assert c1.x == 3 and c1.y == 2

    Args:
        obj: The object to replace.
        **kwargs: The fields to replace and their new values.

    Returns:
        The new object.
    """
    return replace(obj, **kwargs)


def _copy_class(obj: object) -> object:
    """Return a new object with the same fields as the original."""
    return replace(obj)


def _field_module_dir(obj: Any, key: str | None = None) -> str | None:
    """Return module parent package path for an object or one of its declared fields."""
    cls = type(obj)
    if key is not None:
        # Use nearest declaration in MRO (subclass override wins).
        # We prefer __configclass_own_fields__ (the snapshot taken before
        # _process_mutable_types copies parent fields into every subclass's
        # __dict__) so that {DIR} resolves relative to the class that
        # *originally* declared the field, not the subclass that inherited it.
        for mro_cls in cls.__mro__:
            if mro_cls is object:
                continue
            own_fields = getattr(mro_cls, "__configclass_own_fields__", None)
            if own_fields is not None:
                if key in own_fields:
                    cls = mro_cls
                    break
            elif key in mro_cls.__dict__:
                cls = mro_cls
                break
    module_name = getattr(cls, "__module__", "")
    return module_name.rsplit(".", 1)[0] if "." in module_name else (module_name or None)


def _wrap_resolvable_strings(value: Any, module_dir: str | None = None, _seen: set[int] | None = None) -> Any:
    """Recursively wrap callable-like strings with :class:`ResolvableString`."""
    if isinstance(value, str) and (_CALLABLE_STR_RE.match(value) or _CALLABLE_STR_WITH_DIR_RE.match(value)):
        if "{DIR}" in value:
            if module_dir is None:
                raise ValueError(f"Cannot resolve '{{DIR}}' in '{value}' because no module context is available.")
            value = value.replace("{DIR}", module_dir)
        return ResolvableString(value)
    is_dataclass_instance = hasattr(value, "__dataclass_fields__") and hasattr(value, "__dict__")
    is_container = isinstance(value, (list, tuple, dict))
    if is_dataclass_instance or is_container:
        if _seen is None:
            _seen = set()
        value_id = id(value)
        if value_id in _seen:
            return value
        _seen.add(value_id)
    if isinstance(value, list):
        wrapped = [_wrap_resolvable_strings(item, module_dir=module_dir, _seen=_seen) for item in value]
        if len(wrapped) == len(value) and all(new_item is old_item for new_item, old_item in zip(wrapped, value)):
            return value
        return wrapped
    if isinstance(value, tuple):
        wrapped = tuple(_wrap_resolvable_strings(item, module_dir=module_dir, _seen=_seen) for item in value)
        if len(wrapped) == len(value) and all(new_item is old_item for new_item, old_item in zip(wrapped, value)):
            return value
        return wrapped
    if isinstance(value, dict):
        wrapped = {
            key: _wrap_resolvable_strings(item, module_dir=module_dir, _seen=_seen) for key, item in value.items()
        }
        if len(wrapped) == len(value) and all(wrapped[key] is value[key] for key in value):
            return value
        return wrapped
    if is_dataclass_instance:
        for key, item in value.__dict__.items():
            nested_module_dir = _field_module_dir(value, key)
            setattr(value, key, _wrap_resolvable_strings(item, module_dir=nested_module_dir, _seen=_seen))
    return value


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
    # get type hints
    hints = {}
    # iterate over class inheritance
    # we add annotations from base classes first
    for base in reversed(cls.__mro__):
        # check if base is object
        if base is object:
            continue
        # get base class annotations
        ann = base.__dict__.get("__annotations__", {})
        # directly add all annotations from base class
        hints.update(ann)
        # iterate over base class members
        # Note: Do not change this to dir(base) since it orders the members alphabetically.
        #   This is not desirable since the order of the members is important in some cases.
        for key in base.__dict__:
            # get class member
            value = getattr(base, key)
            # skip members
            if _skippable_class_member(key, value, hints):
                continue
            # add type annotations for members that don't have explicit type annotations
            # for these, we deduce the type from the default value
            if not isinstance(value, type):
                if key not in hints:
                    # check if var type is not MISSING
                    # we cannot deduce type from MISSING!
                    if value is MISSING:
                        raise TypeError(
                            f"Missing type annotation for '{key}' in class '{cls.__name__}'."
                            " Please add a type annotation or set a default value."
                        )
                    # add type annotation
                    hints[key] = type(value)
            elif key != value.__name__:
                # note: we don't want to add type annotations for nested configclass. Thus, we check if
                #   the name of the type matches the name of the variable.
                # since Python 3.10, type hints are stored as strings
                hints[key] = f"type[{value.__name__}]"

    # Note: Do not change this line. `cls.__dict__.get("__annotations__", {})` is different from
    #   `cls.__annotations__` because of inheritance.
    cls.__annotations__ = cls.__dict__.get("__annotations__", {})
    cls.__annotations__ = hints


def _validate(obj: object, prefix: str = "") -> list[str]:
    """Check the validity of configclass object.

    This function checks if the object is a valid configclass object. A valid configclass object contains no MISSING
    entries. Additionally, if the top-level object defines a ``_validate_config`` method, it is called to perform
    domain-specific validation.

    Subclasses can define ``validate_config(self)`` to add custom checks::

        @configclass
        class MyEnvCfg(ManagerBasedEnvCfg):
            def validate_config(self):
                if self.some_field == "bad":
                    raise ValueError("some_field cannot be 'bad'.")

    Args:
        obj: The object to check.
        prefix: The prefix to add to the missing fields. Defaults to ''.

    Returns:
        A list of missing fields.

    Raises:
        TypeError: When the object is not a valid configuration object.
    """
    missing_fields = []

    if type(obj).__name__ == "MeshConverterCfg":
        return missing_fields

    if type(obj) is type(MISSING):
        missing_fields.append(prefix)
        return missing_fields
    elif isinstance(obj, (list, tuple)):
        for index, item in enumerate(obj):
            current_path = f"{prefix}[{index}]"
            missing_fields.extend(_validate(item, prefix=current_path))
        return missing_fields
    elif isinstance(obj, dict):
        # Convert any non-string keys to strings to allow validation of dict with non-string keys
        if any(not isinstance(key, str) for key in obj.keys()):
            obj_dict = {str(key): value for key, value in obj.items()}
        else:
            obj_dict = obj
    elif hasattr(obj, "__dict__"):
        obj_dict = obj.__dict__
    else:
        return missing_fields

    for key, value in obj_dict.items():
        # disregard builtin attributes
        if key.startswith("__"):
            continue
        current_path = f"{prefix}.{key}" if prefix else key
        missing_fields.extend(_validate(value, prefix=current_path))

    # raise an error only once at the top-level call
    if prefix == "" and missing_fields:
        formatted_message = "\n".join(f"  - {field}" for field in missing_fields)
        raise TypeError(
            f"Missing values detected in object {obj.__class__.__name__} for the following"
            f" fields:\n{formatted_message}\n"
        )
    # invoke custom validation hook if defined on the object
    if prefix == "":
        custom_validate = getattr(obj, "validate_config", None)
        if callable(custom_validate):
            custom_validate()
    return missing_fields


def _process_mutable_types(cls):
    """Initialize all mutable elements through :obj:`dataclasses.Field` to avoid unnecessary complaints.

    By default, dataclass requires usage of :obj:`field(default_factory=...)` to reinitialize mutable objects
    every time a new class instance is created. If a member has a mutable type and it is created without
    specifying the `field(default_factory=...)`, then Python throws an error requiring the usage of `default_factory`.

    Additionally, Python only explicitly checks for field specification when the type is a list, set or dict.
    This misses the use-case where the type is class itself. Thus, the code silently carries a bug with it which
    can lead to undesirable effects.

    This function deals with this issue

    This makes the following a feasible operation:

    @dataclass
    class State:
        pos: list = [0.0, 0.0, 0.0]
           ^^
           If the function is NOT used, the following value-error is returned:
           ValueError: mutable default <class 'list'> for field pos is not allowed: use default_factory
    """
    # note: Need to set this up in the same order as annotations. Otherwise, it
    #   complains about missing positional arguments.
    ann = cls.__dict__.get("__annotations__", {})

    # iterate over all class members and store them in a dictionary
    class_members = {}
    for base in reversed(cls.__mro__):
        # check if base is object
        if base is object:
            continue
        # iterate over base class members
        for key in base.__dict__:
            # get class member
            f = getattr(base, key)
            # skip members
            if _skippable_class_member(key, f):
                continue
            # store class member if it is not a type or if it is already present in annotations
            if not isinstance(f, type) or key in ann:
                class_members[key] = f
        # iterate over base class data fields
        # in previous call, things that became a dataclass field were removed from class members
        # so we need to add them back here as a dataclass field directly
        for key, f in base.__dict__.get("__dataclass_fields__", {}).items():
            # store class member
            if not isinstance(f, type):
                class_members[key] = f

    # check that all annotations are present in class members
    # note: mainly for debugging purposes
    if len(class_members) != len(ann):
        raise ValueError(
            f"In class '{cls.__name__}', number of annotations ({len(ann)}) does not match number of class members"
            f" ({len(class_members)}). Please check that all class members have type annotations and/or a default"
            " value. If you don't want to specify a default value, please use the literal `dataclasses.MISSING`."
        )
    # iterate over annotations and add field factory for mutable types
    for key in ann:
        # find matching field in class
        value = class_members.get(key, MISSING)
        # check if key belongs to ClassVar
        # in that case, we cannot use default_factory!
        origin = getattr(ann[key], "__origin__", None)
        if origin is ClassVar:
            continue
        # check if f is MISSING
        # note: commented out for now since it causes issue with inheritance
        #   of dataclasses when parent have some positional and some keyword arguments.
        # Ref: https://stackoverflow.com/questions/51575931/class-inheritance-in-python-3-7-dataclasses
        # TODO: check if this is fixed in Python 3.10
        # if f is MISSING:
        #     continue
        if isinstance(value, Field):
            setattr(cls, key, value)
        elif not isinstance(value, type):
            # create field factory for mutable types
            value = field(default_factory=_return_f(value))
            setattr(cls, key, value)


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
        # get data member
        value = getattr(obj, key)
        # check annotation
        ann = obj.__class__.__dict__.get(key)
        # duplicate data members that are mutable
        if not callable(value) and not isinstance(ann, property):
            copied_value = deepcopy(value)
            setattr(obj, key, _wrap_resolvable_strings(copied_value, module_dir=_field_module_dir(obj, key)))


def _combined_function(f1: Callable, f2: Callable) -> Callable:
    """Combine two functions into one.

    Args:
        f1: The first function.
        f2: The second function.

    Returns:
        The combined function.
    """

    def _combined(*args, **kwargs):
        # call both functions
        f1(*args, **kwargs)
        f2(*args, **kwargs)

    return _combined


"""
Helper functions
"""


def _skippable_class_member(key: str, value: Any, hints: dict | None = None) -> bool:
    """Check if the class member should be skipped in configclass processing.

    The following members are skipped:

    * Dunder members: ``__name__``, ``__module__``, ``__qualname__``, ``__annotations__``, ``__dict__``.
    * Manually-added special class functions: From :obj:`_CONFIGCLASS_METHODS`.
    * Members that are already present in the type annotations.
    * Functions bounded to class object or class.
    * Properties bounded to class object.

    Args:
        key: The class member name.
        value: The class member value.
        hints: The type hints for the class. Defaults to None, in which case, the
            members existence in type hints are not checked.

    Returns:
        True if the class member should be skipped, False otherwise.
    """
    # skip dunder members
    if key.startswith("__"):
        return True
    # skip manually-added special class functions
    if key in _CONFIGCLASS_METHODS:
        return True
    # check if key is already present
    if hints is not None and key in hints:
        return True
    # skip functions bounded to class
    if callable(value):
        # FIXME: This doesn't yet work for static methods because they are essentially seen as function types.
        # check for class methods
        if isinstance(value, types.MethodType):
            return True

        if "CollisionAPI" in value.__name__:
            return False

        # check for instance methods
        signature = inspect.signature(value)
        if "self" in signature.parameters or "cls" in signature.parameters:
            return True

    # skip property methods
    if isinstance(value, property):
        return True
    # Otherwise, don't skip
    return False


def _return_f(f: Any) -> Callable[[], Any]:
    """Returns default factory function for creating mutable/immutable variables.

    This function should be used to create default factory functions for variables.

    Example:

        .. code-block:: python

            value = field(default_factory=_return_f(value))
            setattr(cls, key, value)
    """

    def _wrap():
        if isinstance(f, Field):
            if f.default_factory is MISSING:
                return deepcopy(f.default)
            else:
                return f.default_factory
        else:
            return deepcopy(f)

    return _wrap


def resolve_cfg_presets(cfg: object) -> object:
    """Recursively replace preset-wrapper fields with their *default* preset.

    Task configs may use two preset-selector patterns to support multiple physics backends
    (PhysX / Newton) or observation modes. Both patterns produce wrapper objects that are
    **not** valid as the concrete cfg that downstream managers / scene builders expect.
    This function resolves them in-place so the config can be used without a Hydra CLI
    override (e.g. in unit tests or when creating environments directly).

    Supported patterns:

    * **New style** (``PresetCfg`` subclass): a configclass whose MRO contains a class named
      ``PresetCfg``. The active variant is stored in the ``default`` attribute.
    * **Old style** (``presets`` dict): a configclass that has a ``presets: dict[str, Cfg]``
      attribute with a ``"default"`` key.

    Args:
        cfg: Any configclass instance (or any object; non-configclasses are returned as-is).

    Returns:
        The same ``cfg`` object, modified in-place with preset wrappers replaced.
    """
    if not hasattr(cfg, "__dataclass_fields__"):
        return cfg
    for field_name in list(cfg.__dataclass_fields__):
        value = getattr(cfg, field_name, None)
        if value is None or not hasattr(value, "__dataclass_fields__"):
            continue
        # New-style PresetCfg: class hierarchy contains a class named "PresetCfg".
        if any(cls.__name__ == "PresetCfg" for cls in type(value).__mro__):
            resolved = value.default
            setattr(cfg, field_name, resolved)
            resolve_cfg_presets(resolved)
        # Old-style preset: configclass with a ``presets`` dict that has a ``"default"`` key.
        elif isinstance(getattr(value, "presets", None), dict) and "default" in value.presets:
            resolved = value.presets["default"]
            setattr(cfg, field_name, resolved)
            resolve_cfg_presets(resolved)
        else:
            resolve_cfg_presets(value)
    return cfg
