# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities for working with dictionaries."""


import collections.abc
import importlib
import inspect
from typing import Any, Callable, Dict, Iterable, Mapping

from .array import TENSOR_TYPE_CONVERSIONS, TENSOR_TYPES

__all__ = ["class_to_dict", "update_class_from_dict", "convert_dict_to_backend", "update_dict", "print_dict"]

"""
Dictionary <-> Class operations.
"""


def class_to_dict(obj: object) -> Dict[str, Any]:
    """Convert an object into dictionary recursively.

    Note:
        Ignores all names starting with "__" (i.e. built-in methods).

    Args:
        obj (object): An instance of a class to convert.

    Raises:
        ValueError: When input argument is not an object.

    Returns:
        Dict[str, Any]: Converted dictionary mapping.
    """
    # check that input data is class instance
    if not hasattr(obj, "__class__"):
        raise ValueError(f"Expected a class instance. Received: {type(obj)}.")
    # convert object to dictionary
    if isinstance(obj, dict):
        obj_dict = obj
    else:
        obj_dict = obj.__dict__
    # convert to dictionary
    data = dict()
    for key, value in obj_dict.items():
        # disregard builtin attributes
        if key.startswith("__"):
            continue
        # check if attribute is callable -- function
        if callable(value):
            data[key] = f"{value.__module__}:{value.__name__}"
        # check if attribute is a dictionary
        elif hasattr(value, "__dict__") or isinstance(value, dict):
            data[key] = class_to_dict(value)
        else:
            data[key] = value
    return data


def update_class_from_dict(obj, data: Dict[str, Any], _ns: str = "") -> None:
    """Reads a dictionary and sets object variables recursively.

    This function performs in-place update of the class member attributes.

    Args:
        obj (object): An instance of a class to update.
        data (Dict[str, Any]): Input dictionary to update from.
        _ns (str): Namespace of the current object. This is useful for nested configuration
            classes or dictionaries. Defaults to "".

    Raises:
        TypeError: When input is not a dictionary.
        ValueError: When dictionary has a value that does not match default config type.
        KeyError: When dictionary has a key that does not exist in the default config type.
    """
    for key, value in data.items():
        # key_ns is the full namespace of the key
        key_ns = _ns + "/" + key
        # check if key is present in the object
        if hasattr(obj, key):
            obj_mem = getattr(obj, key)
            if isinstance(obj_mem, Mapping):
                # Note: We don't handle two-level nested dictionaries. Just use configclass if this is needed.
                # iterate over the dictionary to look for callable values
                for k, v in obj_mem.items():
                    if callable(v):
                        value[k] = _string_to_callable(value[k])
                setattr(obj, key, value)
            elif isinstance(value, Mapping):
                # recursively call if it is a dictionary
                update_class_from_dict(obj_mem, value, _ns=key_ns)
            elif isinstance(value, Iterable) and not isinstance(value, str):
                # check length of value to be safe
                if len(obj_mem) != len(value) and obj_mem is not None:
                    raise ValueError(
                        f"[Config]: Incorrect length under namespace: {key_ns}. Expected: {len(obj_mem)}, Received: {len(value)}."
                    )
                # set value
                setattr(obj, key, value)
            elif callable(obj_mem):
                # update function name
                value = _string_to_callable(value)
                setattr(obj, key, value)
            elif isinstance(value, type(obj_mem)):
                # check that they are type-safe
                setattr(obj, key, value)
            else:
                raise ValueError(
                    f"[Config]: Incorrect type under namespace: {key_ns}. Expected: {type(obj_mem)}, Received: {type(value)}."
                )
        else:
            raise KeyError(f"[Config]: Key not found under namespace: {key_ns}.")


"""
Dictionary operations.
"""


def convert_dict_to_backend(
    data: dict, backend: str = "numpy", array_types: Iterable[str] = ("numpy", "torch", "warp")
) -> dict:
    """Convert all arrays or tensors in a dictionary to a given backend.

    This function iterates over the dictionary, converts all arrays or tensors with the given types to
    the desired backend, and stores them in a new dictionary. It also works with nested dictionaries.

    Currently supported backends are "numpy", "torch", and "warp".

    Note:
        This function only converts arrays or tensors. Other types of data are left unchanged. Mutable types
        (e.g. lists) are referenced by the new dictionary, so they are not copied.

    Args:
        data (dict): An input dict containing array or tensor data as values.
        backend(str): The backend ("numpy", "torch", "warp") to which arrays in this dict should be converted.
            Defaults to "numpy".
        array_types(Iterable[str]): A list containing the types of arrays that should be converted to
            the desired backend. Defaults to ("numpy", "torch", "warp").

    Raises:
        ValueError: If the specified ``backend`` or ``array_types`` are unknown, i.e. not in the list of supported
            backends ("numpy", "torch", "warp").

    Returns:
        dict: The updated dict with the data converted to the desired backend.
    """
    # THINK: Should we also support converting to a specific device, e.g. "cuda:0"?
    # Check the backend is valid.
    if backend not in TENSOR_TYPE_CONVERSIONS:
        raise ValueError(f"Unknown backend '{backend}'. Supported backends are 'numpy', 'torch', and 'warp'.")
    # Define the conversion functions for each backend.
    tensor_type_conversions = TENSOR_TYPE_CONVERSIONS[backend]

    # Parse the array types and convert them to the corresponding types: "numpy" -> np.ndarray, etc.
    parsed_types = list()
    for t in array_types:
        # Check type is valid.
        if t not in TENSOR_TYPES:
            raise ValueError(f"Unknown array type: '{t}'. Supported array types are 'numpy', 'torch', and 'warp'.")
        # Exclude types that match the backend, since we do not need to convert these.
        if t == backend:
            continue
        # Convert the string types to the corresponding types.
        parsed_types.append(TENSOR_TYPES[t])

    # Convert the data to the desired backend.
    output_dict = dict()
    for key, value in data.items():
        # Obtain the data type of the current value.
        data_type = type(value)
        # -- arrays
        if data_type in parsed_types:
            # check if we have a known conversion.
            if data_type not in tensor_type_conversions:
                raise ValueError(f"No registered conversion for data type: {data_type} to {backend}!")
            # convert the data to the desired backend.
            output_dict[key] = tensor_type_conversions[data_type](value)
        # -- nested dictionaries
        elif isinstance(data[key], dict):
            output_dict[key] = convert_dict_to_backend(value)
        # -- everything else
        else:
            output_dict[key] = value

    return output_dict


def update_dict(orig_dict: dict, new_dict: collections.abc.Mapping) -> dict:
    """Updates existing dictionary with values from a new dictionary.

    This function mimics the dict.update() function. However, it works for
    nested dictionaries as well.

    Reference:
        https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth

    Args:
        orig_dict (dict): The original dictionary to insert items to.
        new_dict (collections.abc.Mapping): The new dictionary to insert items from.

    Returns:
        dict: The updated dictionary.
    """
    for keyname, value in new_dict.items():
        if isinstance(value, collections.abc.Mapping):
            orig_dict[keyname] = update_dict(orig_dict.get(keyname, {}), value)
        else:
            orig_dict[keyname] = value
    return orig_dict


def print_dict(val, nesting: int = -4, start: bool = True):
    """Outputs a nested dictionary."""
    if type(val) == dict:
        if not start:
            print("")
        nesting += 4
        for k in val:
            print(nesting * " ", end="")
            print(k, end=": ")
            print_dict(val[k], nesting, start=False)
    else:
        # deal with functions in print statements
        if callable(val) and val.__name__ == "<lambda>":
            print("lambda", inspect.getsourcelines(val)[0][0].strip().split("lambda")[1].strip()[:-1])
        elif callable(val):
            print(f"{val.__module__}:{val.__name__}")
        else:
            print(val)


"""
Private helper functions.
"""


def _string_to_callable(name: str) -> Callable:
    """Resolves the module and function names to return the function.

    Args:
        name (str): The function name. The format should be 'module:attribute_name'.

    Raises:
        ValueError: When the resolved attribute is not a function.
        ValueError: _description_

    Returns:
        Callable: The function loaded from the module.
    """
    try:
        mod_name, attr_name = name.split(":")
        mod = importlib.import_module(mod_name)
        callable_object = getattr(mod, attr_name)
        # check if attribute is callable
        if callable(callable_object):
            return callable_object
        else:
            raise ValueError(f"The imported object is not callable: '{name}'")
    except AttributeError as e:
        msg = (
            "While updating the config from a dictionary, we could not interpret the entry"
            "as a callable object. The format of input should be 'module:attribute_name'\n"
            f"While processing input '{name}', received the error:\n {e}."
        )
        raise ValueError(msg)
