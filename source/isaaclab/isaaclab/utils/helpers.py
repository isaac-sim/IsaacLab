# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utility functions for functions."""

import functools
import inspect
import torch
from collections.abc import Callable

import omni.log
import warp as wp


def deprecated(
    *replacement_function_names: str,
    message: str = "",
    since: str | None = None,
    remove_in: str | None = None,
):
    """A decorator to mark functions as deprecated.

    It will result in a warning being emitted when the function is used.

    Args:
        replacement_function_names: The names of the functions to use instead.
        message: A custom message to append to the warning.
        since: The version in which the function was deprecated.
        remove_in: The version in which the function will be removed.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Form deprecation message.
            deprecation_message = f"Call to deprecated function '{func.__name__}'."
            # Add version information if provided.
            if since:
                deprecation_message += f" It was deprecated in version {since}."
            if remove_in:
                deprecation_message += f" It will be removed in version {remove_in}."
            else:
                deprecation_message += " It will be removed in a future version."
            # Add replacement function information if provided.
            if replacement_function_names:
                deprecation_message += f" Use {', '.join(replacement_function_names)} instead."
            # Add custom message if provided.
            if message:
                deprecation_message += f" {message}"

            # Emit warning.
            omni.log.warn(
                deprecation_message,
            )
            # Call the original function.
            return func(*args, **kwargs)

        return wrapper

    return decorator


def warn_overhead_cost(
    *replacement_function_names: str,
    message: str = "",
):
    """A decorator to mark functions as having a high overhead cost.

    It will result in a warning being emitted when the function is used.

    Args:
        replacement_function_names: The names of the functions to use instead.
        message: A custom message to append to the warning.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Form deprecation message.
            warning_message = f"Call to '{func.__name__}' which is a high overhead operation."
            # Add replacement function information if provided.
            if replacement_function_names:
                warning_message += f" Use {', '.join(replacement_function_names)} instead."
            # Add custom message if provided.
            if message:
                warning_message += f" {message}"

            # Emit warning.
            omni.log.warn(
                warning_message,
            )
            # Call the original function.
            return func(*args, **kwargs)

        return wrapper

    return decorator


def _analyze_and_convert_args(self, func, *args, **kwargs):
    """A helper to analyze and convert arguments from PyTorch to Warp."""
    sig = inspect.signature(func)
    bound_args = sig.bind(self, *args, **kwargs)
    bound_args.apply_defaults()
    arguments = bound_args.arguments

    # -- Device conversion
    device = getattr(self, "device", "cpu")

    # -- Tensor conversion
    spec = getattr(func, "_torch_frontend_spec", {})
    tensor_args = spec.get("tensor_args", {})
    first_torch_tensor = None
    for arg_name in tensor_args:
        arg_value = arguments.get(arg_name)
        if isinstance(arg_value, torch.Tensor):
            if first_torch_tensor is None:
                first_torch_tensor = arg_value
            tensor = arguments[arg_name]
            dtype = tensor_args[arg_name]
            arguments[arg_name] = wp.from_torch(tensor, dtype=dtype)

    # -- Mask conversion
    mask_configs = {
        "env_mask": {"id_arg": "env_ids", "shape_attrs": ["num_instances", "num_envs"]},
        "joint_mask": {"id_arg": "joint_ids", "shape_attrs": ["num_joints"]},
        "body_mask": {"id_arg": "body_ids", "shape_attrs": ["num_bodies"]},
    }

    for mask_name, config in mask_configs.items():
        id_arg_name = config["id_arg"]
        if mask_name in sig.parameters and id_arg_name in arguments and arguments[id_arg_name] is not None:
            indices = arguments.pop(id_arg_name)
            shape_val = 0
            for attr in config["shape_attrs"]:
                val = getattr(self, attr, None)
                if val is not None:
                    shape_val = val
                    break
            if shape_val == 0:
                raise ValueError(
                    f"Cannot convert '{id_arg_name}' to '{mask_name}'. The instance is missing one of the "
                    f"following attributes: {config['shape_attrs']}."
                )

            mask_torch = torch.zeros(shape_val, dtype=torch.bool, device=device)

            if isinstance(indices, slice):
                mask_torch[indices] = True
            elif isinstance(indices, (list, tuple, torch.Tensor)):
                mask_torch[torch.as_tensor(indices, device=device)] = True
            else:
                raise TypeError(f"Unsupported type for indices '{type(indices)}'.")

            arguments[mask_name] = wp.from_torch(mask_torch)

    arguments.pop("self")
    return arguments


def _convert_output_to_torch(data):
    """Recursively convert warp arrays in a data structure to torch tensors."""
    if isinstance(data, wp.array):
        return wp.to_torch(data)
    elif isinstance(data, (list, tuple)):
        return type(data)(_convert_output_to_torch(item) for item in data)
    elif isinstance(data, dict):
        return {key: _convert_output_to_torch(value) for key, value in data.items()}
    else:
        return data


def torch_frontend_method(tensor_args: dict[str, any] | None = None, *, convert_output: bool = False):
    """A method decorator to specify tensor conversion rules for the torch frontend.

    This decorator attaches metadata to a method, which is then used by the
    `torch_frontend_class` decorator to apply the correct data conversions.

    Args:
        tensor_args: A dictionary mapping tensor argument names to their
                     target `warp.dtype`. Defaults to None.
        convert_output: If True, the output of the decorated function will be
                        converted from warp arrays to torch tensors. Defaults to False.

    Example:
        >>> @torch_frontend_class
        ... class MyAsset:
        ...     @torch_frontend_method({"root_state": wp.transformf})
        ...     def write_root_state_to_sim(self, root_state: wp.array, env_mask: wp.array | None = None):
        ...         pass
        ...
        ...     @torch_frontend_method(convert_output=True)
        ...     def get_root_state(self) -> wp.array:
        ...         pass
    """
    if tensor_args is None:
        tensor_args = {}

    def decorator(func: Callable) -> Callable:
        setattr(func, "_torch_frontend_spec", {"tensor_args": tensor_args, "convert_output": convert_output})
        return func

    return decorator


def torch_frontend_class(cls=None, *, indices_arg: str = "env_ids", mask_arg: str = "env_mask"):
    """A class decorator to add a PyTorch frontend to a class that uses a Warp backend.

    This decorator patches the ``__init__`` method of a class. After the original ``__init__`` is called,
    it checks for a `frontend` attribute on the instance. If `self.frontend` is "torch", it inspects
    the class for methods decorated with `@torch_frontend_method` and wraps them to make them
    accept PyTorch tensors.

    The wrapped methods will:
    - Convert specified PyTorch tensor arguments to Warp arrays based on the rules defined
      in the `@torch_frontend_method` decorator.
    - Convert an argument with indices (e.g., `env_ids`) to a boolean mask (e.g., `env_mask`).

    If `self.frontend` is not "torch", the class's methods remain unchanged, ensuring zero
    overhead for the default Warp backend.

    Args:
        cls: The class to decorate. This is handled automatically by Python.
        indices_arg: The name of the argument that may contain indices for conversion to a mask.
                     Defaults to "env_ids".
        mask_arg: The name of the argument that will receive the generated boolean mask.
                  Defaults to "env_mask".

    Example:
        >>> import warp as wp
        >>> import torch
        >>>
        >>> @torch_frontend_class
        ... class MyAsset:
        ...     def __init__(self, num_envs, device, frontend="warp"):
        ...         self.num_instances = num_envs
        ...         self.device = device
        ...         self.frontend = frontend
        ...         wp.init()
        ...
        ...     @torch_frontend_method({"root_state": wp.transformf})
        ...     def write_root_state_to_sim(self, root_state: wp.array, env_mask: wp.array | None = None):
        ...         print(f"root_state type: {type(root_state)}")
        ...         if env_mask:
        ...             print(f"env_mask type: {type(env_mask)}")
        ...
        >>> # -- Using warp frontend (no overhead)
        >>> asset_wp = MyAsset(num_envs=4, device="cpu", frontend="warp")
        >>> root_state_wp = wp.zeros(4, dtype=wp.transformf)
        >>> asset_wp.write_root_state_to_sim(root_state_wp)
        root_state type: <class 'warp.types.array'>
        >>>
        >>> # -- Using torch frontend (methods are patched)
        >>> asset_torch = MyAsset(num_envs=4, device="cpu", frontend="torch")
        >>> root_state_torch = torch.rand(4, 7)
        >>> asset_torch.write_root_state_to_sim(root_state_torch, env_ids=[0, 2])
        root_state type: <class 'warp.types.array'>
        env_mask type: <class 'warp.types.array'>
    """
    # This allows using the decorator with or without parentheses:
    # @torch_frontend_class or @torch_frontend_class(indices_arg="...")
    if cls is None:
        return functools.partial(torch_frontend_class, indices_arg=indices_arg, mask_arg=mask_arg)

    original_init = cls.__init__

    @functools.wraps(original_init)
    def new_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)

        if getattr(self, "frontend", "warp") == "torch":
            for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
                if hasattr(method, "_torch_frontend_spec"):
                    spec = getattr(method, "_torch_frontend_spec")
                    convert_output = spec.get("convert_output", False)

                    @functools.wraps(method)
                    def adapted_method_wrapper(self, *args, method=method, **kwargs):
                        converted_args = _analyze_and_convert_args(self, method, *args, **kwargs)
                        output = method(self, **converted_args)

                        if convert_output:
                            return _convert_output_to_torch(output)
                        else:
                            return output

                    setattr(self, name, adapted_method_wrapper.__get__(self, cls))

    cls.__init__ = new_init
    return cls
