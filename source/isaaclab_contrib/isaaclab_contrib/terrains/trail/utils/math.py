# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2024-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import random


def interp(
    param0: float | tuple[float, float], param1: float | tuple[float, float], x: float
) -> float | tuple[float, float]:
    """Linearly interpolate between ``param0`` and ``param1``.

    Each parameter may be either a float or a tuple of two floats.  When a
    tuple is provided the result is a tuple obtained by interpolating each
    element independently (useful for representing ranges).  The interpolation
    is performed as ``param0 + (param1 - param0) * x`` on scalar values.

    Args:
        param0: value or (min, max) for which ``x`` is zero.
        param1: value or (min, max) for which ``x`` is one.
        x: fraction between 0 and 1 indicating interpolation position.

    Returns:
        Interpolated value; same type as the inputs (float or tuple).

    Raises:
        TypeError: if ``param0`` or ``param1`` are not either float or a
            2-tuple of floats.
        ValueError: if the tuple arguments are not length two.
    """

    # check validity of params
    def validate(p, name: str):
        if isinstance(p, float):
            return
        if isinstance(p, tuple):
            if len(p) != 2:
                raise ValueError(f"{name} tuple must have length 2, got {len(p)}")
            if not all(isinstance(v, float) for v in p):
                raise TypeError(f"{name} tuple elements must be floats")
            return
        raise TypeError(f"{name} must be float or tuple[float,float], got {type(p)}")

    validate(param0, "param0")
    validate(param1, "param1")

    # check validity of x
    if x < 0.0 or x > 1.0:
        raise ValueError(f"x must be between 0 and 1, got {x}")

    # interpolation
    if isinstance(param0, float) and isinstance(param1, float):
        return param0 + (param1 - param0) * x
    elif isinstance(param0, float):
        return (interp(param0, param1[0], x), interp(param0, param1[1], x))
    elif isinstance(param1, float):
        return (interp(param0[0], param1, x), interp(param0[1], param1, x))
    else:
        return (interp(param0[0], param1[0], x), interp(param0[1], param1[1], x))


def interp_dict_and_sample(cp0: object, cp1: object, x: float) -> dict[str, float]:
    """Linearly interpolate dictionary of parameters.

    Args:
        cp0: object containing params (explanation in notes) for which x is zero.
        cp1: object containing params (explanation in notes) for which x is one.
        x: value between zero and one.

    Both c0 and cp1 contain a member called "params", which are dictionaries.
        The key is the corresponding name and the value is of type float or tuple[float,float]:
        * float: a single value
        * tuple[float, float]: the limits of a uniform distribution

    Returns:
        a dictionary of interpolated and sampled values.

    Raises:
        RuntimeError: If the arguments are not of the correct type.
    """
    # Ensure that arguments contain a `params` attribute
    if not hasattr(cp0, "params") or not hasattr(cp1, "params"):
        raise RuntimeError("cp0 or cp1 do not contain parameters that can be interpolated.")
    # Ensure that `params` is a dictionary
    if not isinstance(cp0.params, dict) or not isinstance(cp1.params, dict):
        raise RuntimeError("params must be a dict.")
    # Interpolate each parameter and sample if necessary
    values: dict[str, float] = {}
    for name in cp0.params.keys():
        values[name] = sample(interp(param0=cp0.params[name], param1=cp1.params[name], x=x))
    return values


def sample(limits: float | int | tuple[float | int, float | int]) -> float | int:
    """If ``limits`` is a tuple, sample uniformly from U(min, max).

    If ``limits`` is a float or int, return it unchanged.

    Note: booleans are treated as integers.

    Args:
        limits: Either a scalar value or a (min, max) pair specifying the
            uniform sampling range.

    Returns:
        A single sampled or forwarded value (int or float).
    """
    # return single value
    if not isinstance(limits, tuple):
        return limits
    # sample random int or float
    elif isinstance(limits[0], int):
        return int(round(random.uniform(*limits)))
    # sample random float
    return random.uniform(*limits)


def sample_sign() -> float:
    """Sample +1 or -1 with equal probability."""
    return 1.0 if random.uniform(0.0, 1.0) > 0.5 else -1.0


def in_limits(
    value: object,
    limits: float | int | tuple[float | int, float | int],
    rel_tol: float = 0.0,
) -> object:
    """Test if value is within specified limits.

    If ``limits`` is a tuple, check whether ``value`` is within
    [limits[0], limits[1]] (inclusive). If ``limits`` is a scalar, check
    whether ``value`` equals ``limits`` within the specified relative
    tolerance.

    Note: This function assumes that the type of ``value`` implements the
    ``>=`` and ``<=`` operators against float or int (or is a NumPy / torch
    array supporting elementwise comparisons).

    Args:
        value: The value to check.
        limits: Either a scalar or a (min, max) pair containing lower and
            upper bounds.
        rel_tol: Relative tolerance. Default is 0.0.

    Returns:
        A boolean or boolean-like object indicating whether ``value`` is
        within the specified limits.
    """
    if isinstance(limits, tuple):
        return (value >= limits[0] * (1.0 - rel_tol)) & (value <= limits[1] * (1.0 + rel_tol))
    else:
        return (value >= limits * (1.0 - rel_tol)) & (value <= limits * (1.0 + rel_tol))
