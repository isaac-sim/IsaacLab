# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Transformations of strings."""


import re
from typing import Optional


def to_camel_case(snake_str: str, to: Optional[str] = "cC") -> str:
    """Converts a string from snake case to camel case.

    Args:
        snake_str (str): A string in snake case (i.e. with '_')
        to (Optional[str], optional): Convention to convert string to. Defaults to "cC".

    Raises:
        ValueError: Invalid input argument `to`, i.e. not "cC" or "CC".

    Returns:
        str: A string in camel-case format.
    """
    # check input is correct
    if to not in ["cC", "CC"]:
        msg = "to_camel_case(): Choose a valid `to` argument (CC or cC)"
        raise ValueError(msg)
    # convert string to lower case and split
    components = snake_str.lower().split("_")
    if to == "cC":
        # We capitalize the first letter of each component except the first one
        # with the 'title' method and join them together.
        return components[0] + "".join(x.title() for x in components[1:])
    else:
        # Capitalize first letter in all the components
        return "".join(x.title() for x in components)


def to_snake_case(camel_str: str) -> str:
    """Converts a string from camel case to snake case.

    Args:
        camel_str (str): A string in camel case.

    Returns:
        str: A string in snake case (i.e. with '_')
    """
    camel_str = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", camel_str)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", camel_str).lower()
