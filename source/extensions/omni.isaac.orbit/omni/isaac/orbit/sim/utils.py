# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import functools
from typing import Any, Callable

import carb
import omni.isaac.core.utils.stage as stage_utils
import omni.kit.commands
from pxr import Sdf, Usd

from omni.isaac.orbit.utils.string import to_camel_case


def safe_set_attribute_on_usd_schema(schema_api: Usd.APISchemaBase, name: str, value: Any, camel_case: bool = True):
    """Set the value of an attribute on its USD schema if it exists.

    A USD API schema serves as an interface or API for authoring and extracting a set of attributes.
    They typically derive from the :class:`pxr.Usd.SchemaBase` class. This function checks if the
    attribute exists on the schema and sets the value of the attribute if it exists.

    Args:
        schema_api (Usd.APISchemaBase): The USD schema to set the attribute on.
        name (str): The name of the attribute.
        value (Any): The value to set the attribute to.
        camel_case (bool, optional): Whether to convert the attribute name to camel case. Defaults to True.

    Raises:
        TypeError: When the input attribute name does not exist on the provided schema API.
    """
    # if value is None, do nothing
    if value is None:
        return
    # convert attribute name to camel case
    if camel_case:
        attr_name = to_camel_case(name, to="CC")
    else:
        attr_name = name
    # retrieve the attribute
    # reference: https://openusd.org/dev/api/_usd__page__common_idioms.html#Usd_Create_Or_Get_Property
    attr = getattr(schema_api, f"Create{attr_name}Attr", None)
    # check if attribute exists
    if attr is not None:
        attr().Set(value)
    else:
        # think: do we ever need to create the attribute if it doesn't exist?
        #   currently, we are not doing this since the schemas are already created with some defaults.
        carb.log_error(f"Attribute '{attr_name}' does not exist on prim '{schema_api.GetPath()}'.")
        raise TypeError(f"Attribute '{attr_name}' does not exist on prim '{schema_api.GetPath()}'.")


def safe_set_attribute_on_usd_prim(prim: Usd.Prim, attr_name: str, value: Any, camel_case: bool = True):
    """Set the value of a attribute on its USD prim.

    The function creates a new attribute if it does not exist on the prim. This is because in some cases (such
    as with shaders), their attributes are not exposed as USD prim properties that can be altered. This function
    allows us to set the value of the attributes in these cases.

    Args:
        prim (Usd.Prim): The USD prim to set the attribute on.
        attr_name (str): The name of the attribute.
        value (Any): The value to set the attribute to.
        camel_case (bool, optional): Whether to convert the attribute name to camel case. Defaults to True.
    """
    # if value is None, do nothing
    if value is None:
        return
    # convert attribute name to camel case
    if camel_case:
        attr_name = to_camel_case(attr_name, to="cC")
    # resolve sdf type based on value
    if isinstance(value, bool):
        sdf_type = Sdf.ValueTypeNames.Bool
    elif isinstance(value, int):
        sdf_type = Sdf.ValueTypeNames.Int
    elif isinstance(value, float):
        sdf_type = Sdf.ValueTypeNames.Float
    elif isinstance(value, (tuple, list)) and len(value) == 3 and any(isinstance(v, float) for v in value):
        sdf_type = Sdf.ValueTypeNames.Float3
    elif isinstance(value, (tuple, list)) and len(value) == 2 and any(isinstance(v, float) for v in value):
        sdf_type = Sdf.ValueTypeNames.Float2
    else:
        raise NotImplementedError(
            f"Cannot set attribute '{attr_name}' with value '{value}'. Please modify the code to support this type."
        )
    # change property
    omni.kit.commands.execute(
        "ChangePropertyCommand",
        prop_path=Sdf.Path(f"{prim.GetPath()}.{attr_name}"),
        value=value,
        prev=None,
        type_to_create_if_not_exist=sdf_type,
        usd_context_name=prim.GetStage(),
    )


def apply_nested(func: Callable) -> Callable:
    """Decorator to apply a function to all prims under a specified prim-path.

    The function iterates over the provided prim path and all its children to apply input function
    to all prims under the specified prim path.

    Note:
        If the function succeeds to apply to a prim, it will not look at the children of that prim.
        This is based on the physics behavior that nested schemas are not allowed. For example, a parent prim
        and its child prim cannot both have a rigid-body schema applied on them, or it is not possible to
        have nested articulations.

    Args:
        func (Callable): The function to apply to all prims under a specified prim-path. The function
            must take the prim-path, the configuration object and the stage as inputs. It should return
            a boolean indicating whether the function succeeded or not.

    Returns:
        Callable: The wrapped function that applies the function to all prims under a specified prim-path.
    """

    @functools.wraps(func)
    def wrapper(prim_path: str, cfg: object, stage: Usd.Stage | None = None):
        # get current stage
        if stage is None:
            stage = stage_utils.get_current_stage()
        # get USD prim
        prim: Usd.Prim = stage.GetPrimAtPath(prim_path)
        # check if prim is valid
        if not prim.IsValid():
            raise ValueError(f"Prim at path '{prim_path}' is not valid.")
        # iterate over all prims under prim-path
        all_prims = [prim]
        while len(all_prims) > 0:
            # get current prim
            child_prim = all_prims.pop(0)
            child_prim_path = child_prim.GetPath().pathString
            # check if prim is a prototype
            # note: we prefer throwing a warning instead of ignoring the prim since the user may
            #   have intended to set properties on the prototype prim.
            if child_prim.IsInstance():
                carb.log_warn(f"Cannot perform '{func.__name__}' on instanced prim: '{child_prim_path}'")
                continue
            # set properties
            success = func(child_prim_path, cfg, stage=stage)
            # if successful, do not look at children
            # this is based on the physics behavior that nested schemas are not allowed
            if not success:
                all_prims += child_prim.GetChildren()

    return wrapper
