# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities for applying and removing semantic labels to USD prims."""

from __future__ import annotations

import contextlib
import logging

from pxr import Usd, UsdGeom

# USD Semantics is only available in Isaac Sim 5.0 and later.
with contextlib.suppress(ModuleNotFoundError, ImportError):
    from pxr import UsdSemantics

from isaaclab.utils.version import get_isaac_sim_version

from .stage import get_current_stage

# import logger
logger = logging.getLogger(__name__)


def add_labels(prim: Usd.Prim, labels: list[str], instance_name: str = "class", overwrite: bool = True) -> None:
    """Apply semantic labels to a prim using the :class:`UsdSemantics.LabelsAPI`.

    This function is a wrapper around the :func:`omni.replicator.core.functional.modify.semantics` function.
    It applies the labels to the prim using the :class:`UsdSemantics.LabelsAPI`.

    .. versionadded:: 2.3.0
        This function is available in Isaac Sim 5.0 and later, which introduces the :class:`UsdSemantics.LabelsAPI`.
        For previous versions, the function falls back to use the deprecated :class:`UsdSemantics.SemanticsAPI` instead.

    Example:
        >>> prim = sim_utils.create_prim("/World/Test/Sphere", "Sphere", stage=stage, attributes={"radius": 10.0})
        >>> sim_utils.add_labels(prim, labels=["sphere"], instance_name="class")

    Args:
        prim: The USD prim to add or update labels on.
        labels: The list of labels to apply.
        instance_name: The name of the semantic instance. Defaults to "class".
        overwrite: Whether to overwrite existing labels for this instance. If False,
          the new labels are appended to existing ones (if any). Defaults to True.
    """
    # Try modern approach (Isaac Sim >= 5.0)
    try:
        import omni.replicator.core.functional as rep_functional

        mode = "replace" if overwrite else "add"
        rep_functional.modify.semantics(prim, {instance_name: labels}, mode=mode)

        return
    except (ModuleNotFoundError, ImportError) as e:
        # check if we are using isaac sim 5.0
        if get_isaac_sim_version().major >= 5:
            logger.warning(
                f"Failed to add labels to prim {prim.GetPath()} using Replicator API: {e}. "
                "\nPlease ensure Replicator API is enabled by passing '--enable_cameras' to the AppLauncher."
                "\nFalling back to legacy approach."
            )

    # Try legacy approach (Isaac Sim < 5.0)
    try:
        import Semantics

        # check we have only one label
        if len(labels) != 1:
            raise ValueError(f"Only one label can be applied to a prim. Received: {labels}")
        # set the semantic API for the instance
        instance_name = f"{instance_name}_{labels[0]}"
        sem = Semantics.SemanticsAPI.Apply(prim, instance_name)
        # create semantic type and data attributes
        sem.CreateSemanticTypeAttr()
        sem.CreateSemanticDataAttr()
        sem.GetSemanticTypeAttr().Set(instance_name)
        sem.GetSemanticDataAttr().Set(labels[0])
    except Exception as e:
        logger.warning(
            f"Failed to add labels to prim {prim.GetPath()} using legacy API: {e}. "
            "\nSemantics functionality may not be available in this Isaac Sim version."
            " Please open an issue at https://github.com/isaac-sim/IsaacLab/issues if you believe this is a bug."
        )


def get_labels(prim: Usd.Prim) -> dict[str, list[str]]:
    """Get all semantic labels (:class:`UsdSemantics.LabelsAPI`) applied to a prim.

    .. versionadded:: 2.3.0
        This function is available in Isaac Sim 5.0 and later. For previous versions,
        please use :mod:`isaacsim.core.utils.semantics` module instead.

    Args:
        prim: The USD prim to return labels for.

    Returns:
        A dictionary mapping instance names to a list of labels.
        If no labels are found, it returns an empty dictionary.
    """
    result = {}
    for schema_name in prim.GetAppliedSchemas():
        if schema_name.startswith("SemanticsLabelsAPI:"):
            instance_name = schema_name.split(":", 1)[1]
            sem_api = UsdSemantics.LabelsAPI(prim, instance_name)
            labels_attr = sem_api.GetLabelsAttr()
            if labels_attr:
                labels = labels_attr.Get()
                result[instance_name] = list(labels) if labels is not None else []
            else:
                result[instance_name] = []
    return result


def remove_labels(prim: Usd.Prim, instance_name: str | None = None, include_descendants: bool = False):
    """Removes semantic labels (:class:`UsdSemantics.LabelsAPI`) from a prim and optionally its descendants.

    .. versionadded:: 2.3.0
        This function is available in Isaac Sim 5.0 and later. For previous versions,
        please use :mod:`isaacsim.core.utils.semantics` module instead.

    Args:
        prim: The USD prim to remove labels from.
        instance_name: The specific instance name to remove. Defaults to None, in which case
            *all* labels are removed.
        include_descendants: Whether to also traverse children and remove labels recursively.
            Defaults to False.
    """

    def _remove_single_prim_labels(target_prim: Usd.Prim):
        """Helper function to remove labels from a single prim."""
        schemas_to_remove = []
        for schema_name in target_prim.GetAppliedSchemas():
            if schema_name.startswith("SemanticsLabelsAPI:"):
                current_instance = schema_name.split(":", 1)[1]
                if instance_name is None or current_instance == instance_name:
                    schemas_to_remove.append(current_instance)

        for inst_to_remove in schemas_to_remove:
            target_prim.RemoveAPI(UsdSemantics.LabelsAPI, inst_to_remove)

    if include_descendants:
        for p in Usd.PrimRange(prim):
            _remove_single_prim_labels(p)
    else:
        _remove_single_prim_labels(prim)


def check_missing_labels(prim_path: str | None = None, stage: Usd.Stage | None = None) -> list[str]:
    """Checks whether the prim and its descendants at the provided path have missing
    semantic labels (:class:`UsdSemantics.LabelsAPI`).

    .. note::
        The function checks only prims that are :class:`UsdGeom.Gprim` type.

    .. versionadded:: 2.3.0
        This function is available in Isaac Sim 5.0 and later. For previous versions,
        please use :mod:`isaacsim.core.utils.semantics` module instead.

    Args:
        prim_path: The prim path to search from. If None, the entire stage is inspected.
        stage: The stage to search from. If None, the current stage is used.

    Returns:
        A list containing prim paths to prims with no labels applied.
    """
    # check if stage is valid
    stage = stage if stage else get_current_stage()

    # check if inspect path is valid
    start_prim = stage.GetPrimAtPath(prim_path) if prim_path else stage.GetPseudoRoot()
    if not start_prim:
        # Allow None prim_path for whole stage check, warn if path specified but not found
        if prim_path:
            logger.warning(f"No prim found at path '{prim_path}'. Returning from check for semantic labels.")
        return []

    # iterate over prim and its children
    prim_paths = []
    for prim in Usd.PrimRange(start_prim):
        if prim.IsA(UsdGeom.Gprim):
            has_any_label = False
            for schema_name in prim.GetAppliedSchemas():
                if schema_name.startswith("SemanticsLabelsAPI:"):
                    has_any_label = True
                    break
            if not has_any_label:
                prim_paths.append(prim.GetPath().pathString)

    return prim_paths


def count_total_labels(prim_path: str | None = None, stage: Usd.Stage | None = None) -> dict[str, int]:
    """Counts the number of semantic labels (:class:`UsdSemantics.LabelsAPI`) applied to the prims at the provided path.

    This function iterates over all the prims from the provided path and counts the number of times
    each label is applied to the prims. It returns a dictionary of labels and their corresponding count.

    .. versionadded:: 2.3.0
        This function is available in Isaac Sim 5.0 and later. For previous versions,
        please use :mod:`isaacsim.core.utils.semantics` module instead.

    Args:
        prim_path: The prim path to search from. If None, the entire stage is inspected.
        stage: The stage to search from. If None, the current stage is used.

    Returns:
        A dictionary mapping individual labels to their total count across all instances.
        The dictionary includes a 'missing_labels' count for prims with no labels.
    """
    stage = stage if stage else get_current_stage()

    start_prim = stage.GetPrimAtPath(prim_path) if prim_path else stage.GetPseudoRoot()
    if not start_prim:
        if prim_path:
            logger.warning(f"No prim found at path '{prim_path}'. Returning from count for semantic labels.")
        return {"missing_labels": 0}

    labels_counter = {"missing_labels": 0}
    for prim in Usd.PrimRange(start_prim):
        if prim.IsA(UsdGeom.Gprim):
            labels_dict = get_labels(prim)
            if not labels_dict:
                labels_counter["missing_labels"] += 1
            else:
                # Iterate through all labels from all instances on the prim
                all_labels = [label for sublist in labels_dict.values() for label in sublist if label]
                for label in all_labels:
                    labels_counter[label] = labels_counter.get(label, 0) + 1

    return labels_counter
