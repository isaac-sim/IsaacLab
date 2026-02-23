# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities for applying and removing semantic labels to USD prims."""

from __future__ import annotations

import logging

from pxr import Usd, UsdGeom, UsdSemantics

from .stage import get_current_stage

# import logger
logger = logging.getLogger(__name__)


def add_labels(prim: Usd.Prim, labels: list[str], instance_name: str = "class", overwrite: bool = True) -> None:
    """Apply semantic labels to a prim using :class:`UsdSemantics.LabelsAPI`.

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
    labels_api = UsdSemantics.LabelsAPI.Apply(prim, instance_name)
    labels_attr = labels_api.CreateLabelsAttr()
    if overwrite:
        labels_attr.Set(labels)
    else:
        existing = labels_attr.Get()
        if existing:
            combined = list(existing) + [lbl for lbl in labels if lbl not in existing]
            labels_attr.Set(combined)
        else:
            labels_attr.Set(labels)


def get_labels(prim: Usd.Prim) -> dict[str, list[str]]:
    """Get all semantic labels (:class:`UsdSemantics.LabelsAPI`) applied to a prim.

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
