# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities for applying and removing semantic labels to USD prims."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .stage import get_current_stage

if TYPE_CHECKING:
    from pxr import Usd, UsdGeom, UsdSemantics  # noqa: F401

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
    from pxr import UsdSemantics  # noqa: PLC0415

    labels_attr = UsdSemantics.LabelsAPI.Apply(prim, instance_name).CreateLabelsAttr()
    existing = None if overwrite else labels_attr.Get()
    if existing:
        labels = list(existing) + [lbl for lbl in labels if lbl not in existing]
    labels_attr.Set(labels)


def get_labels(prim: Usd.Prim) -> dict[str, list[str]]:
    """Get all semantic labels (:class:`UsdSemantics.LabelsAPI`) applied to a prim.

    Args:
        prim: The USD prim to return labels for.

    Returns:
        A dictionary mapping instance names to a list of labels.
        If no labels are found, it returns an empty dictionary.
    """
    from pxr import UsdSemantics  # noqa: PLC0415

    result = {}
    for instance_name in _label_instance_names(prim):
        labels_attr = UsdSemantics.LabelsAPI(prim, instance_name).GetLabelsAttr()
        labels = labels_attr.Get() if labels_attr else None
        result[instance_name] = list(labels) if labels is not None else []
    return result


def _label_instance_names(prim: Usd.Prim) -> list[str]:
    """Return the instance names of all ``UsdSemantics.LabelsAPI`` schemas applied to a prim."""
    prefix = "SemanticsLabelsAPI:"
    return [schema_name[len(prefix) :] for schema_name in prim.GetAppliedSchemas() if schema_name.startswith(prefix)]


def remove_labels(prim: Usd.Prim, instance_name: str | None = None, include_descendants: bool = False):
    """Removes semantic labels (:class:`UsdSemantics.LabelsAPI`) from a prim and optionally its descendants.

    Args:
        prim: The USD prim to remove labels from.
        instance_name: The specific instance name to remove. Defaults to None, in which case
            *all* labels are removed.
        include_descendants: Whether to also traverse children and remove labels recursively.
            Defaults to False.
    """
    from pxr import Usd, UsdSemantics  # noqa: PLC0415

    for target_prim in Usd.PrimRange(prim) if include_descendants else [prim]:
        for current_instance in _label_instance_names(target_prim):
            if instance_name is None or current_instance == instance_name:
                target_prim.RemoveAPI(UsdSemantics.LabelsAPI, current_instance)


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
    from pxr import Usd, UsdGeom  # noqa: PLC0415

    stage = stage if stage else get_current_stage()
    start_prim = stage.GetPrimAtPath(prim_path) if prim_path else stage.GetPseudoRoot()
    if not start_prim:
        if prim_path:
            logger.warning(f"No prim found at path '{prim_path}'. Returning from check for semantic labels.")
        return []

    return [
        prim.GetPath().pathString
        for prim in Usd.PrimRange(start_prim)
        if prim.IsA(UsdGeom.Gprim) and not _label_instance_names(prim)
    ]


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
    from pxr import Usd, UsdGeom  # noqa: PLC0415

    stage = stage if stage else get_current_stage()

    start_prim = stage.GetPrimAtPath(prim_path) if prim_path else stage.GetPseudoRoot()
    if not start_prim:
        if prim_path:
            logger.warning(f"No prim found at path '{prim_path}'. Returning from count for semantic labels.")
        return {"missing_labels": 0}

    labels_counter = {"missing_labels": 0}
    for prim in Usd.PrimRange(start_prim):
        if not prim.IsA(UsdGeom.Gprim):
            continue
        labels_dict = get_labels(prim)
        if not labels_dict:
            labels_counter["missing_labels"] += 1
        for label in (label for labels in labels_dict.values() for label in labels if label):
            labels_counter[label] = labels_counter.get(label, 0) + 1
    return labels_counter
