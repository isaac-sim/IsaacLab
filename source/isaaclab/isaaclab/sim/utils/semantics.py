# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import logging

from pxr import Usd, UsdGeom

from .stage import get_current_stage

# from Isaac Sim 4.2 onwards, pxr.Semantics is deprecated
try:
    import Semantics
except ModuleNotFoundError:
    from pxr import Semantics

# import logger
logger = logging.getLogger(__name__)


def add_labels(prim: Usd.Prim, labels: list[str], instance_name: str = "class", overwrite: bool = True) -> None:
    """Apply semantic labels to a prim using the :class:`Semantics.LabelsAPI`.

    Args:
        prim: The USD prim to add or update labels on.
        labels: The list of labels to apply.
        instance_name: The name of the semantic instance. Defaults to "class".
        overwrite: Whether to overwrite existing labels for this instance. If False,
          the new labels are appended to existing ones (if any). Defaults to True.
    """
    import omni.replicator.core.functional as F

    mode = "replace" if overwrite else "add"
    F.modify.semantics(prim, {instance_name: labels}, mode=mode)


def get_labels(prim: Usd.Prim) -> dict[str, list[str]]:
    """Returns semantic labels (:class:`Semantics.LabelsAPI`) applied to a prim.

    Args:
        prim: The USD prim to return labels for.

    Returns:
        A dictionary mapping instance names to a list of labels.
        If no :attr:`LabelsAPI` instances are found, it returns an empty dict.
    """
    result = {}
    for schema_name in prim.GetAppliedSchemas():
        if schema_name.startswith("SemanticsLabelsAPI:"):
            instance_name = schema_name.split(":", 1)[1]
            sem_api = Semantics.LabelsAPI(prim, instance_name)
            labels_attr = sem_api.GetLabelsAttr()
            if labels_attr:
                labels = labels_attr.Get()
                result[instance_name] = list(labels) if labels is not None else []
            else:
                result[instance_name] = []
    return result


def remove_labels(prim: Usd.Prim, instance_name: str | None = None, include_descendants: bool = False):
    """Removes semantic labels (:class:`Semantics.LabelsAPI`) from a prim.

    Args:
        prim: The USD prim to remove labels from.
        instance_name: The specific instance name to remove. Defaults to None, in which case
            *all* :attr:`LabelsAPI` instances are removed.
        include_descendants: Whether to also traverse children and remove labels recursively.
            Defaults to False.
    """

    def _remove_single_prim_labels(target_prim: Usd.Prim):
        schemas_to_remove = []
        for schema_name in target_prim.GetAppliedSchemas():
            if schema_name.startswith("SemanticsLabelsAPI:"):
                current_instance = schema_name.split(":", 1)[1]
                if instance_name is None or current_instance == instance_name:
                    schemas_to_remove.append(current_instance)

        for inst_to_remove in schemas_to_remove:
            target_prim.RemoveAPI(Semantics.LabelsAPI, inst_to_remove)

    if include_descendants:
        for p in Usd.PrimRange(prim):
            _remove_single_prim_labels(p)
    else:
        _remove_single_prim_labels(prim)


def check_missing_labels(prim_path: str | None = None) -> list[str]:
    """Checks whether the prim and its descendants at the provided path have missing
    semantic labels (:class:`Semantics.LabelsAPI`).

    Note:
        The function checks only prims that are "Mesh" type.

    Args:
        prim_path: The prim path to search from. If None, the entire stage is inspected.

    Returns:
        A list containing prim paths to prims with no :class:`LabelsAPI` applied.
    """
    # check if stage is valid
    stage = get_current_stage()
    if stage is None:
        logger.warning("Invalid stage. Skipping check for semantic labels.")
        return []

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
        if prim.IsA(UsdGeom.Mesh):
            has_any_label = False
            for schema_name in prim.GetAppliedSchemas():
                if schema_name.startswith("SemanticsLabelsAPI:"):
                    has_any_label = True
                    break
            if not has_any_label:
                prim_paths.append(prim.GetPath().pathString)

    return prim_paths


def check_incorrect_labels(prim_path: str | None = None) -> list[list[str]]:
    """Check whether the prim and its descendants at the provided path have incorrect
    semantic labels (:class:`Semantics.LabelsAPI`).

    A label is considered incorrect if it is not found within the prim's path string
    (case-insensitive, ignoring '_' and '-'). For example, if the prim path is "/World/Cube",
    and the label is "cube", it is considered incorrect because it is not found within the prim's
    path string.
    
    Returns a list of [prim_path, label] for meshes where at least one semantic label (LabelsAPI)
       is not found within the prim's path string (case-insensitive, ignoring '_' and '-').

    Args:
        prim_path: This will check Prim path and its childrens' labels. If None, checks the whole stage.

    Returns:
        List containing pairs of [prim_path, first_incorrect_label].
    """
    stage = get_current_stage()
    if stage is None:
        logger.warning("Invalid stage, skipping label check")
        return []

    start_prim = stage.GetPrimAtPath(prim_path) if prim_path else stage.GetPseudoRoot()
    if not start_prim:
        if prim_path:
            logger.warning(f"Prim path not found: {prim_path}")
        return []

    incorrect_pairs = []
    for prim in Usd.PrimRange(start_prim):
        if prim.IsA(UsdGeom.Mesh):
            labels_dict = get_labels(prim)
            if labels_dict:
                prim_path_str = prim.GetPath().pathString.lower()
                all_labels = [
                    label for sublist in labels_dict.values() for label in sublist if label
                ]  # Flatten and filter None/empty
                for label in all_labels:
                    label_lower = label.lower()
                    # Check if label (or label without separators) is in path
                    if (
                        label_lower not in prim_path_str
                        and label_lower.replace("_", "") not in prim_path_str
                        and label_lower.replace("-", "") not in prim_path_str
                    ):
                        incorrect_pair = [prim.GetPath().pathString, label]
                        incorrect_pairs.append(incorrect_pair)
                        break  # Only report first incorrect label per prim
    return incorrect_pairs


def count_labels_in_scene(prim_path: str | None = None) -> dict[str, int]:
    """Returns a dictionary of semantic labels (Semantics.LabelsAPI) and their corresponding count.

    Args:
        prim_path: This will check Prim path and its childrens' labels. If None, checks the whole stage.

    Returns:
        Dictionary mapping individual labels to their total count across all instances.
        Includes a 'missing_labels' count for meshes with no LabelsAPI.
    """
    labels_counter = {"missing_labels": 0}
    stage = get_current_stage()
    if stage is None:
        logger.warning("Invalid stage, skipping label check")
        return labels_counter

    start_prim = stage.GetPrimAtPath(prim_path) if prim_path else stage.GetPseudoRoot()
    if not start_prim:
        if prim_path:
            logger.warning(f"Prim path not found: {prim_path}")
        return labels_counter

    for prim in Usd.PrimRange(start_prim):
        if prim.IsA(UsdGeom.Mesh):
            labels_dict = get_labels(prim)
            if not labels_dict:
                labels_counter["missing_labels"] += 1
            else:
                # Iterate through all labels from all instances on the prim
                all_labels = [label for sublist in labels_dict.values() for label in sublist if label]
                for label in all_labels:
                    labels_counter[label] = labels_counter.get(label, 0) + 1

    return labels_counter


def upgrade_prim_semantics_to_labels(prim: Usd.Prim, include_descendants: bool = False) -> int:
    """Upgrades a prim and optionally its descendants from the deprecated SemanticsAPI
    to the new Semantics.LabelsAPI.

    Converts each found SemanticsAPI instance on the processed prim(s) to a corresponding
    LabelsAPI instance. The old 'semanticType' becomes the new LabelsAPI
    'instance_name', and the old 'semanticData' becomes the single label in the
    new 'labels' list. The old SemanticsAPI is always removed after upgrading.

    Args:
        prim: The starting prim to upgrade.
        include_descendants: If True, upgrades the prim and all its descendants.
                                     If False (default), upgrades only the specified prim.

    Returns:
        The total number of SemanticsAPI instances successfully upgraded to LabelsAPI.
    """
    total_upgraded = 0

    prims_to_process = Usd.PrimRange(prim) if include_descendants else [prim]

    for current_prim in prims_to_process:
        if not current_prim:
            continue

        old_semantics = {}
        for prop in current_prim.GetProperties():
            if Semantics.SemanticsAPI.IsSemanticsAPIPath(prop.GetPath()):
                instance_name = prop.SplitName()[1]  # Get instance name (e.g., 'Semantics', 'Semantics_a')
                sem_api = Semantics.SemanticsAPI.Get(current_prim, instance_name)
                if sem_api:
                    typeAttr = sem_api.GetSemanticTypeAttr()
                    dataAttr = sem_api.GetSemanticDataAttr()
                    if typeAttr and dataAttr and instance_name not in old_semantics:
                        old_semantics[instance_name] = (typeAttr.Get(), dataAttr.Get())

        if not old_semantics:
            continue

        for old_instance_name, (old_type, old_data) in old_semantics.items():

            if not old_type or not old_data:
                logger.warning(
                    f"[upgrade_prim] Skipping instance '{old_instance_name}' on {current_prim.GetPath()} due to missing"
                    " type or data."
                )
                continue

            new_instance_name = old_type
            new_labels = [old_data]

            try:
                old_sem_api_to_remove = Semantics.SemanticsAPI.Get(current_prim, old_instance_name)
                if old_sem_api_to_remove:
                    typeAttr = old_sem_api_to_remove.GetSemanticTypeAttr()
                    dataAttr = old_sem_api_to_remove.GetSemanticDataAttr()
                    # Ensure attributes are valid before trying to remove them by name
                    if typeAttr and typeAttr.IsDefined():
                        current_prim.RemoveProperty(typeAttr.GetName())
                    if dataAttr and dataAttr.IsDefined():
                        current_prim.RemoveProperty(dataAttr.GetName())
                    current_prim.RemoveAPI(Semantics.SemanticsAPI, old_instance_name)

                add_labels(current_prim, new_labels, instance_name=new_instance_name, overwrite=False)

                total_upgraded += 1

            except Exception as e:
                logger.warning(f"Failed to upgrade instance '{old_instance_name}' on {current_prim.GetPath()}: {e}")
                continue
    return total_upgraded
