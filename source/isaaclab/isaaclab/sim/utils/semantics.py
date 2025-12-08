# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import logging

from pxr import Usd, UsdGeom

from isaaclab.sim.utils.stage import get_current_stage

# from Isaac Sim 4.2 onwards, pxr.Semantics is deprecated
try:
    import Semantics
except ModuleNotFoundError:
    from pxr import Semantics

# import logger
logger = logging.getLogger(__name__)


def add_labels(prim: Usd.Prim, labels: list[str], instance_name: str = "class", overwrite: bool = True) -> None:
    """Apply semantic labels to a prim using the Semantics.LabelsAPI.

    Args:
        prim (Usd.Prim): Usd Prim to add or update labels on.
        labels (list): The list of labels to apply.
        instance_name (str, optional): The name of the semantic instance. Defaults to "class".
        overwrite (bool, optional): If True (default), existing labels for this instance are replaced.
                                   If False, the new labels are appended to existing ones (if any).
    """
    import omni.replicator.core.functional as F

    mode = "replace" if overwrite else "add"
    F.modify.semantics(prim, {instance_name: labels}, mode=mode)


def get_labels(prim: Usd.Prim) -> dict[str, list[str]]:
    """Returns semantic labels (Semantics.LabelsAPI) applied to a prim.

    Args:
        prim (Usd.Prim): Prim to return labels for.

    Returns:
        dict[str, list[str]]: Dictionary mapping instance names to a list of labels.
                              Returns an empty dict if no LabelsAPI instances are found.
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


def remove_labels(prim: Usd.Prim, instance_name: str | None = None, include_descendants: bool = False) -> None:
    """Removes semantic labels (Semantics.LabelsAPI) from a prim.

    Args:
        prim (Usd.Prim): Prim to remove labels from.
        instance_name (str | None, optional): Specific instance name to remove.
                                              If None (default), removes *all* LabelsAPI instances.
        include_descendants (bool, optional): Also traverse children and remove labels recursively. Defaults to False.
    """

    def remove_single_prim_labels(target_prim: Usd.Prim):
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
            remove_single_prim_labels(p)
    else:
        remove_single_prim_labels(prim)


def check_missing_labels(prim_path: str | None = None) -> list[str]:
    """Returns a list of prim paths of meshes with missing semantic labels (Semantics.LabelsAPI).

    Args:
        prim_path (str | None): This will check Prim path and its childrens' labels. If None, checks the whole stage.

    Returns:
        list[str]: Prim paths of meshes with no LabelsAPI applied.
    """
    prim_paths = []
    stage = get_current_stage()
    if stage is None:
        logger.warning("Invalid stage, skipping label check")
        return prim_paths

    start_prim = stage.GetPrimAtPath(prim_path) if prim_path else stage.GetPseudoRoot()
    if not start_prim:
        # Allow None prim_path for whole stage check, warn if path specified but not found
        if prim_path:
            logger.warning(f"Prim path not found: {prim_path}")
        return prim_paths

    prims_to_check = Usd.PrimRange(start_prim)

    for prim in prims_to_check:
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
    """Returns a list of [prim_path, label] for meshes where at least one semantic label (LabelsAPI)
       is not found within the prim's path string (case-insensitive, ignoring '_' and '-').

    Args:
        prim_path (str | None): This will check Prim path and its childrens' labels. If None, checks the whole stage.

    Returns:
        list[list[str]]: List containing pairs of [prim_path, first_incorrect_label].
    """
    incorrect_pairs = []
    stage = get_current_stage()
    if stage is None:
        logger.warning("Invalid stage, skipping label check")
        return incorrect_pairs

    start_prim = stage.GetPrimAtPath(prim_path) if prim_path else stage.GetPseudoRoot()
    if not start_prim:
        if prim_path:
            logger.warning(f"Prim path not found: {prim_path}")
        return incorrect_pairs

    prims_to_check = Usd.PrimRange(start_prim)

    for prim in prims_to_check:
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
        prim_path (str | None): This will check Prim path and its childrens' labels. If None, checks the whole stage.

    Returns:
        dict[str, int]: Dictionary mapping individual labels to their total count across all instances.
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

    prims_to_check = Usd.PrimRange(start_prim)

    for prim in prims_to_check:
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
        prim (Usd.Prim): The starting prim to upgrade.
        include_descendants (bool, optional): If True, upgrades the prim and all its descendants.
                                     If False (default), upgrades only the specified prim.

    Returns:
        int: The total number of SemanticsAPI instances successfully upgraded to LabelsAPI.
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
