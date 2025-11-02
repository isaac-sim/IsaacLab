# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Dict, List, Optional, Tuple

import carb
import isaaclab.utils.prims as prim_utils
import omni.usd
import Semantics
from isaaclab.utils.stage import get_current_stage, get_current_stage_id
from pxr import Usd, UsdGeom, UsdSemantics


def add_update_semantics(prim: Usd.Prim, semantic_label: str, type_label: str = "class", suffix="") -> None:
    """[DEPRECATED] Apply a semantic label to a prim or update an existing label using the old SemanticsAPI.

    Args:
        prim (Usd.Prim): Usd Prim to add or update semantics on
        semantic_label (str): The label we want to apply
        type_label (str): The type of semantic information we are specifying (default = "class")
        suffix (str): Additional suffix used to specify multiple semantic attribute names.
    """
    carb.log_warn(
        "Function 'add_update_semantics' uses the deprecated SemanticsAPI. "
        "Consider upgrading the semantics with `upgrade_prim_semantics_to_labels(prim)` first and then using `add_labels` instead."
    )
    # Apply or acquire the existing SemanticAPI
    semantic_api = Semantics.SemanticsAPI.Get(prim, "Semantics" + suffix)
    if not semantic_api:
        semantic_api = Semantics.SemanticsAPI.Apply(prim, "Semantics" + suffix)
        semantic_api.CreateSemanticTypeAttr()
        semantic_api.CreateSemanticDataAttr()

    type_attr = semantic_api.GetSemanticTypeAttr()
    data_attr = semantic_api.GetSemanticDataAttr()

    # Set the type and data for the SemanticAPI
    if type_label is not None:
        type_attr.Set(type_label)
    if semantic_label is not None:
        data_attr.Set(semantic_label)
    return


def remove_all_semantics(prim: Usd.Prim, recursive: bool = False) -> None:
    """[DEPRECATED] Removes all semantic tags (old SemanticsAPI) from a given prim and its children

    Args:
        prim (Usd.Prim): Prim to remove any applied semantic APIs on
        recursive (bool, optional): Also traverse children and remove semantics recursively. Defaults to False.
    """
    carb.log_warn(
        "Function 'remove_all_semantics' uses the deprecated SemanticsAPI. "
        "Consider upgrading the semantics with `upgrade_prim_semantics_to_labels(prim, recursive=True)` first and then using `remove_labels` instead."
    )

    def remove_semantics(input_prim: Usd.Prim):
        for prop in input_prim.GetProperties():
            is_semantic = Semantics.SemanticsAPI.IsSemanticsAPIPath(prop.GetPath())
            if is_semantic:
                name = prop.SplitName()[1]
                sem = Semantics.SemanticsAPI.Get(input_prim, name)

                typeAttr = sem.GetSemanticTypeAttr()
                dataAttr = sem.GetSemanticDataAttr()
                input_prim.RemoveProperty(typeAttr.GetName())
                input_prim.RemoveProperty(dataAttr.GetName())
                input_prim.RemoveAPI(Semantics.SemanticsAPI, name)

    if recursive:
        for p in Usd.PrimRange(prim.GetPrim()):
            remove_semantics(p)
    else:
        remove_semantics(prim)


def get_semantics(prim: Usd.Prim) -> Dict[str, Tuple[str, str]]:
    """[DEPRECATED] Returns semantics (old SemanticsAPI) that are applied to a prim

    Args:
        prim (Usd.Prim): Prim to return semantics for

    Returns:
        Dict[str, Tuple[str,str]]: Dictionary containing the name of the applied semantic, and the type and data associated with that semantic.
    """
    carb.log_warn(
        "Function 'get_semantics' reads the deprecated SemanticsAPI. "
        "Consider upgrading the semantics with `upgrade_prim_semantics_to_labels(prim)` first and then using `get_labels` instead."
    )
    result = {}
    for prop in prim.GetProperties():
        is_semantic = Semantics.SemanticsAPI.IsSemanticsAPIPath(prop.GetPath())
        if is_semantic:
            name = prop.SplitName()[1]
            sem = Semantics.SemanticsAPI.Get(prim, name)

            typeAttr = sem.GetSemanticTypeAttr()
            dataAttr = sem.GetSemanticDataAttr()
            result[name] = (typeAttr.Get(), dataAttr.Get())
    return result


def check_missing_semantics(prim_path: str = None) -> List[str]:
    """[DEPRECATED] Returns a list of prim path of meshes with missing semantics (old SemanticsAPI)

    Args:
        prim_path (str): This will check Prim path and its childrens' semantics

    Returns:
        List[str]: Prim paths
    """
    carb.log_warn(
        "Function 'check_missing_semantics' uses the deprecated SemanticsAPI via get_semantics. "
        "Consider upgrading the semantics with `upgrade_prim_semantics_to_labels(prim, recursive=True)` first and then using `check_missing_labels` instead."
    )
    prim_paths = []
    prims = []

    if prim_path is None:
        stage = get_current_stage()
        prims = stage.Traverse()
    else:
        prims = prim_utils.get_all_matching_child_prims(prim_path, lambda prim: True)

    for prim in prims:
        if prim.IsA(UsdGeom.Mesh):
            semantics = get_semantics(prim)
            if not semantics:
                prim_paths.append(prim.GetPath().pathString)
    return prim_paths


def check_incorrect_semantics(prim_path: str = None) -> List[List[str]]:
    """[DEPRECATED] Returns a list of prim path of meshes with different semantics labels (old SemanticsAPI) than their prim path and their semantic labels

    Args:
        prim_path (str): This will check Prim path and its childrens' semantics

    Returns:
        List[List[str]]: List of prim path and semantic label
    """
    carb.log_warn(
        "Function 'check_incorrect_semantics' uses the deprecated SemanticsAPI via get_semantics. "
        "Consider upgrading the semantics with `upgrade_prim_semantics_to_labels(prim, recursive=True)` first and then using `check_incorrect_labels` instead."
    )
    incorrect_pairs = []
    if prim_path is None:
        stage = get_current_stage()
        prims = stage.Traverse()
    else:
        prims = prim_utils.get_all_matching_child_prims(prim_path, lambda prim: True)

    for prim in prims:
        if prim.IsA(UsdGeom.Mesh):
            semantics = get_semantics(prim)
            prim_path = prim.GetPath().pathString
            if semantics:
                for key in list(semantics.keys()):
                    semantic_label = semantics[key][1].lower()

                    if (
                        semantic_label not in prim_path.lower()
                        and semantic_label.replace("_", "") not in prim_path.lower()
                        and semantic_label.replace("-", "") not in prim_path.lower()
                    ):
                        incorrect_pair = [prim_path, semantics[key][1]]
                        incorrect_pairs.append(incorrect_pair)
                        break
    return incorrect_pairs


def count_semantics_in_scene(prim_path: str = None) -> Dict[str, int]:
    """[DEPRECATED] Returns a dictionary of labels (old SemanticsAPI) and the corresponding count

    Args:
        prim_path (str): This will check Prim path and its childrens' semantics

    Returns:
        Dict[str, int]: labels and count
    """
    carb.log_warn(
        "Function 'count_semantics_in_scene' uses the deprecated SemanticsAPI via get_semantics. "
        "Consider upgrading the semantics with `upgrade_prim_semantics_to_labels(prim, recursive=True)` first and then using `count_labels_in_scene` instead."
    )
    semantics_counter = {"missing": 0}
    if prim_path is None:
        stage = get_current_stage()
        prims = stage.Traverse()
    else:
        prims = prim_utils.get_all_matching_child_prims(prim_path, lambda prim: True)
        print(len(prims))
    for prim in prims:
        if prim.IsA(UsdGeom.Mesh):
            semantics = get_semantics(prim)
            if not semantics:
                semantics_counter["missing"] += 1
            else:
                for key in list(semantics.keys()):
                    if semantics[key][1] not in semantics_counter:
                        semantics_counter[semantics[key][1]] = 1
                    else:
                        semantics_counter[semantics[key][1]] += 1

    return semantics_counter


def add_labels(prim: Usd.Prim, labels: list[str], instance_name: str = "class", overwrite: bool = True) -> None:
    """Apply semantic labels to a prim using the UsdSemantics.LabelsAPI.

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
    """Returns semantic labels (UsdSemantics.LabelsAPI) applied to a prim.

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
            sem_api = UsdSemantics.LabelsAPI(prim, instance_name)
            labels_attr = sem_api.GetLabelsAttr()
            if labels_attr:
                labels = labels_attr.Get()
                result[instance_name] = list(labels) if labels is not None else []
            else:
                result[instance_name] = []
    return result


def remove_labels(prim: Usd.Prim, instance_name: str | None = None, include_descendants: bool = False) -> None:
    """Removes semantic labels (UsdSemantics.LabelsAPI) from a prim.

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
            target_prim.RemoveAPI(UsdSemantics.LabelsAPI, inst_to_remove)

    if include_descendants:
        for p in Usd.PrimRange(prim):
            remove_single_prim_labels(p)
    else:
        remove_single_prim_labels(prim)


def check_missing_labels(prim_path: str | None = None) -> list[str]:
    """Returns a list of prim paths of meshes with missing semantic labels (UsdSemantics.LabelsAPI).

    Args:
        prim_path (str | None): This will check Prim path and its childrens' labels. If None, checks the whole stage.

    Returns:
        list[str]: Prim paths of meshes with no LabelsAPI applied.
    """
    prim_paths = []
    stage = get_current_stage()
    if stage is None:
        carb.log_warn("Invalid stage, skipping label check")
        return prim_paths

    start_prim = stage.GetPrimAtPath(prim_path) if prim_path else stage.GetPseudoRoot()
    if not start_prim:
        # Allow None prim_path for whole stage check, warn if path specified but not found
        if prim_path:
            carb.log_warn(f"Prim path not found: {prim_path}")
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
        carb.log_warn("Invalid stage, skipping label check")
        return incorrect_pairs

    start_prim = stage.GetPrimAtPath(prim_path) if prim_path else stage.GetPseudoRoot()
    if not start_prim:
        if prim_path:
            carb.log_warn(f"Prim path not found: {prim_path}")
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
    """Returns a dictionary of semantic labels (UsdSemantics.LabelsAPI) and their corresponding count.

    Args:
        prim_path (str | None): This will check Prim path and its childrens' labels. If None, checks the whole stage.

    Returns:
        dict[str, int]: Dictionary mapping individual labels to their total count across all instances.
                       Includes a 'missing_labels' count for meshes with no LabelsAPI.
    """
    labels_counter = {"missing_labels": 0}
    stage = get_current_stage()
    if stage is None:
        carb.log_warn("Invalid stage, skipping label check")
        return labels_counter

    start_prim = stage.GetPrimAtPath(prim_path) if prim_path else stage.GetPseudoRoot()
    if not start_prim:
        if prim_path:
            carb.log_warn(f"Prim path not found: {prim_path}")
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
    to the new UsdSemantics.LabelsAPI.

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
                carb.log_warn(
                    f"[upgrade_prim] Skipping instance '{old_instance_name}' on {current_prim.GetPath()} due to missing type or data."
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
                carb.log_warn(f"Failed to upgrade instance '{old_instance_name}' on {current_prim.GetPath()}: {e}")
                continue
    return total_upgraded
