# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities built around USD operations."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from .legacy import *  # noqa: F403
    from .prims import *  # noqa: F403
    from .queries import *  # noqa: F403
    from .semantics import *  # noqa: F403
    from .stage import *  # noqa: F403
    from .transforms import *  # noqa: F403

from isaaclab.utils.module import lazy_export

lazy_export(
    ("legacy", [
        "add_reference_to_stage",
        "get_stage_up_axis",
        "traverse_stage",
        "get_prim_at_path",
        "get_prim_path",
        "is_prim_path_valid",
        "define_prim",
        "get_prim_type_name",
        "get_next_free_path",
    ]),
    ("prims", [
        "create_prim",
        "delete_prim",
        "make_uninstanceable",
        "set_prim_visibility",
        "safe_set_attribute_on_usd_schema",
        "safe_set_attribute_on_usd_prim",
        "change_prim_property",
        "export_prim_to_file",
        "apply_nested",
        "clone",
        "bind_visual_material",
        "bind_physics_material",
        "add_usd_reference",
        "get_usd_references",
        "select_usd_variants",
    ]),
    ("queries", [
        "get_next_free_prim_path",
        "get_first_matching_ancestor_prim",
        "get_first_matching_child_prim",
        "get_all_matching_child_prims",
        "find_first_matching_prim",
        "find_matching_prims",
        "find_matching_prim_paths",
        "find_global_fixed_joint_prim",
    ]),
    ("semantics", [
        "add_labels",
        "get_labels",
        "remove_labels",
        "check_missing_labels",
        "count_total_labels",
    ]),
    ("stage", [
        "resolve_paths",
        "create_new_stage",
        "is_current_stage_in_memory",
        "open_stage",
        "use_stage",
        "update_stage",
        "save_stage",
        "close_stage",
        "clear_stage",
        "get_current_stage",
        "get_current_stage_id",
    ]),
    ("transforms", [
        "standardize_xform_ops",
        "validate_standard_xform_ops",
        "resolve_prim_pose",
        "resolve_prim_scale",
        "convert_world_pose_to_local",
    ]),
)
