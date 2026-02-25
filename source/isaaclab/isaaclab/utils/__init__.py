# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-package containing utilities for common operations and helper functions."""

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=[],
    submod_attrs={
        "array": [
            "TensorData",
            "TENSOR_TYPES",
            "TENSOR_TYPE_CONVERSIONS",
            "convert_to_torch",
        ],
        "buffers": [
            "CircularBuffer",
            "DelayBuffer",
            "TimestampedBuffer",
            "TimestampedBufferWarp",
        ],
        "configclass": [
            "configclass",
        ],
        "dict": [
            "class_to_dict",
            "update_class_from_dict",
            "dict_to_md5_hash",
            "convert_dict_to_backend",
            "update_dict",
            "replace_slices_with_strings",
            "replace_strings_with_slices",
            "print_dict",
        ],
        "interpolation": [
            "LinearInterpolation",
        ],
        "logger": [
            "configure_logging",
            "ColoredFormatter",
            "RateLimitFilter",
        ],
        "mesh": [
            "create_trimesh_from_geom_mesh",
            "create_trimesh_from_geom_shape",
            "convert_faces_to_triangles",
            "PRIMITIVE_MESH_TYPES",
        ],
        "modifiers": [
            "ModifierCfg",
            "ModifierBase",
            "DigitalFilter",
            "DigitalFilterCfg",
            "Integrator",
            "IntegratorCfg",
            "bias",
            "clip",
            "scale",
        ],
        "string": [
            "DeferredClass",
            "to_camel_case",
            "to_snake_case",
            "string_to_slice",
            "is_lambda_expression",
            "callable_to_string",
            "string_to_callable",
            "resolve_matching_names",
            "resolve_matching_names_values",
            "find_unique_string_name",
            "find_root_prim_path_from_regex",
        ],
        "timer": [
            "Timer",
            "TimerError",
        ],
        "types": [
            "ArticulationActions",
        ],
        "version": [
            "has_kit",
            "get_isaac_sim_version",
            "compare_versions",
        ],
    },
)
