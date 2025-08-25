# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Policy transfer utilities for handling joint ordering differences between physics engines.

This module provides functionality to handle policy transfer between different physics engines
that may have different joint orderings. It supports loading joint mappings from YAML files
and provides observation and action remapping functions.
"""

import yaml


def get_joint_mappings(args_cli, action_space_dim):
    """Get joint mappings based on command line arguments.

    Args:
            args_cli: Command line arguments
            action_space_dim: Dimension of the action space (number of joints)

    Returns:
            tuple: (source_to_target_list, target_to_source_list, source_to_target_obs_list)
    """
    num_joints = action_space_dim
    if args_cli.policy_transfer_file:
        # Load from YAML file
        try:
            with open(args_cli.policy_transfer_file) as file:
                config = yaml.safe_load(file)
        except Exception as e:
            raise RuntimeError(f"Failed to load joint mapping from {args_cli.policy_transfer_file}: {e}")

        source_joint_names = config["source_joint_names"]
        target_joint_names = config["target_joint_names"]
        # Find joint mapping
        source_to_target = []
        target_to_source = []

        # Create source to target mapping
        for joint_name in source_joint_names:
            if joint_name in target_joint_names:
                source_to_target.append(target_joint_names.index(joint_name))
            else:
                raise ValueError(f"Joint '{joint_name}' not found in target joint names")

        # Create target to source mapping
        for joint_name in target_joint_names:
            if joint_name in source_joint_names:
                target_to_source.append(source_joint_names.index(joint_name))
            else:
                raise ValueError(f"Joint '{joint_name}' not found in source joint names")
        print(f"[INFO] Loaded joint mapping for policy transfer from YAML: {args_cli.policy_transfer_file}")
        assert (
            len(source_to_target) == len(target_to_source) == num_joints
        ), "Number of source and target joints must match"
    else:
        # Use identity mapping (one-to-one)
        identity_map = list(range(num_joints))
        source_to_target, target_to_source = identity_map, identity_map

    # Create observation mapping (first 12 values stay the same for locomotion examples, then map joint-related values)
    obs_map = (
        [0, 1, 2]
        + [3, 4, 5]
        + [6, 7, 8]
        + [9, 10, 11]
        + [i + 12 + num_joints * 0 for i in source_to_target]
        + [i + 12 + num_joints * 1 for i in source_to_target]
        + [i + 12 + num_joints * 2 for i in source_to_target]
    )

    return source_to_target, target_to_source, obs_map
