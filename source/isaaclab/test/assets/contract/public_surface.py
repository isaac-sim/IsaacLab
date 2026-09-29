# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Public base-class surface classified by the shared contract that owns it.

:data:`PUBLIC_SURFACE_CONTRACTS` is the single inventory of the public members declared by the asset base classes.
``test_base_public_surface_has_an_explicit_contract_classification`` diffs it against the classes, so adding a public
member forces its contract classification and removing one is reported as a stale entry.
"""

from dataclasses import dataclass
from enum import StrEnum

from isaaclab.assets.articulation.base_articulation import BaseArticulation
from isaaclab.assets.articulation.base_articulation_data import BaseArticulationData
from isaaclab.assets.asset_base import AssetBase
from isaaclab.assets.rigid_object.base_rigid_object import BaseRigidObject
from isaaclab.assets.rigid_object.base_rigid_object_data import BaseRigidObjectData
from isaaclab.assets.rigid_object_collection.base_rigid_object_collection import BaseRigidObjectCollection
from isaaclab.assets.rigid_object_collection.base_rigid_object_collection_data import BaseRigidObjectCollectionData


class ContractKind(StrEnum):
    """Contract family that owns a public member."""

    API = "api"
    DATA = "data"
    WRITE = "write"
    OUT_OF_SCOPE = "out_of_scope"


@dataclass(frozen=True)
class PublicMemberContract:
    """Classify one public member, with the reason for an out-of-scope exclusion."""

    kind: ContractKind
    reason: str | None = None


BASE_SURFACE_CLASSES = (
    AssetBase,
    BaseRigidObject,
    BaseRigidObjectData,
    BaseRigidObjectCollection,
    BaseRigidObjectCollectionData,
    BaseArticulation,
    BaseArticulationData,
)


def _mapping(class_name: str, members: str, kind: ContractKind, reason: str | None = None):
    """Classify a whitespace-separated group of members declared by one class."""
    return {f"{class_name}.{member}": PublicMemberContract(kind, reason) for member in members.split()}


PUBLIC_SURFACE_CONTRACTS = {
    **_mapping("AssetBase", "data device is_initialized num_instances", ContractKind.API),
    **_mapping(
        "AssetBase",
        "assert_shape_and_dtype assert_shape_and_dtype_mask reset update write_data_to_sim",
        ContractKind.WRITE,
    ),
    **_mapping(
        "AssetBase",
        "has_debug_vis_implementation set_debug_vis",
        ContractKind.OUT_OF_SCOPE,
        reason="debug visualization requires a live USD stage and Kit application",
    ),
    **_mapping(
        "BaseRigidObject",
        """
        body_names data find_bodies instantaneous_wrench_composer num_bodies num_instances permanent_wrench_composer
        root_view
        """,
        ContractKind.API,
    ),
    **_mapping(
        "BaseRigidObject",
        """
        reset set_coms set_coms_index set_coms_mask set_external_force_and_torque set_inertias set_inertias_index
        set_inertias_mask set_masses set_masses_index set_masses_mask update write_data_to_sim
        write_root_com_pose_to_sim write_root_com_pose_to_sim_index write_root_com_pose_to_sim_mask
        write_root_com_state_to_sim write_root_com_velocity_to_sim write_root_com_velocity_to_sim_index
        write_root_com_velocity_to_sim_mask write_root_link_pose_to_sim write_root_link_pose_to_sim_index
        write_root_link_pose_to_sim_mask write_root_link_state_to_sim write_root_link_velocity_to_sim
        write_root_link_velocity_to_sim_index write_root_link_velocity_to_sim_mask write_root_pose_to_sim
        write_root_pose_to_sim_index write_root_pose_to_sim_mask write_root_state_to_sim write_root_velocity_to_sim
        write_root_velocity_to_sim_index write_root_velocity_to_sim_mask
        """,
        ContractKind.WRITE,
    ),
    **_mapping(
        "BaseRigidObjectData",
        """
        body_acc_w body_ang_acc_w body_ang_vel_w body_com_acc_w body_com_ang_acc_w body_com_ang_vel_w
        body_com_lin_acc_w body_com_lin_vel_w body_com_pos_b body_com_pos_w body_com_pose_b body_com_pose_w
        body_com_quat_b body_com_quat_w body_com_state_w body_com_vel_w body_inertia body_lin_acc_w body_lin_vel_w
        body_link_ang_vel_w body_link_lin_vel_w body_link_pos_w body_link_pose_w body_link_quat_w body_link_state_w
        body_link_vel_w body_mass body_names body_pos_w body_pose_w body_quat_w body_state_w body_vel_w com_pos_b
        com_quat_b default_inertia default_mass default_root_pose default_root_state default_root_vel heading_w
        projected_gravity_b root_ang_vel_b root_ang_vel_w root_com_ang_vel_b root_com_ang_vel_w root_com_lin_vel_b
        root_com_lin_vel_w root_com_pos_w root_com_pose_w root_com_quat_w root_com_state_w root_com_vel_w
        root_lin_vel_b root_lin_vel_w root_link_ang_vel_b root_link_ang_vel_w root_link_lin_vel_b root_link_lin_vel_w
        root_link_pos_w root_link_pose_w root_link_quat_w root_link_state_w root_link_vel_w root_pos_w root_pose_w
        root_quat_w root_state_w root_vel_w update
        """,
        ContractKind.DATA,
    ),
    **_mapping(
        "BaseRigidObjectCollection",
        """
        body_names data find_bodies find_objects instantaneous_wrench_composer num_bodies num_instances num_objects
        object_names permanent_wrench_composer root_view
        """,
        ContractKind.API,
    ),
    **_mapping(
        "BaseRigidObjectCollection",
        """
        reset set_coms set_coms_index set_coms_mask set_external_force_and_torque set_inertias set_inertias_index
        set_inertias_mask set_masses set_masses_index set_masses_mask update write_body_com_pose_to_sim
        write_body_com_pose_to_sim_index write_body_com_pose_to_sim_mask write_body_com_state_to_sim
        write_body_com_velocity_to_sim write_body_com_velocity_to_sim_index write_body_com_velocity_to_sim_mask
        write_body_link_pose_to_sim write_body_link_pose_to_sim_index write_body_link_pose_to_sim_mask
        write_body_link_state_to_sim write_body_link_velocity_to_sim write_body_link_velocity_to_sim_index
        write_body_link_velocity_to_sim_mask write_body_pose_to_sim write_body_pose_to_sim_index
        write_body_pose_to_sim_mask write_body_state_to_sim write_body_velocity_to_sim write_body_velocity_to_sim_index
        write_body_velocity_to_sim_mask write_data_to_sim write_object_com_pose_to_sim write_object_com_state_to_sim
        write_object_com_velocity_to_sim write_object_link_pose_to_sim write_object_link_state_to_sim
        write_object_link_velocity_to_sim write_object_pose_to_sim write_object_state_to_sim
        write_object_velocity_to_sim
        """,
        ContractKind.WRITE,
    ),
    **_mapping(
        "BaseRigidObjectCollectionData",
        """
        body_acc_w body_ang_acc_w body_ang_vel_w body_com_acc_w body_com_ang_acc_w body_com_ang_vel_b
        body_com_ang_vel_w body_com_lin_acc_w body_com_lin_vel_b body_com_lin_vel_w body_com_pos_b body_com_pos_w
        body_com_pose_b body_com_pose_w body_com_quat_b body_com_quat_w body_com_state_w body_com_vel_w body_inertia
        body_lin_acc_w body_lin_vel_w body_link_ang_vel_b body_link_ang_vel_w body_link_lin_vel_b body_link_lin_vel_w
        body_link_pos_w body_link_pose_w body_link_quat_w body_link_state_w body_link_vel_w body_mass body_names
        body_pos_w body_pose_w body_quat_w body_state_w body_vel_w com_pos_b com_quat_b default_body_pose
        default_body_state default_body_vel default_inertia default_mass default_object_pose default_object_state
        default_object_vel heading_w object_acc_w object_ang_acc_w object_ang_vel_b object_ang_vel_w object_com_acc_w
        object_com_ang_acc_w object_com_ang_vel_b object_com_ang_vel_w object_com_lin_acc_w object_com_lin_vel_b
        object_com_lin_vel_w object_com_pos_b object_com_pos_w object_com_pose_b object_com_pose_w object_com_quat_b
        object_com_quat_w object_com_state_w object_com_vel_w object_lin_acc_w object_lin_vel_b object_lin_vel_w
        object_link_ang_vel_b object_link_ang_vel_w object_link_lin_vel_b object_link_lin_vel_w object_link_pos_w
        object_link_pose_w object_link_quat_w object_link_state_w object_link_vel_w object_pos_w object_pose_w
        object_quat_w object_state_w object_vel_w projected_gravity_b update
        """,
        ContractKind.DATA,
    ),
    **_mapping(
        "BaseArticulation",
        """
        backend_body_names backend_joint_names body_names body_ordering data find_bodies find_fixed_tendons find_joints
        find_spatial_tendons fixed_tendon_names instantaneous_wrench_composer is_fixed_base joint_names joint_ordering
        map_body_ids_to_backend map_joint_ids_to_backend num_base_dofs num_bodies num_fixed_tendons num_instances
        num_joints num_spatial_tendons permanent_wrench_composer root_view spatial_tendon_names
        """,
        ContractKind.API,
    ),
    **_mapping(
        "BaseArticulation",
        """
        reset set_coms set_coms_index set_coms_mask set_external_force_and_torque set_fixed_tendon_damping
        set_fixed_tendon_damping_index set_fixed_tendon_damping_mask set_fixed_tendon_limit
        set_fixed_tendon_limit_stiffness set_fixed_tendon_limit_stiffness_index set_fixed_tendon_limit_stiffness_mask
        set_fixed_tendon_offset set_fixed_tendon_offset_index set_fixed_tendon_offset_mask
        set_fixed_tendon_position_limit set_fixed_tendon_position_limit_index set_fixed_tendon_position_limit_mask
        set_fixed_tendon_position_target_index set_fixed_tendon_position_target_mask set_fixed_tendon_rest_length
        set_fixed_tendon_rest_length_index set_fixed_tendon_rest_length_mask set_fixed_tendon_stiffness
        set_fixed_tendon_stiffness_index set_fixed_tendon_stiffness_mask set_inertias set_inertias_index
        set_inertias_mask set_joint_effort_target set_joint_effort_target_index set_joint_effort_target_mask
        set_joint_position_target set_joint_position_target_index set_joint_position_target_mask
        set_joint_velocity_target set_joint_velocity_target_index set_joint_velocity_target_mask set_masses
        set_masses_index set_masses_mask set_spatial_tendon_damping set_spatial_tendon_damping_index
        set_spatial_tendon_damping_mask set_spatial_tendon_limit_stiffness set_spatial_tendon_limit_stiffness_index
        set_spatial_tendon_limit_stiffness_mask set_spatial_tendon_offset set_spatial_tendon_offset_index
        set_spatial_tendon_offset_mask set_spatial_tendon_stiffness set_spatial_tendon_stiffness_index
        set_spatial_tendon_stiffness_mask update write_data_to_sim write_fixed_tendon_properties_to_sim
        write_fixed_tendon_properties_to_sim_index write_fixed_tendon_properties_to_sim_mask
        write_joint_armature_to_sim write_joint_armature_to_sim_index write_joint_armature_to_sim_mask
        write_joint_damping_to_sim write_joint_damping_to_sim_index write_joint_damping_to_sim_mask
        write_joint_effort_limit_to_sim write_joint_effort_limit_to_sim_index write_joint_effort_limit_to_sim_mask
        write_joint_friction_coefficient_to_sim write_joint_friction_coefficient_to_sim_index
        write_joint_friction_coefficient_to_sim_mask write_joint_friction_to_sim write_joint_limits_to_sim
        write_joint_position_limit_to_sim write_joint_position_limit_to_sim_index
        write_joint_position_limit_to_sim_mask write_joint_position_to_sim write_joint_position_to_sim_index
        write_joint_position_to_sim_mask write_joint_state_to_sim write_joint_stiffness_to_sim
        write_joint_stiffness_to_sim_index write_joint_stiffness_to_sim_mask write_joint_velocity_limit_to_sim
        write_joint_velocity_limit_to_sim_index write_joint_velocity_limit_to_sim_mask write_joint_velocity_to_sim
        write_joint_velocity_to_sim_index write_joint_velocity_to_sim_mask write_root_com_pose_to_sim
        write_root_com_pose_to_sim_index write_root_com_pose_to_sim_mask write_root_com_state_to_sim
        write_root_com_velocity_to_sim write_root_com_velocity_to_sim_index write_root_com_velocity_to_sim_mask
        write_root_link_pose_to_sim write_root_link_pose_to_sim_index write_root_link_pose_to_sim_mask
        write_root_link_state_to_sim write_root_link_velocity_to_sim write_root_link_velocity_to_sim_index
        write_root_link_velocity_to_sim_mask write_root_pose_to_sim write_root_pose_to_sim_index
        write_root_pose_to_sim_mask write_root_state_to_sim write_root_velocity_to_sim write_root_velocity_to_sim_index
        write_root_velocity_to_sim_mask write_spatial_tendon_properties_to_sim
        write_spatial_tendon_properties_to_sim_index write_spatial_tendon_properties_to_sim_mask
        """,
        ContractKind.WRITE,
    ),
    **_mapping(
        "BaseArticulationData",
        """
        applied_torque bind_actuator_collection body_acc_w body_ang_acc_w body_ang_vel_w body_com_acc_w
        body_com_ang_acc_w body_com_ang_vel_w body_com_jacobian_w body_com_lin_acc_w body_com_lin_vel_w body_com_pos_b
        body_com_pos_w body_com_pose_b body_com_pose_w body_com_quat_b body_com_quat_w body_com_state_w body_com_vel_w
        body_inertia body_lin_acc_w body_lin_vel_w body_link_ang_vel_w body_link_jacobian_w body_link_lin_vel_w
        body_link_pos_w body_link_pose_w body_link_quat_w body_link_state_w body_link_vel_w body_mass body_names
        body_ordering body_pos_w body_pose_w body_quat_w body_state_w body_vel_w com_pos_b com_quat_b computed_torque
        default_fixed_tendon_damping default_fixed_tendon_limit default_fixed_tendon_limit_stiffness
        default_fixed_tendon_offset default_fixed_tendon_pos_limits default_fixed_tendon_rest_length
        default_fixed_tendon_stiffness default_inertia default_joint_armature default_joint_damping
        default_joint_friction default_joint_friction_coeff default_joint_limits default_joint_pos
        default_joint_pos_limits default_joint_stiffness default_joint_vel default_joint_viscous_friction_coeff
        default_mass default_root_pose default_root_state default_root_vel default_spatial_tendon_damping
        default_spatial_tendon_limit_stiffness default_spatial_tendon_offset default_spatial_tendon_stiffness
        fixed_tendon_damping fixed_tendon_limit fixed_tendon_limit_stiffness fixed_tendon_names fixed_tendon_offset
        fixed_tendon_pos_limits fixed_tendon_rest_length fixed_tendon_stiffness gravity_compensation_forces
        has_body_ordering has_joint_ordering heading_w joint_acc joint_armature joint_damping joint_effort_limits
        joint_effort_target joint_friction joint_friction_coeff joint_limits joint_names joint_ordering joint_pos
        joint_pos_limits joint_pos_target joint_stiffness joint_vel joint_vel_limits joint_vel_target
        joint_velocity_limits mass_matrix projected_gravity_b root_ang_vel_b root_ang_vel_w root_com_ang_vel_b
        root_com_ang_vel_w root_com_lin_vel_b root_com_lin_vel_w root_com_pos_w root_com_pose_w root_com_quat_w
        root_com_state_w root_com_vel_w root_lin_vel_b root_lin_vel_w root_link_ang_vel_b root_link_ang_vel_w
        root_link_lin_vel_b root_link_lin_vel_w root_link_pos_w root_link_pose_w root_link_quat_w root_link_state_w
        root_link_vel_w root_pos_w root_pose_w root_quat_w root_state_w root_vel_w soft_joint_pos_limits
        soft_joint_vel_limits spatial_tendon_damping spatial_tendon_limit_stiffness spatial_tendon_names
        spatial_tendon_offset spatial_tendon_stiffness update
        """,
        ContractKind.DATA,
    ),
}


def public_surface_mismatches(classes: tuple[type, ...], contracts: dict[str, PublicMemberContract]) -> dict[str, list]:
    """Diff the public members the classes declare against their contract classifications.

    Args:
        classes: Base classes whose own public members form the audited surface.
        contracts: Qualified member name to contract classification.

    Returns:
        Sorted ``missing`` (declared but unclassified), ``stale`` (classified but no longer declared), and
        ``unreasoned`` (out-of-scope without a reason) member names; all empty when the inventory is exact.
    """
    declared = {f"{cls.__name__}.{name}" for cls in classes for name in vars(cls) if not name.startswith("_")}
    return {
        "missing": sorted(declared - contracts.keys()),
        "stale": sorted(contracts.keys() - declared),
        "unreasoned": sorted(
            name
            for name, contract in contracts.items()
            if contract.kind is ContractKind.OUT_OF_SCOPE and not contract.reason
        ),
    }
