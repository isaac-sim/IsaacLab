# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Public base-surface classification for shared asset contracts."""

from dataclasses import dataclass
from enum import StrEnum
from importlib import import_module

from isaaclab.assets.articulation.base_articulation import BaseArticulation
from isaaclab.assets.articulation.base_articulation_data import BaseArticulationData
from isaaclab.assets.asset_base import AssetBase
from isaaclab.assets.rigid_object.base_rigid_object import BaseRigidObject
from isaaclab.assets.rigid_object.base_rigid_object_data import BaseRigidObjectData
from isaaclab.assets.rigid_object_collection.base_rigid_object_collection import BaseRigidObjectCollection
from isaaclab.assets.rigid_object_collection.base_rigid_object_collection_data import BaseRigidObjectCollectionData


class ContractKind(StrEnum):
    """Supported classifications for a declared public asset member."""

    API = "api"
    DATA = "data"
    WRITE = "write"
    UNSUPPORTED = "unsupported"
    OUT_OF_SCOPE = "out_of_scope"


BASE_SURFACE_CLASSES = (
    AssetBase,
    BaseRigidObject,
    BaseRigidObjectData,
    BaseRigidObjectCollection,
    BaseRigidObjectCollectionData,
    BaseArticulation,
    BaseArticulationData,
)


_PUBLIC_MEMBER_SNAPSHOT = {
    "AssetBase": """
        assert_shape_and_dtype assert_shape_and_dtype_mask data device has_debug_vis_implementation is_initialized
        num_instances reset set_debug_vis set_visibility update write_data_to_sim
    """,
    "BaseRigidObject": """
        body_names data find_bodies instantaneous_wrench_composer num_bodies num_instances permanent_wrench_composer
        reset root_view set_coms set_coms_index set_coms_mask set_external_force_and_torque set_inertias
        set_inertias_index set_inertias_mask set_masses set_masses_index set_masses_mask update write_data_to_sim
        write_root_com_pose_to_sim write_root_com_pose_to_sim_index write_root_com_pose_to_sim_mask
        write_root_com_state_to_sim write_root_com_velocity_to_sim write_root_com_velocity_to_sim_index
        write_root_com_velocity_to_sim_mask write_root_link_pose_to_sim write_root_link_pose_to_sim_index
        write_root_link_pose_to_sim_mask write_root_link_state_to_sim write_root_link_velocity_to_sim
        write_root_link_velocity_to_sim_index write_root_link_velocity_to_sim_mask write_root_pose_to_sim
        write_root_pose_to_sim_index write_root_pose_to_sim_mask write_root_state_to_sim write_root_velocity_to_sim
        write_root_velocity_to_sim_index write_root_velocity_to_sim_mask
    """,
    "BaseRigidObjectData": """
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
    "BaseRigidObjectCollection": """
        body_names data find_bodies find_objects instantaneous_wrench_composer num_bodies num_instances num_objects
        object_names permanent_wrench_composer reset root_view set_coms set_coms_index set_coms_mask
        set_external_force_and_torque set_inertias set_inertias_index set_inertias_mask set_masses set_masses_index
        set_masses_mask update write_body_com_pose_to_sim write_body_com_pose_to_sim_index
        write_body_com_pose_to_sim_mask write_body_com_state_to_sim write_body_com_velocity_to_sim
        write_body_com_velocity_to_sim_index write_body_com_velocity_to_sim_mask write_body_link_pose_to_sim
        write_body_link_pose_to_sim_index write_body_link_pose_to_sim_mask write_body_link_state_to_sim
        write_body_link_velocity_to_sim write_body_link_velocity_to_sim_index write_body_link_velocity_to_sim_mask
        write_body_pose_to_sim write_body_pose_to_sim_index write_body_pose_to_sim_mask write_body_state_to_sim
        write_body_velocity_to_sim write_body_velocity_to_sim_index write_body_velocity_to_sim_mask write_data_to_sim
        write_object_com_pose_to_sim write_object_com_state_to_sim write_object_com_velocity_to_sim
        write_object_link_pose_to_sim write_object_link_state_to_sim write_object_link_velocity_to_sim
        write_object_pose_to_sim write_object_state_to_sim write_object_velocity_to_sim
    """,
    "BaseRigidObjectCollectionData": """
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
    "BaseArticulation": """
        backend_body_names backend_joint_names body_names body_ordering data find_bodies find_fixed_tendons find_joints
        find_spatial_tendons fixed_tendon_names instantaneous_wrench_composer is_fixed_base joint_names joint_ordering
        map_body_ids_to_backend map_joint_ids_to_backend num_base_dofs num_bodies num_fixed_tendons num_instances
        num_joints num_spatial_tendons permanent_wrench_composer reset root_view set_coms set_coms_index set_coms_mask
        set_external_force_and_torque set_fixed_tendon_damping set_fixed_tendon_damping_index
        set_fixed_tendon_damping_mask set_fixed_tendon_limit set_fixed_tendon_limit_stiffness
        set_fixed_tendon_limit_stiffness_index set_fixed_tendon_limit_stiffness_mask set_fixed_tendon_offset
        set_fixed_tendon_offset_index set_fixed_tendon_offset_mask set_fixed_tendon_position_limit
        set_fixed_tendon_position_limit_index set_fixed_tendon_position_limit_mask set_fixed_tendon_rest_length
        set_fixed_tendon_rest_length_index set_fixed_tendon_rest_length_mask set_fixed_tendon_stiffness
        set_fixed_tendon_stiffness_index set_fixed_tendon_stiffness_mask set_inertias set_inertias_index
        set_inertias_mask set_joint_effort_target set_joint_effort_target_index set_joint_effort_target_mask
        set_joint_position_target set_joint_position_target_index set_joint_position_target_mask
        set_joint_velocity_target set_joint_velocity_target_index set_joint_velocity_target_mask set_masses
        set_masses_index set_masses_mask set_spatial_tendon_damping set_spatial_tendon_damping_index
        set_spatial_tendon_damping_mask set_spatial_tendon_limit_stiffness
        set_spatial_tendon_limit_stiffness_index set_spatial_tendon_limit_stiffness_mask set_spatial_tendon_offset
        set_spatial_tendon_offset_index set_spatial_tendon_offset_mask set_spatial_tendon_stiffness
        set_spatial_tendon_stiffness_index set_spatial_tendon_stiffness_mask spatial_tendon_names update
        write_data_to_sim write_fixed_tendon_properties_to_sim write_fixed_tendon_properties_to_sim_index
        write_fixed_tendon_properties_to_sim_mask write_joint_armature_to_sim write_joint_armature_to_sim_index
        write_joint_armature_to_sim_mask write_joint_damping_to_sim write_joint_damping_to_sim_index
        write_joint_damping_to_sim_mask write_joint_effort_limit_to_sim write_joint_effort_limit_to_sim_index
        write_joint_effort_limit_to_sim_mask write_joint_friction_coefficient_to_sim
        write_joint_friction_coefficient_to_sim_index write_joint_friction_coefficient_to_sim_mask
        write_joint_friction_to_sim write_joint_limits_to_sim write_joint_position_limit_to_sim
        write_joint_position_limit_to_sim_index write_joint_position_limit_to_sim_mask write_joint_position_to_sim
        write_joint_position_to_sim_index write_joint_position_to_sim_mask write_joint_state_to_sim
        write_joint_stiffness_to_sim write_joint_stiffness_to_sim_index write_joint_stiffness_to_sim_mask
        write_joint_velocity_limit_to_sim write_joint_velocity_limit_to_sim_index write_joint_velocity_limit_to_sim_mask
        write_joint_velocity_to_sim write_joint_velocity_to_sim_index write_joint_velocity_to_sim_mask
        write_root_com_pose_to_sim write_root_com_pose_to_sim_index write_root_com_pose_to_sim_mask
        write_root_com_state_to_sim write_root_com_velocity_to_sim write_root_com_velocity_to_sim_index
        write_root_com_velocity_to_sim_mask write_root_link_pose_to_sim write_root_link_pose_to_sim_index
        write_root_link_pose_to_sim_mask write_root_link_state_to_sim write_root_link_velocity_to_sim
        write_root_link_velocity_to_sim_index write_root_link_velocity_to_sim_mask write_root_pose_to_sim
        write_root_pose_to_sim_index write_root_pose_to_sim_mask write_root_state_to_sim write_root_velocity_to_sim
        write_root_velocity_to_sim_index write_root_velocity_to_sim_mask write_spatial_tendon_properties_to_sim
        write_spatial_tendon_properties_to_sim_index write_spatial_tendon_properties_to_sim_mask
    """,
    "BaseArticulationData": """
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
}


@dataclass(frozen=True)
class PublicMemberContract:
    """Map one public member to a concrete contract target or justified exclusion."""

    kind: ContractKind
    target: str | None = None
    reason: str | None = None


@dataclass(frozen=True)
class PublicSurfaceAudit:
    """Describe mismatches between declared members and their explicit mappings."""

    missing: frozenset[str]
    stale: frozenset[str]
    unreasoned: frozenset[str]
    untargeted: frozenset[str]

    @property
    def is_valid(self) -> bool:
        """Return whether the public inventory and mappings agree exactly."""
        return not (self.missing or self.stale or self.unreasoned or self.untargeted)

    def format_errors(self) -> str:
        """Format all mapping failures for one actionable pytest diagnostic."""
        return "; ".join(
            f"{label}: {sorted(values)}"
            for label, values in (
                ("missing", self.missing),
                ("stale", self.stale),
                ("unreasoned exclusions", self.unreasoned),
                ("covered members without targets", self.untargeted),
            )
            if values
        )


def _mapping(
    class_name: str,
    members: str,
    kind: ContractKind,
    *,
    target: str | None = None,
    reason: str | None = None,
) -> dict[str, PublicMemberContract]:
    """Build explicit qualified mappings for one reviewed member group."""
    return {
        f"{class_name}.{member_name}": PublicMemberContract(kind=kind, target=target, reason=reason)
        for member_name in members.split()
    }


_ARTICULATION_API_CONTRACT = "contract._articulation_contract_cases.TestArticulationProperties"
_ARTICULATION_DATA_CONTRACT = "contract._articulation_contract_cases.TestArticulationDataRootState"
_ARTICULATION_WRITE_CONTRACT = "contract._articulation_contract_cases.TestArticulationWritersRoot"
_COLLECTION_API_CONTRACT = "contract._rigid_object_collection_contract_cases.TestCollectionProperties"
_COLLECTION_DATA_CONTRACT = "contract._rigid_object_collection_contract_cases.TestCollectionDataBodyState"
_COLLECTION_WRITE_CONTRACT = "contract._rigid_object_collection_contract_cases.TestCollectionWritersPose"
_RIGID_OBJECT_API_CONTRACT = "contract._rigid_object_contract_cases.TestRigidObjectProperties"
_RIGID_OBJECT_DATA_CONTRACT = "contract._rigid_object_contract_cases.TestRigidObjectDataRootState"
_RIGID_OBJECT_WRITE_CONTRACT = "contract._rigid_object_contract_cases.TestRigidObjectWritersRoot"

PUBLIC_SURFACE_CONTRACTS = {
    **_mapping(
        "AssetBase",
        "data device is_initialized num_instances",
        ContractKind.API,
        target=_ARTICULATION_API_CONTRACT,
    ),
    **_mapping(
        "AssetBase",
        "assert_shape_and_dtype assert_shape_and_dtype_mask reset update write_data_to_sim",
        ContractKind.WRITE,
        target=_ARTICULATION_WRITE_CONTRACT,
    ),
    **_mapping(
        "AssetBase",
        "has_debug_vis_implementation set_debug_vis set_visibility",
        ContractKind.OUT_OF_SCOPE,
        reason="debug visualization and stage visibility require a live USD/Kit application",
    ),
    **_mapping(
        "BaseRigidObject",
        """
            body_names data find_bodies instantaneous_wrench_composer num_bodies num_instances
            permanent_wrench_composer root_view
        """,
        ContractKind.API,
        target=_RIGID_OBJECT_API_CONTRACT,
    ),
    **_mapping(
        "BaseRigidObject",
        """
            reset set_coms set_coms_index set_coms_mask set_external_force_and_torque set_inertias
            set_inertias_index set_inertias_mask set_masses set_masses_index set_masses_mask update
            write_data_to_sim write_root_com_pose_to_sim write_root_com_pose_to_sim_index
            write_root_com_pose_to_sim_mask write_root_com_state_to_sim write_root_com_velocity_to_sim
            write_root_com_velocity_to_sim_index write_root_com_velocity_to_sim_mask write_root_link_pose_to_sim
            write_root_link_pose_to_sim_index write_root_link_pose_to_sim_mask write_root_link_state_to_sim
            write_root_link_velocity_to_sim write_root_link_velocity_to_sim_index
            write_root_link_velocity_to_sim_mask write_root_pose_to_sim write_root_pose_to_sim_index
            write_root_pose_to_sim_mask write_root_state_to_sim write_root_velocity_to_sim
            write_root_velocity_to_sim_index write_root_velocity_to_sim_mask
        """,
        ContractKind.WRITE,
        target=_RIGID_OBJECT_WRITE_CONTRACT,
    ),
    **_mapping(
        "BaseRigidObjectData",
        _PUBLIC_MEMBER_SNAPSHOT["BaseRigidObjectData"],
        ContractKind.DATA,
        target=_RIGID_OBJECT_DATA_CONTRACT,
    ),
    **_mapping(
        "BaseRigidObjectCollection",
        """
            body_names data find_bodies find_objects instantaneous_wrench_composer num_bodies num_instances
            num_objects object_names permanent_wrench_composer root_view
        """,
        ContractKind.API,
        target=_COLLECTION_API_CONTRACT,
    ),
    **_mapping(
        "BaseRigidObjectCollection",
        """
            reset set_coms set_coms_index set_coms_mask set_external_force_and_torque set_inertias
            set_inertias_index set_inertias_mask set_masses set_masses_index set_masses_mask update
            write_body_com_pose_to_sim write_body_com_pose_to_sim_index write_body_com_pose_to_sim_mask
            write_body_com_state_to_sim write_body_com_velocity_to_sim write_body_com_velocity_to_sim_index
            write_body_com_velocity_to_sim_mask write_body_link_pose_to_sim write_body_link_pose_to_sim_index
            write_body_link_pose_to_sim_mask write_body_link_state_to_sim write_body_link_velocity_to_sim
            write_body_link_velocity_to_sim_index write_body_link_velocity_to_sim_mask write_body_pose_to_sim
            write_body_pose_to_sim_index write_body_pose_to_sim_mask write_body_state_to_sim
            write_body_velocity_to_sim write_body_velocity_to_sim_index write_body_velocity_to_sim_mask
            write_data_to_sim write_object_com_pose_to_sim write_object_com_state_to_sim
            write_object_com_velocity_to_sim write_object_link_pose_to_sim write_object_link_state_to_sim
            write_object_link_velocity_to_sim write_object_pose_to_sim write_object_state_to_sim
            write_object_velocity_to_sim
        """,
        ContractKind.WRITE,
        target=_COLLECTION_WRITE_CONTRACT,
    ),
    **_mapping(
        "BaseRigidObjectCollectionData",
        _PUBLIC_MEMBER_SNAPSHOT["BaseRigidObjectCollectionData"],
        ContractKind.DATA,
        target=_COLLECTION_DATA_CONTRACT,
    ),
    **_mapping(
        "BaseArticulation",
        """
            backend_body_names backend_joint_names body_names body_ordering data find_bodies find_fixed_tendons
            find_joints find_spatial_tendons fixed_tendon_names instantaneous_wrench_composer is_fixed_base
            joint_names joint_ordering map_body_ids_to_backend map_joint_ids_to_backend num_base_dofs num_bodies
            num_fixed_tendons num_instances num_joints num_spatial_tendons permanent_wrench_composer root_view
            spatial_tendon_names
        """,
        ContractKind.API,
        target=_ARTICULATION_API_CONTRACT,
    ),
    **_mapping(
        "BaseArticulation",
        """
            reset set_coms set_coms_index set_coms_mask set_external_force_and_torque set_fixed_tendon_damping
            set_fixed_tendon_damping_index set_fixed_tendon_damping_mask set_fixed_tendon_limit
            set_fixed_tendon_limit_stiffness set_fixed_tendon_limit_stiffness_index
            set_fixed_tendon_limit_stiffness_mask set_fixed_tendon_offset set_fixed_tendon_offset_index
            set_fixed_tendon_offset_mask set_fixed_tendon_position_limit set_fixed_tendon_position_limit_index
            set_fixed_tendon_position_limit_mask set_fixed_tendon_rest_length set_fixed_tendon_rest_length_index
            set_fixed_tendon_rest_length_mask set_fixed_tendon_stiffness set_fixed_tendon_stiffness_index
            set_fixed_tendon_stiffness_mask set_inertias set_inertias_index set_inertias_mask
            set_joint_effort_target set_joint_effort_target_index set_joint_effort_target_mask
            set_joint_position_target set_joint_position_target_index set_joint_position_target_mask
            set_joint_velocity_target set_joint_velocity_target_index set_joint_velocity_target_mask set_masses
            set_masses_index set_masses_mask set_spatial_tendon_damping set_spatial_tendon_damping_index
            set_spatial_tendon_damping_mask set_spatial_tendon_limit_stiffness
            set_spatial_tendon_limit_stiffness_index set_spatial_tendon_limit_stiffness_mask
            set_spatial_tendon_offset set_spatial_tendon_offset_index set_spatial_tendon_offset_mask
            set_spatial_tendon_stiffness set_spatial_tendon_stiffness_index set_spatial_tendon_stiffness_mask update
            write_data_to_sim write_fixed_tendon_properties_to_sim write_fixed_tendon_properties_to_sim_index
            write_fixed_tendon_properties_to_sim_mask write_joint_armature_to_sim write_joint_armature_to_sim_index
            write_joint_armature_to_sim_mask write_joint_damping_to_sim write_joint_damping_to_sim_index
            write_joint_damping_to_sim_mask write_joint_effort_limit_to_sim write_joint_effort_limit_to_sim_index
            write_joint_effort_limit_to_sim_mask write_joint_friction_coefficient_to_sim
            write_joint_friction_coefficient_to_sim_index write_joint_friction_coefficient_to_sim_mask
            write_joint_friction_to_sim write_joint_limits_to_sim write_joint_position_limit_to_sim
            write_joint_position_limit_to_sim_index write_joint_position_limit_to_sim_mask
            write_joint_position_to_sim write_joint_position_to_sim_index write_joint_position_to_sim_mask
            write_joint_state_to_sim write_joint_stiffness_to_sim write_joint_stiffness_to_sim_index
            write_joint_stiffness_to_sim_mask write_joint_velocity_limit_to_sim
            write_joint_velocity_limit_to_sim_index write_joint_velocity_limit_to_sim_mask
            write_joint_velocity_to_sim write_joint_velocity_to_sim_index write_joint_velocity_to_sim_mask
            write_root_com_pose_to_sim write_root_com_pose_to_sim_index write_root_com_pose_to_sim_mask
            write_root_com_state_to_sim write_root_com_velocity_to_sim write_root_com_velocity_to_sim_index
            write_root_com_velocity_to_sim_mask write_root_link_pose_to_sim write_root_link_pose_to_sim_index
            write_root_link_pose_to_sim_mask write_root_link_state_to_sim write_root_link_velocity_to_sim
            write_root_link_velocity_to_sim_index write_root_link_velocity_to_sim_mask write_root_pose_to_sim
            write_root_pose_to_sim_index write_root_pose_to_sim_mask write_root_state_to_sim
            write_root_velocity_to_sim write_root_velocity_to_sim_index write_root_velocity_to_sim_mask
            write_spatial_tendon_properties_to_sim write_spatial_tendon_properties_to_sim_index
            write_spatial_tendon_properties_to_sim_mask
        """,
        ContractKind.WRITE,
        target=_ARTICULATION_WRITE_CONTRACT,
    ),
    **_mapping(
        "BaseArticulationData",
        _PUBLIC_MEMBER_SNAPSHOT["BaseArticulationData"],
        ContractKind.DATA,
        target=_ARTICULATION_DATA_CONTRACT,
    ),
}

PUBLIC_SURFACE_CLASSIFICATIONS = {
    member_name: contract.kind for member_name, contract in PUBLIC_SURFACE_CONTRACTS.items()
}


def audit_public_surface(classes: tuple[type, ...], mappings: dict[str, PublicMemberContract]) -> PublicSurfaceAudit:
    """Compare the current declared inventory with exact, reasoned mappings."""
    declared_members = {
        f"{cls.__name__}.{member_name}"
        for cls in classes
        for member_name in cls.__dict__
        if not member_name.startswith("_")
    }
    mapped_members = set(mappings)
    exclusion_kinds = {ContractKind.UNSUPPORTED, ContractKind.OUT_OF_SCOPE}
    covered_kinds = {ContractKind.API, ContractKind.DATA, ContractKind.WRITE}
    return PublicSurfaceAudit(
        missing=frozenset(declared_members - mapped_members),
        stale=frozenset(mapped_members - declared_members),
        unreasoned=frozenset(
            member_name
            for member_name, contract in mappings.items()
            if contract.kind in exclusion_kinds and not contract.reason
        ),
        untargeted=frozenset(
            member_name
            for member_name, contract in mappings.items()
            if contract.kind in covered_kinds and not contract.target
        ),
    )


def unresolved_contract_targets(mappings: dict[str, PublicMemberContract]) -> frozenset[str]:
    """Return concrete covered-contract targets that cannot be imported."""
    unresolved = set()
    for contract in mappings.values():
        if contract.target is None:
            continue
        try:
            import_module(contract.target)
        except ModuleNotFoundError:
            module_name, _, attribute = contract.target.rpartition(".")
            try:
                module = import_module(module_name)
            except ModuleNotFoundError:
                unresolved.add(contract.target)
            else:
                if not hasattr(module, attribute):
                    unresolved.add(contract.target)
    return frozenset(unresolved)


def unclassified_public_members(classes: tuple[type, ...], classifications: dict[str, ContractKind]) -> set[str]:
    """Return declared public members that have no contract classification."""
    declared_members = {
        f"{cls.__name__}.{member_name}"
        for cls in classes
        for member_name in cls.__dict__
        if not member_name.startswith("_")
    }
    return declared_members - classifications.keys()
