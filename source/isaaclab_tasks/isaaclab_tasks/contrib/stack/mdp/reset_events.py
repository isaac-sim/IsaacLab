# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Collision-free phase reset states for Franka cube-stack reinforcement learning."""

from __future__ import annotations

import math
from collections.abc import Sequence
from enum import IntEnum
from itertools import permutations
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import EventTermCfg, ManagerTermBase
from isaaclab.utils.math import matrix_from_quat, normalize, quat_mul

from ..constants import FRANKA_STACK_ARM_WORKSPACE_LOWER, FRANKA_STACK_ARM_WORKSPACE_UPPER
from .runtime_state import create_stack_reset_runtime_state

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv


# These center-column seeds were generated with constrained Franka
# differential IK against the active Newton articulation. Rows are workspace
# anchors; columns are
# near-grasp, pre-grasp, lifted/aligned, final-aligned, and release heights.
# The gripper tool poses are 0.04, 0.08, 0.14, 0.17, and 0.1175 m above the
# table with a common downward-facing tool orientation.
_ARM_POSES = (
    (
        (-0.1572749, 0.4348711, -0.1066657, -2.3711381, 0.1346695, 2.8009815, 0.4043023),
        (-0.1275118, 0.3274296, -0.1422084, -2.3924634, 0.1102695, 2.7145238, 0.4227046),
        (-0.0635889, 0.1786218, -0.2144134, -2.3994358, 0.0702603, 2.5729306, 0.4514749),
        (-0.0389951, 0.1104695, -0.2425126, -2.3916225, 0.0441574, 2.4984162, 0.4699694),
        (-0.1165679, 0.2313573, -0.1596608, -2.4005361, 0.0742505, 2.6278727, 0.4486592),
    ),
    (
        (0.0746626, 0.3993853, -0.0614371, -2.4394584, 0.0797585, 2.8372586, 0.7273404),
        (0.0850676, 0.2860484, -0.0767592, -2.4610789, 0.0561473, 2.7457693, 0.7450000),
        (0.0856794, 0.1286848, -0.0835223, -2.4678085, 0.0206312, 2.5959551, 0.7706067),
        (0.0800510, 0.0578652, -0.0794689, -2.4595215, 0.0078535, 2.5171897, 0.7797412),
        (0.0834069, 0.1854414, -0.0795486, -2.4689438, 0.0312558, 2.6536098, 0.7630054),
    ),
    (
        (0.3427543, 0.4323400, -0.0496637, -2.3724809, 0.0627942, 2.8037355, 1.0237991),
        (0.3287566, 0.3239155, -0.0409105, -2.3935080, 0.0316034, 2.7169836, 1.0465709),
        (0.2909108, 0.1743402, -0.0069321, -2.3998964, 0.0022373, 2.5742316, 1.0675952),
        (0.2686295, 0.1071830, 0.0150251, -2.3918052, -0.0026820, 2.4989743, 1.0711135),
        (0.3086198, 0.2282527, -0.0237074, -2.4010389, 0.0109410, 2.6292043, 1.0613892),
    ),
)

# The reset table extends those three x=0.48 m seeds to a two-dimensional
# workspace grid. The x=0.43 m and x=0.53 m rows were solved with constrained
# IK while preserving the downward tool orientation and all task joint bounds.
# Rows are ordered by y, then x; pose columns use the same heights as the seeds.
_STATE_TABLE_ANCHORS = (
    (0.43, -0.14),
    (0.48, -0.14),
    (0.53, -0.14),
    (0.43, 0.00),
    (0.48, 0.00),
    (0.53, 0.00),
    (0.43, 0.14),
    (0.48, 0.14),
    (0.53, 0.14),
)
_STATE_TABLE_ARM_POSES = (
    (
        (-0.1827602, 0.3566961, -0.1050870, -2.5290027, 0.1427738, 2.8813359, 0.3660968),
        (-0.1405546, 0.2356171, -0.1561482, -2.5522243, 0.1037001, 2.7832902, 0.3957875),
        (-0.0762273, 0.0652657, -0.2344268, -2.5599217, 0.0305949, 2.6231837, 0.4486480),
        (-0.0567678, -0.0127596, -0.2585924, -2.5512829, -0.0057401, 2.5389172, 0.4747868),
        (-0.1281367, 0.1264810, -0.1788741, -2.5610143, 0.0509304, 2.6849632, 0.4340674),
    ),
    _ARM_POSES[0],
    (
        (-0.1393943, 0.5199573, -0.1061595, -2.2020593, 0.1280282, 2.7165339, 0.4370434),
        (-0.1174986, 0.4231741, -0.1322305, -2.2222437, 0.1127438, 2.6394885, 0.4482890),
        (-0.0615496, 0.2909696, -0.1942067, -2.2286049, 0.0942004, 2.5123129, 0.4613796),
        (-0.0375556, 0.2308812, -0.2210795, -2.2211458, 0.0782906, 2.4451143, 0.4723408),
        (-0.1101384, 0.3372053, -0.1440187, -2.2298284, 0.0868206, 2.5620956, 0.4665478),
    ),
    (
        (0.0759730, 0.3203934, -0.0586236, -2.6010992, 0.0841265, 2.9201983, 0.7236422),
        (0.0905887, 0.1910140, -0.0812289, -2.6248224, 0.0480300, 2.8148493, 0.7507357),
        (0.0851180, 0.0105646, -0.0848236, -2.6321305, 0.0018844, 2.6426351, 0.7840414),
        (0.0770806, -0.0701720, -0.0783512, -2.6230035, -0.0098754, 2.5530024, 0.7925323),
        (0.0854193, 0.0756269, -0.0828133, -2.6334135, 0.0149155, 2.7087184, 0.7746989),
    ),
    _ARM_POSES[1],
    (
        (0.0715528, 0.4848095, -0.0624548, -2.2682690, 0.0765225, 2.7512705, 0.7308913),
        (0.0786326, 0.3836816, -0.0728342, -2.2885879, 0.0600667, 2.6706451, 0.7429353),
        (0.0798587, 0.2440989, -0.0782379, -2.2949124, 0.0332810, 2.5380599, 0.7619260),
        (0.0758883, 0.1812225, -0.0755081, -2.2871444, 0.0217849, 2.4677707, 0.7699870),
        (0.0770234, 0.2943766, -0.0742011, -2.2959744, 0.0409981, 2.5892367, 0.7564962),
    ),
    (
        (0.3733737, 0.3543693, -0.0466062, -2.5305452, 0.0635564, 2.8840421, 1.0535889),
        (0.3549292, 0.2323785, -0.0360207, -2.5531699, 0.0238133, 2.7852886, 1.0829521),
        (0.3077923, 0.0634289, 0.0068468, -2.5600088, -0.0008622, 2.6234150, 1.1007691),
        (0.2840671, -0.0123435, 0.0307596, -2.5512871, 0.0006774, 2.5389282, 1.0996615),
        (0.3281968, 0.1243308, -0.0128874, -2.5612529, 0.0036501, 2.6855496, 1.0975263),
    ),
    _ARM_POSES[2],
    (
        (0.3161607, 0.5173145, -0.0516801, -2.2032379, 0.0623529, 2.7192778, 0.9997473),
        (0.3037271, 0.4195258, -0.0427033, -2.2233085, 0.0363061, 2.6422440, 1.0182483),
        (0.2721164, 0.2853546, -0.0136850, -2.2294290, 0.0065578, 2.5147738, 1.0390748),
        (0.2527010, 0.2252003, 0.0055606, -2.2217196, -0.0019461, 2.4469407, 1.0450170),
        (0.2871863, 0.3336694, -0.0281290, -2.2305052, 0.0168593, 2.5640130, 1.0318844),
    ),
)

# Every stack location receives both source-order variants. Source anchors are
# chosen by maximin distance, so every pick begins at least 14.8 cm from the
# other two roles and no physical left/right order is privileged.
_STATE_TABLE_LAYOUTS = (
    (0, 5, 6),
    (0, 6, 5),
    (1, 5, 6),
    (1, 6, 5),
    (2, 3, 8),
    (2, 8, 3),
    (3, 2, 8),
    (3, 8, 2),
    (4, 2, 8),
    (4, 8, 2),
    (5, 0, 6),
    (5, 6, 0),
    (6, 0, 5),
    (6, 5, 0),
    (7, 0, 5),
    (7, 5, 0),
    (8, 2, 3),
    (8, 3, 2),
)
_NEAR_GRASP_POSE_INDEX = 0
_PREGRASP_POSE_INDEX = 1
_LIFTED_OR_RED_ALIGNED_POSE_INDEX = 2
_GREEN_ALIGNED_POSE_INDEX = 3
_GREEN_RELEASE_POSE_INDEX = 4


class StackResetRecipe(IntEnum):
    """Physical reset recipes in the order-invariant stack table."""

    FINAL_RELEASE = 0
    SECOND_PLACE = 1
    SECOND_TRANSPORT = 2
    SECOND_PICK = 3
    PAIR_READY = 4
    FIRST_PLACE = 5
    FIRST_TRANSPORT = 6
    FIRST_PICK = 7
    TABLE = 8


_RECIPE_TARGET_POTENTIAL = {
    StackResetRecipe.FINAL_RELEASE: 10.0,
    StackResetRecipe.SECOND_PLACE: 10.0,
    StackResetRecipe.SECOND_TRANSPORT: 8.0,
    StackResetRecipe.SECOND_PICK: 6.0,
    StackResetRecipe.PAIR_READY: 6.0,
    StackResetRecipe.FIRST_PLACE: 5.0,
    StackResetRecipe.FIRST_TRANSPORT: 3.0,
    StackResetRecipe.FIRST_PICK: 1.0,
    StackResetRecipe.TABLE: 1.0,
}


_FRANKA_JOINT_ORIGINS = (
    ((0.0, 0.0, 0.333), (0.0, 0.0, 0.0)),
    ((0.0, 0.0, 0.0), (-math.pi / 2.0, 0.0, 0.0)),
    ((0.0, -0.316, 0.0), (math.pi / 2.0, 0.0, 0.0)),
    ((0.0825, 0.0, 0.0), (math.pi / 2.0, 0.0, 0.0)),
    ((-0.0825, 0.384, 0.0), (-math.pi / 2.0, 0.0, 0.0)),
    ((0.0, 0.0, 0.0), (math.pi / 2.0, 0.0, 0.0)),
    ((0.088, 0.0, 0.0), (math.pi / 2.0, 0.0, 0.0)),
)
_FRANKA_HAND_AND_TOOL_OFFSET = 0.107 + 0.1034


def _rotation_matrix_from_rpy(
    roll: float,
    pitch: float,
    yaw: float,
    reference: torch.Tensor,
) -> torch.Tensor:
    """Return a fixed-origin XYZ roll-pitch-yaw rotation matrix."""
    cx, sx = math.cos(roll), math.sin(roll)
    cy, sy = math.cos(pitch), math.sin(pitch)
    cz, sz = math.cos(yaw), math.sin(yaw)
    return reference.new_tensor(
        (
            (cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx),
            (sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx),
            (-sy, cy * sx, cy * cx),
        )
    )


def _oriented_cube_pair_intersections(
    first_centers: torch.Tensor,
    first_rotations: torch.Tensor,
    second_centers: torch.Tensor,
    second_rotations: torch.Tensor,
    *,
    edge_length: float,
    penetration_tolerance: float = 1.0e-4,
) -> torch.Tensor:
    """Return rows where two oriented cubes overlap by more than a tolerance.

    This is an exact 15-axis separating-axis test for two oriented boxes.
    Face-normal checks alone are insufficient for tilted reset cubes, and the
    previous center-distance approximation missed corner/edge intersections.
    Degenerate cross products from parallel box axes are ignored.
    """
    if first_centers.shape != second_centers.shape or first_centers.shape[-1] != 3:
        raise ValueError("Cube centers must have matching shapes ending in three coordinates.")
    expected_rotation_shape = first_centers.shape[:-1] + (3, 3)
    if first_rotations.shape != expected_rotation_shape or second_rotations.shape != expected_rotation_shape:
        raise ValueError("Cube rotations must match the center batch shape and end in (3, 3).")
    if edge_length <= 0.0:
        raise ValueError("Cube edge length must be positive.")
    if penetration_tolerance < 0.0:
        raise ValueError("Cube penetration tolerance must be non-negative.")

    # Rotation-matrix columns are each box's local axes in world coordinates.
    first_axes = first_rotations.transpose(-1, -2)
    second_axes = second_rotations.transpose(-1, -2)
    cross_axes = torch.linalg.cross(
        first_axes.unsqueeze(-2),
        second_axes.unsqueeze(-3),
        dim=-1,
    ).flatten(start_dim=-3, end_dim=-2)
    candidate_axes = torch.cat((first_axes, second_axes, cross_axes), dim=-2)
    axis_norms = torch.linalg.vector_norm(candidate_axes, dim=-1)
    valid_axes = axis_norms > 1.0e-7
    normalized_axes = candidate_axes / axis_norms.clamp_min(1.0e-7).unsqueeze(-1)

    center_delta = second_centers - first_centers
    center_projection = torch.abs(torch.sum(center_delta.unsqueeze(-2) * normalized_axes, dim=-1))
    first_projection = torch.abs(torch.sum(first_axes.unsqueeze(-3) * normalized_axes.unsqueeze(-2), dim=-1)).sum(
        dim=-1
    )
    second_projection = torch.abs(torch.sum(second_axes.unsqueeze(-3) * normalized_axes.unsqueeze(-2), dim=-1)).sum(
        dim=-1
    )
    projected_radius = 0.5 * edge_length * (first_projection + second_projection)

    separated = valid_axes & (center_projection >= projected_radius - penetration_tolerance)
    return ~torch.any(separated, dim=-1)


def _franka_tool_position(joint_positions: torch.Tensor) -> torch.Tensor:
    """Compute configured Panda tool-center positions from seven joints.

    Reset rows must preserve the rigid transform between a held cube and the
    fingers. Interpolating robot joints and cube Cartesian positions
    independently violates that transform by up to two centimeters and
    produces a floating object that immediately falls under gravity. This
    compact forward kinematics follows the Franka joint origins authored in
    the task's robot asset and includes the configured 10.34 cm action-frame
    offset from ``panda_hand``.

    Args:
        joint_positions: Franka arm positions [rad], shape ``(..., 7)``.

    Returns:
        Tool-center positions [m], shape ``(..., 3)``.
    """
    if joint_positions.ndim < 1 or joint_positions.shape[-1] != 7:
        raise ValueError("Franka reset forward kinematics expects seven joint positions.")
    batch_shape = joint_positions.shape[:-1]
    flat_joints = joint_positions.reshape(-1, 7)
    batch_size = flat_joints.shape[0]
    rotation = (
        torch.eye(3, dtype=joint_positions.dtype, device=joint_positions.device)
        .unsqueeze(0)
        .expand(batch_size, -1, -1)
        .clone()
    )
    position = torch.zeros((batch_size, 3), dtype=joint_positions.dtype, device=joint_positions.device)
    reference = flat_joints[0] if batch_size > 0 else joint_positions.new_zeros(7)
    for joint_id, (origin_position, origin_rpy) in enumerate(_FRANKA_JOINT_ORIGINS):
        origin = joint_positions.new_tensor(origin_position).expand(batch_size, -1)
        position = position + torch.bmm(rotation, origin.unsqueeze(-1)).squeeze(-1)
        origin_rotation = _rotation_matrix_from_rpy(*origin_rpy, reference=reference)
        rotation = torch.matmul(rotation, origin_rotation)
        angle = flat_joints[:, joint_id]
        cosine, sine = torch.cos(angle), torch.sin(angle)
        zeros = torch.zeros_like(angle)
        ones = torch.ones_like(angle)
        joint_rotation = torch.stack(
            (
                torch.stack((cosine, -sine, zeros), dim=1),
                torch.stack((sine, cosine, zeros), dim=1),
                torch.stack((zeros, zeros, ones), dim=1),
            ),
            dim=1,
        )
        rotation = torch.bmm(rotation, joint_rotation)
    tool_offset = joint_positions.new_tensor((0.0, 0.0, _FRANKA_HAND_AND_TOOL_OFFSET)).expand(batch_size, -1)
    tool_position = position + torch.bmm(rotation, tool_offset.unsqueeze(-1)).squeeze(-1)
    return tool_position.reshape(*batch_shape, 3)


class StackResetStateTable(ManagerTermBase):
    """Apply a validated cache of order-invariant stack states.

    Rows describe physical roles (base, first movable cube, second movable
    cube), not cube colors.  Every reset independently permutes the three
    colored assets over those roles.  The curriculum therefore learns one
    physical reset manifold and aggregates evidence across all six color
    orders instead of treating color order as task semantics.

    The cache deliberately contains many more rows than there are semantic
    phases. Motion phases are densely interpolated and table starts span a
    deterministic low-discrepancy layout set. Complete physical states are
    sampled from a shared validated cache, with a learned success rate for
    each state.
    """

    _TABLE_HEIGHT = 0.0205
    _CUBE_HEIGHT = 0.04
    _PICK_PROGRESS_BINS = 17
    _RELEASE_PROGRESS_BINS = 17
    _MOTION_PROGRESS_BINS = 33
    # Eighteen workspace/source-order layouts each contribute 64 independent
    # table arrangements: 1,152 deployment starts and 6,786 rows overall.
    _TABLE_ROWS_PER_LAYOUT = 64
    _EXPECTED_ARM_JOINTS = 7
    _EXPECTED_HAND_JOINTS = 2

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._robot: Articulation = env.scene["robot"]
        self._cubes: tuple[RigidObject, ...] = tuple(env.scene[name] for name in ("cube_1", "cube_2", "cube_3"))
        self._arm_joint_ids = self._robot.find_joints("panda_joint.*", preserve_order=True)[0]
        self._hand_joint_ids = self._robot.find_joints("panda_finger.*", preserve_order=True)[0]
        if (
            len(self._arm_joint_ids) != self._EXPECTED_ARM_JOINTS
            or len(self._hand_joint_ids) != self._EXPECTED_HAND_JOINTS
        ):
            raise ValueError(
                "Franka stack resets require "
                f"{self._EXPECTED_ARM_JOINTS} arm and {self._EXPECTED_HAND_JOINTS} active-hand joints, "
                f"found {len(self._arm_joint_ids)} and {len(self._hand_joint_ids)}."
            )

        self._arm_anchors = torch.tensor(_STATE_TABLE_ARM_POSES, dtype=torch.float32, device=env.device)
        self._role_permutations = torch.tensor(
            tuple(permutations(range(3))),
            dtype=torch.long,
            device=env.device,
        )
        self._build_table(
            closed_finger_position=float(cfg.params["closed_finger_position"]),
            placed_finger_position=float(cfg.params["placed_finger_position"]),
            open_finger_position=float(cfg.params["open_finger_position"]),
            table_rows_per_layout=int(cfg.params["table_rows_per_layout"]),
        )
        self._validate_table()
        self._rows_by_recipe = tuple(
            torch.nonzero(self._recipe_ids == int(recipe), as_tuple=False).flatten() for recipe in StackResetRecipe
        )
        self._runtime_state = create_stack_reset_runtime_state(env)
        self._sampling_probabilities: torch.Tensor | None = None

    @property
    def row_count(self) -> int:
        """Number of physical rows in the reset table."""
        return int(self._arm_positions.shape[0])

    @property
    def recipe_ids(self) -> torch.Tensor:
        """Recipe ID for every reset-table row."""
        return self._recipe_ids

    @property
    def layout_ids(self) -> torch.Tensor:
        """Workspace/source-order layout ID for every reset-table row."""
        return self._layout_ids

    @property
    def layout_count(self) -> int:
        """Number of workspace/source-order layouts represented by the table."""
        return len(_STATE_TABLE_LAYOUTS)

    @property
    def recipe_names(self) -> tuple[str, ...]:
        """Stable diagnostic names for reset recipes."""
        return tuple(recipe.name.lower() for recipe in StackResetRecipe)

    def update_sampling_probabilities(self, probabilities: torch.Tensor) -> None:
        """Use the curriculum's distribution for subsequent reset draws."""
        self._sampling_probabilities = probabilities

    def _build_table(
        self,
        *,
        closed_finger_position: float,
        placed_finger_position: float,
        open_finger_position: float,
        table_rows_per_layout: int = _TABLE_ROWS_PER_LAYOUT,
    ) -> None:
        """Construct dense pick, transport, place, release, and table rows."""
        if table_rows_per_layout < 1:
            raise ValueError("table_rows_per_layout must be positive.")
        if not closed_finger_position < open_finger_position:
            raise ValueError("closed_finger_position must be less than open_finger_position.")
        arm_rows: list[torch.Tensor] = []
        hand_rows: list[torch.Tensor] = []
        position_rows: list[torch.Tensor] = []
        recipe_rows: list[int] = []
        held_role_rows: list[int] = []
        layout_rows: list[int] = []

        def append(
            recipe: StackResetRecipe,
            arm_position: torch.Tensor,
            role_positions: torch.Tensor,
            finger_position: float,
            held_role: int,
        ) -> None:
            arm_rows.append(arm_position)
            hand_rows.append(self._arm_anchors.new_full((self._EXPECTED_HAND_JOINTS,), finger_position))
            position_rows.append(role_positions)
            recipe_rows.append(int(recipe))
            held_role_rows.append(held_role)
            layout_rows.append(layout_id)

        # Author scalar waypoints on the CPU; iterating GPU tensors would synchronize each row.
        pick_progress = tuple(index / (self._PICK_PROGRESS_BINS - 1) for index in range(self._PICK_PROGRESS_BINS))
        motion_progress = tuple(index / (self._MOTION_PROGRESS_BINS - 1) for index in range(self._MOTION_PROGRESS_BINS))
        release_progress = tuple(
            index / (self._RELEASE_PROGRESS_BINS - 1) for index in range(self._RELEASE_PROGRESS_BINS)
        )
        for layout_id, (base_anchor, first_anchor, second_anchor) in enumerate(_STATE_TABLE_LAYOUTS):
            base_x, base_y = _STATE_TABLE_ANCHORS[base_anchor]
            first_x, first_y = _STATE_TABLE_ANCHORS[first_anchor]
            second_x, second_y = _STATE_TABLE_ANCHORS[second_anchor]
            base_position = self._arm_anchors.new_tensor((base_x, base_y, self._TABLE_HEIGHT))
            first_source = base_position.new_tensor((first_x, first_y, self._TABLE_HEIGHT))
            second_source = base_position.new_tensor((second_x, second_y, self._TABLE_HEIGHT))
            table_positions = torch.stack((base_position, first_source, second_source))
            first_stack_position = base_position + base_position.new_tensor((0.0, 0.0, self._CUBE_HEIGHT))
            second_stack_position = first_stack_position + base_position.new_tensor((0.0, 0.0, self._CUBE_HEIGHT))
            first_placed_alpha = float((first_stack_position[2] - 0.04) / (0.08 - 0.04))
            second_placed_alpha = float((second_stack_position[2] - 0.08) / (0.1175 - 0.08))
            first_placed_arm = torch.lerp(
                self._arm_anchors[base_anchor, _NEAR_GRASP_POSE_INDEX],
                self._arm_anchors[base_anchor, _PREGRASP_POSE_INDEX],
                first_placed_alpha,
            )
            second_placed_arm = torch.lerp(
                self._arm_anchors[base_anchor, _PREGRASP_POSE_INDEX],
                self._arm_anchors[base_anchor, _GREEN_RELEASE_POSE_INDEX],
                second_placed_alpha,
            )

            append(
                StackResetRecipe.FINAL_RELEASE,
                # The easiest row still requires a policy action. Starting
                # with the top cube held at its support avoids giving the
                # terminal tower away for free at reset.
                second_placed_arm,
                torch.stack((base_position, first_stack_position, second_stack_position)),
                closed_finger_position,
                2,
            )

            # Reset sampling must cover the terminal transition, not only the
            # state just before it. These rows keep the completed
            # tower supported while continuously opening and retracting the
            # gripper. The easiest rows deliberately bootstrap success; the
            # adaptive sampler then expands toward harder precursor states.
            full_stack_positions = torch.stack((base_position, first_stack_position, second_stack_position))
            for progress in release_progress:
                value = float(progress)
                append(
                    StackResetRecipe.FINAL_RELEASE,
                    torch.lerp(
                        second_placed_arm,
                        self._arm_anchors[base_anchor, _GREEN_RELEASE_POSE_INDEX],
                        progress,
                    ),
                    full_stack_positions,
                    placed_finger_position + value * (open_finger_position - placed_finger_position),
                    -1,
                )

            # Both movable cubes follow the same place, lift, transport, and
            # pick path. Only the support height and calibrated arm anchors differ.
            for phase in (
                (
                    2,
                    second_anchor,
                    second_placed_arm,
                    _GREEN_ALIGNED_POSE_INDEX,
                    (StackResetRecipe.SECOND_PLACE, StackResetRecipe.SECOND_TRANSPORT, StackResetRecipe.SECOND_PICK),
                ),
                (
                    1,
                    first_anchor,
                    first_placed_arm,
                    _LIFTED_OR_RED_ALIGNED_POSE_INDEX,
                    (StackResetRecipe.FIRST_PLACE, StackResetRecipe.FIRST_TRANSPORT, StackResetRecipe.FIRST_PICK),
                ),
            ):
                role, source_anchor, placed_arm, aligned_pose, recipes = phase
                place_recipe, transport_recipe, pick_recipe = recipes
                source_positions = table_positions.clone()
                if role == 2:
                    source_positions[1] = first_stack_position
                placed_positions = source_positions.clone()
                placed_positions[role] = base_position + base_position.new_tensor((0.0, 0.0, role * self._CUBE_HEIGHT))
                near_grasp_arm = self._arm_anchors[source_anchor, _NEAR_GRASP_POSE_INDEX]
                pregrasp_arm = self._arm_anchors[source_anchor, _PREGRASP_POSE_INDEX]
                lifted_arm = self._arm_anchors[source_anchor, _LIFTED_OR_RED_ALIGNED_POSE_INDEX]
                aligned_arm = self._arm_anchors[base_anchor, aligned_pose]

                for progress in motion_progress:
                    supported = progress == 1.0
                    append(
                        place_recipe,
                        torch.lerp(aligned_arm, placed_arm, progress),
                        placed_positions if supported else source_positions,
                        placed_finger_position if supported else closed_finger_position,
                        -1 if supported else role,
                    )

                # Join the lift and transport at one shared waypoint.
                for progress in motion_progress:
                    append(
                        transport_recipe,
                        torch.lerp(near_grasp_arm, lifted_arm, progress),
                        source_positions,
                        closed_finger_position,
                        role,
                    )
                for progress in motion_progress[1:]:
                    append(
                        transport_recipe,
                        torch.lerp(lifted_arm, aligned_arm, progress),
                        source_positions,
                        closed_finger_position,
                        role,
                    )

                # Include finger closure so the policy can learn acquisition,
                # rather than only continuing from a reset-supplied grasp.
                for progress in pick_progress:
                    append(
                        pick_recipe,
                        torch.lerp(pregrasp_arm, near_grasp_arm, progress),
                        source_positions,
                        open_finger_position,
                        -1,
                    )
                for progress in pick_progress[1:]:
                    append(
                        pick_recipe,
                        near_grasp_arm,
                        source_positions,
                        open_finger_position + progress * (closed_finger_position - open_finger_position),
                        role if progress == 1.0 else -1,
                    )

                if role == 2:
                    # Connect first-cube release to second-cube acquisition.
                    for progress in motion_progress:
                        append(
                            StackResetRecipe.PAIR_READY,
                            torch.lerp(
                                self._arm_anchors[base_anchor, _GREEN_RELEASE_POSE_INDEX], pregrasp_arm, progress
                            ),
                            source_positions,
                            open_finger_position,
                            -1,
                        )

            # Deployment starts independently sample all three roles over the
            # reachable rectangle. Rejection keeps enough clearance for a
            # top-down grasp; unlike the old side-by-side template, no role is
            # assigned the spatial center or a preferred left/right order.
            for table_row in range(table_rows_per_layout):
                lattice_index = layout_id * table_rows_per_layout + table_row

                def lattice(multiplier: int, offset: int) -> float:
                    return ((lattice_index * multiplier + offset) % 4093) / 4092.0

                table_xy: tuple[tuple[float, float], ...] | None = None
                for attempt in range(32):
                    candidates = tuple(
                        (
                            0.40
                            + 0.16
                            * lattice(
                                151 + 38 * role_id + 12 * attempt,
                                67 + 43 * role_id + 19 * attempt,
                            ),
                            -0.18
                            + 0.36
                            * lattice(
                                193 + 46 * role_id + 14 * attempt,
                                89 + 53 * role_id + 23 * attempt,
                            ),
                        )
                        for role_id in range(3)
                    )
                    if all(
                        math.dist(candidates[first], candidates[second]) >= 0.085
                        for first in range(3)
                        for second in range(first + 1, 3)
                    ):
                        table_xy = candidates
                        break
                if table_xy is None:
                    table_xy = tuple(
                        _STATE_TABLE_ANCHORS[anchor] for anchor in (base_anchor, first_anchor, second_anchor)
                    )
                randomized_table_positions = base_position.new_tensor(
                    tuple((x, y, self._TABLE_HEIGHT) for x, y in table_xy)
                )
                approach_anchor = min(int(lattice(263, 113) * len(_STATE_TABLE_ANCHORS)), len(_STATE_TABLE_ANCHORS) - 1)
                arm_noise = self._arm_anchors.new_tensor(
                    tuple(0.04 * (lattice(271 + 18 * joint_id, 127 + 31 * joint_id) - 0.5) for joint_id in range(7))
                )
                append(
                    StackResetRecipe.TABLE,
                    self._arm_anchors[approach_anchor, _PREGRASP_POSE_INDEX] + arm_noise,
                    randomized_table_positions,
                    open_finger_position,
                    -1,
                )

        self._arm_positions = torch.stack(arm_rows)
        self._hand_positions = torch.stack(hand_rows)
        self._role_positions = torch.stack(position_rows)
        self._recipe_ids = torch.tensor(recipe_rows, dtype=torch.long, device=self.device)
        self._held_roles = torch.tensor(held_role_rows, dtype=torch.long, device=self.device)
        self._layout_ids = torch.tensor(layout_rows, dtype=torch.long, device=self.device)
        self._target_potentials = torch.tensor(
            tuple(_RECIPE_TARGET_POTENTIAL[StackResetRecipe(recipe)] for recipe in recipe_rows),
            dtype=torch.float32,
            device=self.device,
        )
        # Cache complete cube pose state, not only position. Random yaw is
        # physically immaterial for perfect cubes but prevents the reset
        # mechanism from relying on one authored quaternion representation.
        row_ids = torch.arange(self._role_positions.shape[0], device=self.device).unsqueeze(1)
        role_ids = torch.arange(3, device=self.device).unsqueeze(0)
        yaw_fraction = torch.remainder(row_ids * 193 + role_ids * 389 + 17, 1021).float() / 1020.0
        half_yaw = math.pi * (2.0 * yaw_fraction - 1.0)
        self._role_quaternions = torch.zeros(
            (self._role_positions.shape[0], 3, 4),
            dtype=torch.float32,
            device=self.device,
        )
        self._role_quaternions[..., 2] = torch.sin(half_yaw)
        self._role_quaternions[..., 3] = torch.cos(half_yaw)
        held_rows = torch.nonzero(self._held_roles >= 0, as_tuple=False).flatten()
        if held_rows.numel() > 0:
            # Held cubes follow the actual hand FK, not an independent Cartesian trajectory.
            self._role_positions[held_rows, self._held_roles[held_rows]] = _franka_tool_position(
                self._arm_positions[held_rows]
            )
            self._role_quaternions[held_rows, self._held_roles[held_rows]] = self._role_quaternions.new_tensor(
                (0.0, 0.0, 0.0, 1.0)
            )

    def _validate_table(self) -> None:
        """Reject non-finite, penetrating, or semantically inconsistent rows."""
        tensors = (
            self._arm_positions,
            self._hand_positions,
            self._role_positions,
            self._role_quaternions,
        )
        if any(not bool(torch.isfinite(value).all()) for value in tensors):
            raise RuntimeError("Stack reset table contains non-finite values.")
        if self._layout_ids.shape != self._recipe_ids.shape:
            raise RuntimeError("Stack reset table layout IDs do not align with its physical rows.")
        if bool(torch.any((self._layout_ids < 0) | (self._layout_ids >= self.layout_count))):
            raise RuntimeError("Stack reset table contains an invalid workspace layout ID.")
        if bool(torch.any(self._role_positions[..., 2] < self._TABLE_HEIGHT - 1.0e-6)):
            raise RuntimeError("Stack reset table places a cube below the table support height.")

        role_rotations = matrix_from_quat(self._role_quaternions)
        for first_role, second_role in ((0, 1), (0, 2), (1, 2)):
            intersecting = _oriented_cube_pair_intersections(
                self._role_positions[:, first_role],
                role_rotations[:, first_role],
                self._role_positions[:, second_role],
                role_rotations[:, second_role],
                edge_length=self._CUBE_HEIGHT,
            )
            if bool(torch.any(intersecting)):
                invalid_rows = torch.nonzero(intersecting, as_tuple=False).flatten()
                raise RuntimeError(
                    "Stack reset table contains intersecting oriented cube volumes "
                    f"(roles={first_role}/{second_role}, rows={invalid_rows[:8].tolist()}, "
                    f"count={invalid_rows.numel()})."
                )
        # Every phase after the first placement assumes role one is already
        # supported on role zero. A geometrically valid row with role one at
        # its source makes the second-stage target potential impossible.
        second_stage_recipes = torch.tensor(
            (
                int(StackResetRecipe.FINAL_RELEASE),
                int(StackResetRecipe.SECOND_PLACE),
                int(StackResetRecipe.SECOND_TRANSPORT),
                int(StackResetRecipe.SECOND_PICK),
                int(StackResetRecipe.PAIR_READY),
            ),
            dtype=self._recipe_ids.dtype,
            device=self.device,
        )
        second_stage_rows = torch.isin(self._recipe_ids, second_stage_recipes)
        first_support_delta = self._role_positions[second_stage_rows, 1] - self._role_positions[second_stage_rows, 0]
        expected_support_delta = first_support_delta.new_tensor((0.0, 0.0, self._CUBE_HEIGHT))
        if bool(torch.any(torch.abs(first_support_delta - expected_support_delta) > 1.0e-5)):
            raise RuntimeError("A second-stage stack reset row does not place the first movable cube on the base.")

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int] | torch.Tensor | slice | None,
        closed_finger_position: float = 0.020,
        placed_finger_position: float = 0.021,
        open_finger_position: float = 0.040,
        table_rows_per_layout: int = _TABLE_ROWS_PER_LAYOUT,
        fixed_recipe: int | None = None,
        fixed_role_permutation: int | None = None,
        arm_joint_noise_range: float = 0.0,
        table_arm_joint_noise_range: float = 0.0,
        table_cube_planar_translation_range: float = 0.0,
        table_cube_rotation_range: float = 0.0,
    ) -> None:
        """Restore cached states with bounded continuous state randomization."""
        del (
            closed_finger_position,
            placed_finger_position,
            open_finger_position,
            table_rows_per_layout,
        )
        if env_ids is None or isinstance(env_ids, slice):
            env_ids = torch.arange(env.num_envs, device=env.device)[env_ids if env_ids is not None else slice(None)]
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=env.device).flatten()
        if env_ids.numel() == 0:
            return
        if (
            min(
                arm_joint_noise_range,
                table_arm_joint_noise_range,
                table_cube_planar_translation_range,
                table_cube_rotation_range,
            )
            < 0.0
        ):
            raise ValueError("Reset-state randomization ranges must be non-negative.")
        state = self._runtime_state
        if fixed_recipe is None and self._sampling_probabilities is not None:
            row_ids = torch.multinomial(self._sampling_probabilities, env_ids.numel(), replacement=True)
        else:
            recipe = StackResetRecipe.TABLE if fixed_recipe is None else StackResetRecipe(fixed_recipe)
            recipe_rows = self._rows_by_recipe[recipe]
            row_ids = recipe_rows[torch.randint(recipe_rows.numel(), (env_ids.numel(),), device=self.device)]
        if fixed_role_permutation is None:
            permutation_ids = torch.randint(
                self._role_permutations.shape[0],
                (env_ids.numel(),),
                device=self.device,
            )
        else:
            if not 0 <= fixed_role_permutation < self._role_permutations.shape[0]:
                raise ValueError("fixed_role_permutation is outside the six cube permutations.")
            permutation_ids = torch.full(
                (env_ids.numel(),),
                fixed_role_permutation,
                dtype=torch.long,
                device=self.device,
            )
        role_to_cube = self._role_permutations[permutation_ids]
        held_roles = self._held_roles[row_ids]
        has_held_cube = held_roles >= 0
        selected_cube_ids = role_to_cube.gather(1, held_roles.clamp_min(0).unsqueeze(1)).squeeze(1)
        held_cube_ids = torch.where(has_held_cube, selected_cube_ids, -1)

        joint_positions = self._robot.data.default_joint_pos.torch[env_ids].clone()
        joint_velocities = torch.zeros_like(joint_positions)
        joint_positions[:, self._arm_joint_ids] = self._arm_positions[row_ids]
        joint_positions[:, self._hand_joint_ids] = self._hand_positions[row_ids]
        table_rows = self._recipe_ids[row_ids] == int(StackResetRecipe.TABLE)
        arm_noise_scale = torch.full(
            (env_ids.numel(), 1),
            arm_joint_noise_range,
            dtype=torch.float32,
            device=self.device,
        )
        arm_noise_scale[table_rows] = table_arm_joint_noise_range
        # Joint-ID indexing may be advanced indexing and therefore return a
        # copy. Write the randomized values back explicitly after clamping.
        arm_positions = joint_positions[:, self._arm_joint_ids].clone()
        arm_positions += (2.0 * torch.rand_like(arm_positions) - 1.0) * arm_noise_scale
        arm_lower = arm_positions.new_tensor(FRANKA_STACK_ARM_WORKSPACE_LOWER)
        arm_upper = arm_positions.new_tensor(FRANKA_STACK_ARM_WORKSPACE_UPPER)
        arm_positions.clamp_(min=arm_lower, max=arm_upper)
        joint_positions[:, self._arm_joint_ids] = arm_positions
        self._robot.actuators.target_command.set_position_index(value=joint_positions, env_ids=env_ids)
        self._robot.actuators.target_command.set_velocity_index(value=joint_velocities, env_ids=env_ids)
        self._robot.write_joint_position_to_sim_index(position=joint_positions, env_ids=env_ids)
        self._robot.write_joint_velocity_to_sim_index(velocity=joint_velocities, env_ids=env_ids)

        role_positions = self._role_positions[row_ids].clone()
        role_quaternions = self._role_quaternions[row_ids].clone()
        if table_cube_planar_translation_range > 0.0:
            translation = (
                2.0
                * torch.rand(
                    (env_ids.numel(), 2),
                    dtype=torch.float32,
                    device=self.device,
                )
                - 1.0
            ) * table_cube_planar_translation_range
            role_positions[table_rows, :, :2] += translation[table_rows].unsqueeze(1)
        if table_cube_rotation_range > 0.0:
            angles = (
                2.0
                * torch.rand(
                    env_ids.numel(),
                    dtype=torch.float32,
                    device=self.device,
                )
                - 1.0
            ) * table_cube_rotation_range
            angles = angles[table_rows]
            table_xy = role_positions[table_rows, :, :2]
            centers = table_xy.mean(dim=1, keepdim=True)
            centered = table_xy - centers
            cosine = torch.cos(angles).unsqueeze(1)
            sine = torch.sin(angles).unsqueeze(1)
            rotated_x = cosine * centered[..., 0] - sine * centered[..., 1]
            rotated_y = sine * centered[..., 0] + cosine * centered[..., 1]
            role_positions[table_rows, :, :2] = centers + torch.stack((rotated_x, rotated_y), dim=-1)

            half_angles = 0.5 * angles
            yaw_w = torch.cos(half_angles).unsqueeze(1)
            yaw_z = torch.sin(half_angles).unsqueeze(1)
            yaw_quaternions = torch.zeros_like(role_quaternions[table_rows])
            yaw_quaternions[..., 2] = yaw_z
            yaw_quaternions[..., 3] = yaw_w
            role_quaternions[table_rows] = normalize(quat_mul(yaw_quaternions, role_quaternions[table_rows]))

        # Joint perturbations must never detach a reset-authored grasp. Move
        # the held role to the perturbed FK tool center before colors are
        # scattered over physical roles.
        role_positions[has_held_cube, held_roles[has_held_cube]] = _franka_tool_position(arm_positions[has_held_cube])
        cube_positions = torch.zeros_like(role_positions)
        cube_quaternions = torch.zeros_like(role_quaternions)
        cube_positions.scatter_(
            1,
            role_to_cube.unsqueeze(-1).expand_as(role_positions),
            role_positions,
        )
        cube_quaternions.scatter_(
            1,
            role_to_cube.unsqueeze(-1).expand_as(role_quaternions),
            role_quaternions,
        )
        root_velocity = torch.zeros((env_ids.numel(), 6), dtype=torch.float32, device=self.device)
        for cube_id, cube in enumerate(self._cubes):
            root_pose = cube.data.default_root_pose.torch[env_ids].clone()
            root_pose[:, :3] = cube_positions[:, cube_id] + env.scene.env_origins[env_ids]
            root_pose[:, 3:7] = cube_quaternions[:, cube_id]
            cube.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
            cube.write_root_velocity_to_sim_index(root_velocity=root_velocity, env_ids=env_ids)

        state.row_ids[env_ids] = row_ids
        state.role_to_cube[env_ids] = role_to_cube
        state.recipes[env_ids] = self._recipe_ids[row_ids]
        state.target_potentials[env_ids] = self._target_potentials[row_ids]
        state.held_cube_ids[env_ids] = held_cube_ids
