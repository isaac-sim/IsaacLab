# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Have a Unitree H1 perform the tablecloth trick with Newton VBD and Newton IK.

An ``InteractiveSceneCfg`` declares the stock Isaac Lab ground plane, H1
articulation, table, cloth, and tableware. The runtime controls H1 through Isaac
Lab's articulation command buffers and standard scene write/step/update loop. A
GPU-resident Warp state machine generates hand targets, and ``NewtonIKSolver``
converts them to joint targets without CPU copies in the simulation loop.

The first run downloads the H1-with-hands MJCF from ``newton-assets`` and uses
Isaac Lab's MJCF converter. In a minimal uv installation, enable the existing
``importers`` extra for that conversion.

.. code-block:: bash

    # Newton GL is the default visualizer.
    uv run --extra importers python scripts/demos/newton_tablecloth_h1.py --device cuda:0

    # Run the complete state machine without rendering.
    uv run --extra importers python scripts/demos/newton_tablecloth_h1.py \
        --device cuda:0 --visualizer none --max_steps 312

    # Record the complete state machine as a 60 FPS MP4 through Isaac Sim Kit.
    uv run --extra isaacsim --extra importers --extra video python scripts/demos/newton_tablecloth_h1.py \
        --device cuda:0 --visualizer kit --video
"""

from __future__ import annotations

import argparse
import os
import tempfile
from dataclasses import MISSING, dataclass
from typing import TYPE_CHECKING

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description="Newton VBD H1 tablecloth trick.")
parser.add_argument("--max_steps", type=int, default=-1, help="Stop after this many frames; negative runs forever.")
parser.add_argument("--pull_speed", type=float, default=0.80, help="Task-space pull speed [m/s].")
parser.add_argument("--video", action="store_true", help="Record the complete demo to videos/newton_tablecloth_h1/.")
add_launcher_args(parser)
parser.set_defaults(visualizer=["newton_gl"])
args_cli = parser.parse_args()

import numpy as np
import warp as wp
from _newton_tablecloth_utils import (
    BOWL_LOCAL_HEIGHT,
    BOWL_LOCAL_RADIUS,
    BOWL_LOCAL_Z_MIN,
    BOWL_SCALE,
    BOWL_USD,
    FORK_CENTER_OF_MASS,
    FORK_DIAGONAL_INERTIA,
    FORK_LOCAL_Z_MIN,
    FORK_ROTATION,
    FORK_SCALE,
    FORK_USD,
    KITCHEN_ISLAND_SIZE,
    KITCHEN_ISLAND_USD,
    RIGID_GAP,
    WINE_GLASS_CENTER_OF_MASS,
    WINE_GLASS_DIAGONAL_INERTIA,
    WINE_GLASS_LOCAL_Z_MIN,
    WINE_GLASS_USD,
    VisualTableUsdFileCfg,
    collision_properties,
    create_video_recorder,
    rigid_material,
    rigid_object_cfg,
    tabletop_collider_cfg,
)
from isaaclab_newton.physics import (
    NewtonCfg,
    NewtonCollisionPipelineCfg,
    NewtonShapeCfg,
    NewtonSoftContactCfg,
    VBDSolverCfg,
)
from isaaclab_newton.sim.schemas import NewtonDeformableBodyPropertiesCfg, NewtonSDFCollisionCfg
from isaaclab_newton.sim.spawners.materials import NewtonSurfaceDeformableBodyMaterialCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg, DeformableObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab_newton.ik import NewtonIKSolver
    from newton import Model

FPS = 60
SUBSTEPS = 16
VIDEO_STEPS = 312
VIDEO_OUTPUT_DIR = "videos/newton_tablecloth_h1"
VIDEO_RESOLUTION = (1920, 1080)
TABLE_TOP_Z = 1.09
TABLEWARE_CLEARANCE = 0.008
CLOTH_Z = TABLE_TOP_Z + 0.002
PULL_DISTANCE = 0.40
PULL_RAMP_TIME = wp.constant(0.25)

# Dimensions measured from the SimReady assets. Non-uniform scaling turns the kitchen island into a compact,
# H1-height serving table without changing the manipulation geometry used by the state machine.
KITCHEN_ISLAND_SCALE = (
    0.40 / KITCHEN_ISLAND_SIZE[0],
    0.72 / KITCHEN_ISLAND_SIZE[1],
    TABLE_TOP_Z / KITCHEN_ISLAND_SIZE[2],
)

GROUP_LEFT_THUMB = 0
GROUP_RIGHT_THUMB = 1
GROUP_LEFT_INDEX = 2
GROUP_RIGHT_INDEX = 3
GROUP_OTHER = 4

FINGERS_OPEN = (0.0, 0.0, 0.0, 0.0, 0.0)
FINGERS_CURLED = (0.0, 0.0, 0.0, 0.0, 0.80)
FINGERS_INSERTED = (0.0, 0.0, 0.75, 0.75, 0.80)
FINGERS_PREPINCHED = (0.0, 0.0, 0.737080, 0.713855, 0.80)
FINGERS_PINCHED = (1.0, 1.0, 0.737080, 0.713855, 0.80)


@dataclass(frozen=True)
class _MotionPhase:
    """One timed interpolation in the grasp sequence."""

    name: str
    duration: float
    start_keyframe: int
    end_keyframe: int
    start_fingers: tuple[float, float, float, float, float]
    end_fingers: tuple[float, float, float, float, float]


MOTION_PHASES = (
    _MotionPhase("settle", 0.50, 0, 0, FINGERS_OPEN, FINGERS_OPEN),
    _MotionPhase("approach", 0.80, 0, 2, FINGERS_OPEN, FINGERS_CURLED),
    _MotionPhase("descend", 0.60, 2, 4, FINGERS_CURLED, FINGERS_INSERTED),
    _MotionPhase("insert", 0.50, 4, 6, FINGERS_INSERTED, FINGERS_INSERTED),
    _MotionPhase("prelift", 0.60, 6, 8, FINGERS_INSERTED, FINGERS_PREPINCHED),
    _MotionPhase("close", 0.40, 8, 8, FINGERS_PREPINCHED, FINGERS_PINCHED),
    _MotionPhase("lift", 0.50, 8, 10, FINGERS_PINCHED, FINGERS_PINCHED),
    _MotionPhase("pinch", 0.60, 10, 10, FINGERS_PINCHED, FINGERS_PINCHED),
)
STATE_PULL = wp.constant(len(MOTION_PHASES))
STATE_HOLD = wp.constant(len(MOTION_PHASES) + 1)

HAND_OFFSETS = ((0.146273, -0.068447, 0.028077), (0.148808, 0.068652, 0.026675))
HAND_ROTATIONS = ((-0.09, 0.46, 0.03, 0.88), (0.09023, 0.46115, -0.03008, 0.88221))
HAND_KEYFRAMES = (
    (-0.48, 0.24, 1.24),
    (-0.48, -0.24, 1.24),
    (-0.30, 0.24, 1.16),
    (-0.30, -0.24, 1.16),
    (-0.30, 0.24, 1.050),
    (-0.30, -0.24, 1.052),
    (-0.225, 0.24, 1.050),
    (-0.225, -0.24, 1.052),
    (-0.195, 0.24, 1.110),
    (-0.195, -0.24, 1.110),
    (-0.195, 0.24, 1.115),
    (-0.195, -0.24, 1.115),
)
THUMB_CLOSED_VALUES = (
    (1.273907, 0.160957, 0.369535, 0.892908),
    (1.192278, 0.195421, 0.400690, 0.679765),
)


@wp.func
def _smoothstep(value: float) -> float:
    u = wp.clamp(value, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


@wp.kernel
def _infer_state_machine(
    dt: float,
    pull_speed: float,
    keyframes: wp.array(dtype=wp.vec3),
    phase_durations: wp.array(dtype=float),
    phase_start_keyframes: wp.array(dtype=wp.int32),
    phase_end_keyframes: wp.array(dtype=wp.int32),
    phase_start_fingers: wp.array(dtype=float),
    phase_end_fingers: wp.array(dtype=float),
    left_target: wp.array(dtype=wp.vec3),
    right_target: wp.array(dtype=wp.vec3),
    finger_fractions: wp.array(dtype=float),
    state: wp.array(dtype=wp.int32),
    state_time: wp.array(dtype=float),
    pull_distance: wp.array(dtype=float),
):
    """Advance the single H1 task-space state machine entirely in Warp."""
    current_state = state[0]
    elapsed = state_time[0] + dt

    if current_state < STATE_PULL and elapsed >= phase_durations[current_state]:
        current_state += 1
        elapsed = 0.0

    left = keyframes[0]
    right = keyframes[1]
    if current_state < STATE_PULL:
        duration = phase_durations[current_state]
        u = _smoothstep(elapsed / duration)
        start_keyframe = phase_start_keyframes[current_state]
        end_keyframe = phase_end_keyframes[current_state]
        left = wp.lerp(keyframes[start_keyframe], keyframes[end_keyframe], u)
        right = wp.lerp(keyframes[start_keyframe + 1], keyframes[end_keyframe + 1], u)
        for group in range(5):
            finger_offset = current_state * 5 + group
            finger_fractions[group] = wp.lerp(phase_start_fingers[finger_offset], phase_end_fingers[finger_offset], u)
    elif current_state == STATE_PULL:
        speed_ramp = _smoothstep(elapsed / PULL_RAMP_TIME)
        distance = wp.min(pull_distance[0] + speed_ramp * pull_speed * dt, PULL_DISTANCE)
        pull_distance[0] = distance
        drop = 0.0
        if distance > 0.08:
            drop = 0.175 * _smoothstep((distance - 0.08) / (PULL_DISTANCE - 0.08))
        pull_offset = wp.vec3(-distance, 0.0, -drop)
        left = keyframes[10] + pull_offset
        right = keyframes[11] + pull_offset
        for group in range(5):
            finger_fractions[group] = phase_end_fingers[(STATE_PULL - 1) * 5 + group]
        if distance >= PULL_DISTANCE:
            current_state = STATE_HOLD
    else:
        distance = pull_distance[0]
        drop = 0.175 * _smoothstep((distance - 0.08) / (PULL_DISTANCE - 0.08))
        pull_offset = wp.vec3(-distance, 0.0, -drop)
        left = keyframes[10] + pull_offset
        right = keyframes[11] + pull_offset
        for group in range(5):
            finger_fractions[group] = phase_end_fingers[(STATE_PULL - 1) * 5 + group]

    left_target[0] = left
    right_target[0] = right
    state[0] = current_state
    state_time[0] = elapsed


@wp.kernel
def _set_finger_targets(
    joint_q: wp.array(dtype=float),
    finger_indices: wp.array(dtype=wp.int32),
    closed_values: wp.array(dtype=float),
    finger_groups: wp.array(dtype=wp.int32),
    fractions: wp.array(dtype=float),
):
    i = wp.tid()
    joint_q[finger_indices[i]] = fractions[finger_groups[i]] * closed_values[i]


@wp.kernel
def _update_control_targets(
    desired_q: wp.array(dtype=float),
    previous_q: wp.array(dtype=float),
    dt: float,
    target_q: wp.array(dtype=float),
    target_qd: wp.array(dtype=float),
):
    i = wp.tid()
    delta = wp.clamp(desired_q[i] - previous_q[i], -40.0 * dt, 40.0 * dt)
    target_q[i] = previous_q[i] + delta
    target_qd[i] = delta / dt
    previous_q[i] = target_q[i]


def _find_suffix(labels: list[str], suffix: str) -> int:
    matches = [index for index, label in enumerate(labels) if label.endswith(f"/{suffix}")]
    if len(matches) != 1:
        raise ValueError(f"Expected one label ending in '/{suffix}', found {len(matches)}")
    return matches[0]


def _unit_quat(values: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    values_np = np.asarray(values, dtype=np.float32)
    values_np /= np.linalg.norm(values_np)
    return tuple(float(value) for value in values_np)


def _convert_h1_asset() -> str:
    """Convert and cache the downloaded H1-with-hands MJCF through Isaac Lab."""
    import newton.utils  # noqa: PLC0415

    converter_cfg = MjcfConverterCfg(
        asset_path=str(newton.utils.download_asset("unitree_h1") / "mjcf/h1_with_hand.xml"),
        usd_dir=os.path.join(tempfile.gettempdir(), "IsaacLab", "newton_tablecloth_h1_vbd"),
        usd_file_name="h1_with_hand.usda",
        fix_base=True,
        self_collision=False,
        robot_type="Humanoid",
        # The demo drives VBD through Isaac Lab actuators, so it does not need
        # the importer's extra MuJoCo/PhysX payload conversion or layered asset transform.
        run_asset_transformer=False,
        run_multi_physics_conversion=False,
    )
    return MjcfConverter(converter_cfg).usd_path


@configclass
class _H1ArticulationCfg(ArticulationCfg):
    """H1 configuration that provisions the imported hand meshes for cloth contact."""

    def _post_spawn(self, stage) -> None:
        """Select SDF collision for meshes and bind the fingertip contact material."""
        super()._post_spawn(stage)
        from pxr import UsdGeom, UsdPhysics  # noqa: PLC0415

        from isaaclab.sim.schemas import apply_namespaced  # noqa: PLC0415
        from isaaclab.sim.spawners.materials.physics_materials import (  # noqa: PLC0415
            spawn_rigid_body_material_from_fragments,
        )

        root_path = self.spawn.spawn_path if self.spawn is not None and self.spawn.spawn_path else self.prim_path
        meshes = sim_utils.get_all_matching_child_prims(
            root_path,
            predicate=lambda prim: prim.IsA(UsdGeom.Mesh),
            stage=stage,
        )
        for mesh in meshes:
            mesh_path = mesh.GetPath().pathString
            if "/left_hand_link/" in mesh_path or "/right_hand_link/" in mesh_path:
                usd_mesh = UsdGeom.Mesh(mesh)
                usd_mesh.CreateDisplayColorAttr([(0.0100228, 0.0100228, 0.0100228)])
                usd_mesh.CreateDisplayOpacityAttr([1.0])

        mesh_colliders = [mesh for mesh in meshes if mesh.HasAPI(UsdPhysics.CollisionAPI)]
        for collider in mesh_colliders:
            collider.RemoveAppliedSchema("NewtonMeshCollisionAPI")
            UsdPhysics.MeshCollisionAPI(collider).GetApproximationAttr().Set("none")
            apply_namespaced(
                NewtonSDFCollisionCfg(sdf_max_resolution=64, sdf_padding=0.012),
                collider.GetPath().pathString,
                stage,
            )

        grasp_material_path = f"{root_path}/GraspMaterial"
        spawn_rigid_body_material_from_fragments(
            grasp_material_path,
            rigid_material(
                density=None,
                friction=200.0,
                contact_stiffness=8.0e3,
                contact_damping=2.0e1,
            ),
            stage=stage,
        )
        grasp_roots = {"L_thumb_proximal_base", "L_index_proximal", "R_thumb_proximal_base", "R_index_proximal"}
        finger_prims = sim_utils.get_all_matching_child_prims(
            root_path,
            predicate=lambda prim: prim.GetName() in grasp_roots,
            stage=stage,
        )
        if len(finger_prims) != len(grasp_roots):
            raise RuntimeError(f"Expected {len(grasp_roots)} H1 grasp roots, found {len(finger_prims)}")
        for prim in finger_prims:
            sim_utils.bind_physics_material(prim.GetPath(), grasp_material_path, stage=stage)


def _h1_articulation_cfg(usd_path: str) -> ArticulationCfg:
    """Create the Isaac Lab articulation configuration for H1."""
    return _H1ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            make_uninstanceable=True,
            fix_root_link=True,
            collision_props=collision_properties(),
            physics_material=rigid_material(
                density=None,
                friction=0.50,
                contact_stiffness=1.0e3,
                contact_damping=1.0e-2,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(-0.75, 0.0, 0.0)),
        actuators={
            "body": ImplicitActuatorCfg(
                joint_names_expr=[r"(?!torso_joint$)(?![LR]_(?:thumb|index|middle|ring|pinky)_).+"],
                stiffness=5.0e4,
                damping=5.0e2,
            ),
            "torso": ImplicitActuatorCfg(
                joint_names_expr=["torso_joint"],
                stiffness=2.0e5,
                damping=2.0e3,
            ),
            "fingers": ImplicitActuatorCfg(
                joint_names_expr=[r"[LR]_(?:thumb|index|middle|ring|pinky)_.+"],
                stiffness=4.0e4,
                damping=1.0e2,
            ),
        },
    )


@configclass
class H1TableclothSceneCfg(InteractiveSceneCfg):
    """Isaac Lab scene for the H1 tablecloth manipulation sequence."""

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        collision_group=-1,
    )

    robot: ArticulationCfg = MISSING

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=VisualTableUsdFileCfg(
            usd_path=KITCHEN_ISLAND_USD,
            scale=KITCHEN_ISLAND_SCALE,
            variants={"Physics": "none"},
            make_uninstanceable=True,
            rigid_props=sim_utils.NewtonRigidBodyPropertiesCfg(rigid_body_enabled=False),
            collision_props=sim_utils.NewtonCollisionPropertiesCfg(collision_enabled=False),
        ),
    )
    tabletop_collider = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Tabletop",
        spawn=tabletop_collider_cfg(
            (0.40, 0.72, 0.08),
            friction=0.35,
            contact_stiffness=1.0e4,
            contact_damping=1.0e1,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, TABLE_TOP_Z - 0.04)),
    )

    cloth = DeformableObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cloth",
        spawn=sim_utils.MeshRectangleCfg(
            size=(0.46, 0.70),
            edge_refinement=24,
            deformable_props=NewtonDeformableBodyPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.76, 0.05, 0.05)),
            physics_material=NewtonSurfaceDeformableBodyMaterialCfg(
                density=0.24,
                particle_radius=0.001,
                tri_ke=5.0e4,
                tri_ka=5.0e4,
                tri_kd=5.0e1,
                edge_ke=0.10,
                edge_kd=1.0e-3,
            ),
        ),
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(-0.01, 0.0, CLOTH_Z)),
    )

    bowl = rigid_object_cfg(
        usd_path=BOWL_USD,
        position=(0.08, -0.07, CLOTH_Z + TABLEWARE_CLEARANCE - BOWL_SCALE * BOWL_LOCAL_Z_MIN),
        mass=0.45,
        collision_proxy_cfg=sim_utils.CylinderCfg(
            radius=BOWL_LOCAL_RADIUS,
            height=BOWL_LOCAL_HEIGHT,
            visible=False,
            collision_props=collision_properties(),
            physics_material=rigid_material(
                density=None,
                friction=0.08,
                contact_stiffness=1.0e4,
                contact_damping=1.0e1,
            ),
        ),
        collision_proxy_position=(0.0, 0.0, BOWL_LOCAL_Z_MIN + 0.5 * BOWL_LOCAL_HEIGHT),
        scale=(BOWL_SCALE,) * 3,
    ).replace(prim_path="{ENV_REGEX_NS}/Bowl")
    glass = rigid_object_cfg(
        usd_path=WINE_GLASS_USD,
        position=(0.10, 0.13, CLOTH_Z + TABLEWARE_CLEARANCE - WINE_GLASS_LOCAL_Z_MIN),
        mass=0.35,
        collision_proxy_cfg=sim_utils.CylinderCfg(
            radius=0.040,
            height=0.006,
            visible=False,
            collision_props=collision_properties(),
            physics_material=rigid_material(
                density=None,
                friction=0.01,
                contact_stiffness=1.0e4,
                contact_damping=1.0e1,
            ),
        ),
        collision_proxy_position=(0.0, 0.0, 0.003),
        center_of_mass=WINE_GLASS_CENTER_OF_MASS,
        diagonal_inertia=WINE_GLASS_DIAGONAL_INERTIA,
        visual_color=(0.55, 0.75, 0.90),
        visual_opacity=0.90,
        visual_roughness=0.12,
    ).replace(prim_path="{ENV_REGEX_NS}/Glass")
    fork = rigid_object_cfg(
        usd_path=FORK_USD,
        position=(0.04, -0.22, CLOTH_Z + TABLEWARE_CLEARANCE - FORK_SCALE * FORK_LOCAL_Z_MIN),
        mass=0.07,
        collision_proxy_cfg=sim_utils.CuboidCfg(
            size=(0.014, 0.160, 0.003),
            visible=False,
            collision_props=collision_properties(),
            physics_material=rigid_material(
                density=None,
                friction=0.50,
                contact_stiffness=1.0e4,
                contact_damping=1.0e1,
            ),
        ),
        collision_proxy_position=(0.0, 0.0092, 0.0016),
        orientation=FORK_ROTATION,
        scale=(FORK_SCALE,) * 3,
        center_of_mass=FORK_CENTER_OF_MASS,
        diagonal_inertia=FORK_DIAGONAL_INERTIA,
    ).replace(prim_path="{ENV_REGEX_NS}/Fork")

    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.75, 0.75, 0.75)),
    )


def _finger_data(robot: Articulation) -> tuple[list[int], list[float], list[int]]:
    indices = []
    values = []
    groups = []
    for side_index, side in enumerate(("L", "R")):
        thumb = THUMB_CLOSED_VALUES[side_index]
        entries = (
            ("thumb_proximal_yaw_joint", thumb[0]),
            ("thumb_proximal_pitch_joint", thumb[1]),
            ("thumb_intermediate_joint", thumb[2]),
            ("thumb_distal_joint", thumb[3]),
            ("index_proximal_joint", 1.2),
            ("index_intermediate_joint", 1.2),
            ("middle_proximal_joint", 1.0),
            ("middle_intermediate_joint", 1.0),
            ("ring_proximal_joint", 1.0),
            ("ring_intermediate_joint", 1.0),
            ("pinky_proximal_joint", 1.0),
            ("pinky_intermediate_joint", 1.0),
        )
        for suffix, value in entries:
            joint_name = f"{side}_{suffix}"
            joint_ids, _ = robot.find_joints(joint_name)
            if len(joint_ids) != 1:
                raise ValueError(f"Expected one H1 joint named '{joint_name}', found {len(joint_ids)}")
            indices.append(joint_ids[0])
            values.append(value)
            if suffix.startswith("thumb_"):
                groups.append(GROUP_LEFT_THUMB if side == "L" else GROUP_RIGHT_THUMB)
            elif suffix.startswith("index_"):
                groups.append(GROUP_LEFT_INDEX if side == "L" else GROUP_RIGHT_INDEX)
            else:
                groups.append(GROUP_OTHER)
    return indices, values, groups


def _make_ik(model: Model, bodies: dict[str, int]) -> NewtonIKSolver:
    from isaaclab_newton.ik import (  # noqa: PLC0415
        NewtonIKJointLimitObjectiveCfg,
        NewtonIKPoseObjectiveCfg,
        NewtonIKSolver,
        NewtonIKSolverCfg,
    )

    objectives = [
        NewtonIKPoseObjectiveCfg(
            body_name="left_hand",
            body_offset_pos=HAND_OFFSETS[0],
            use_relative_mode=False,
            position_weight=5.0,
            rotation_weight=0.2,
        ),
        NewtonIKPoseObjectiveCfg(
            body_name="right_hand",
            body_offset_pos=HAND_OFFSETS[1],
            use_relative_mode=False,
            position_weight=5.0,
            rotation_weight=0.2,
        ),
        NewtonIKPoseObjectiveCfg(
            body_name="torso",
            use_relative_mode=False,
            position_weight=50.0,
            rotation_weight=50.0,
        ),
        NewtonIKJointLimitObjectiveCfg(weight=1.0),
    ]
    return NewtonIKSolver(
        NewtonIKSolverCfg(iterations=24, lambda_initial=0.1),
        model=model,
        num_envs=1,
        device=str(model.device),
        objectives=objectives,
        link_resolver=bodies.__getitem__,
    )


class _H1TableclothController:
    """Generate GPU-resident H1 joint targets for the tablecloth sequence."""

    def __init__(self, robot: Articulation, model: Model, sim: sim_utils.SimulationContext, pull_speed: float):
        if model.joint_coord_count < robot.num_joints:
            raise RuntimeError(
                "Newton IK cannot address every H1 joint; "
                f"the model has {model.joint_coord_count} coordinates and H1 exposes {robot.num_joints} joints"
            )

        self._robot = robot
        self._sim_dt = sim.get_physics_dt()
        self._pull_speed = pull_speed
        device = model.device

        bodies = {
            "torso": _find_suffix(model.body_label, "torso_link"),
            "left_hand": _find_suffix(model.body_label, "left_hand_link"),
            "right_hand": _find_suffix(model.body_label, "right_hand_link"),
        }
        self._ik_solver = _make_ik(model, bodies)
        self._left_objective = self._ik_solver.objectives_by_name["left_hand"]
        self._right_objective = self._ik_solver.objectives_by_name["right_hand"]
        torso_objective = self._ik_solver.objectives_by_name["torso"]

        torso_ids, _ = robot.find_bodies("torso_link")
        initial_torso_pose = robot.data.body_link_pose_w.torch[0, torso_ids[0]].cpu().numpy()
        torso_objective.position_objective.set_target_position(0, wp.vec3(*initial_torso_pose[:3]))
        torso_objective.rotation_objective.set_target_rotation(0, wp.quat(*initial_torso_pose[3:]))
        self._left_objective.rotation_objective.set_target_rotation(0, wp.quat(*_unit_quat(HAND_ROTATIONS[0])))
        self._right_objective.rotation_objective.set_target_rotation(0, wp.quat(*_unit_quat(HAND_ROTATIONS[1])))

        self._keyframes = wp.array(HAND_KEYFRAMES, dtype=wp.vec3, device=device)
        self._phase_durations = wp.array([phase.duration for phase in MOTION_PHASES], dtype=float, device=device)
        self._phase_start_keyframes = wp.array(
            [phase.start_keyframe for phase in MOTION_PHASES], dtype=wp.int32, device=device
        )
        self._phase_end_keyframes = wp.array(
            [phase.end_keyframe for phase in MOTION_PHASES], dtype=wp.int32, device=device
        )
        self._phase_start_fingers = wp.array(
            [fraction for phase in MOTION_PHASES for fraction in phase.start_fingers], dtype=float, device=device
        )
        self._phase_end_fingers = wp.array(
            [fraction for phase in MOTION_PHASES for fraction in phase.end_fingers], dtype=float, device=device
        )
        self._state = wp.zeros(1, dtype=wp.int32, device=device)
        self._state_time = wp.zeros(1, dtype=float, device=device)
        self._pull_distance = wp.zeros(1, dtype=float, device=device)
        self._fractions = wp.zeros(5, dtype=float, device=device)

        finger_indices, closed_values, finger_groups = _finger_data(robot)
        self._finger_count = len(finger_indices)
        self._finger_indices = wp.array(finger_indices, dtype=wp.int32, device=device)
        self._closed_values = wp.array(closed_values, dtype=float, device=device)
        self._finger_groups = wp.array(finger_groups, dtype=wp.int32, device=device)

        self._ik_seed = wp.clone(model.joint_q).reshape((1, model.joint_coord_count))
        self._target_q = robot.actuators.target_command.position.warp.reshape((-1,))
        self._target_qd = robot.actuators.target_command.velocity.warp.reshape((-1,))
        self._initialize_rest_pose(sim)

    def step(self) -> None:
        """Advance the state machine and write its targets to Isaac Lab's command buffers."""
        wp.launch(
            _infer_state_machine,
            dim=1,
            inputs=[
                self._sim_dt,
                self._pull_speed,
                self._keyframes,
                self._phase_durations,
                self._phase_start_keyframes,
                self._phase_end_keyframes,
                self._phase_start_fingers,
                self._phase_end_fingers,
                self._left_objective.position_objective.target_positions,
                self._right_objective.position_objective.target_positions,
                self._fractions,
                self._state,
                self._state_time,
                self._pull_distance,
            ],
        )
        solved = self._ik_solver.solve(self._ik_seed)
        solved_flat = solved.reshape((-1,))
        wp.launch(
            _set_finger_targets,
            dim=self._finger_count,
            inputs=[
                solved_flat,
                self._finger_indices,
                self._closed_values,
                self._finger_groups,
                self._fractions,
            ],
        )
        wp.copy(self._ik_seed, solved)
        wp.launch(
            _update_control_targets,
            dim=self._robot.num_joints,
            inputs=[
                solved_flat,
                self._previous_targets,
                self._sim_dt,
                self._target_q,
                self._target_qd,
            ],
        )

    def _initialize_rest_pose(self, sim: sim_utils.SimulationContext) -> None:
        """Solve and apply the first task-space keyframe before simulation begins."""
        self._left_objective.position_objective.set_target_position(0, wp.vec3(*HAND_KEYFRAMES[0]))
        self._right_objective.position_objective.set_target_position(0, wp.vec3(*HAND_KEYFRAMES[1]))
        self._ik_solver.cfg.iterations = 48
        solved = self._ik_solver.solve(self._ik_seed)
        solved_flat = solved.reshape((-1,))
        robot_position = solved_flat[: self._robot.num_joints].reshape((1, self._robot.num_joints))
        self._robot.write_joint_position_to_sim_index(position=robot_position)
        self._robot.write_joint_velocity_to_sim_index(velocity=wp.zeros_like(robot_position))
        sim.forward()

        self._previous_targets = wp.clone(solved_flat[: self._robot.num_joints])
        wp.copy(self._target_q, self._previous_targets)
        self._target_qd.zero_()
        self._ik_solver.cfg.iterations = 24
        wp.copy(self._ik_seed, solved)


def main() -> None:
    """Launch the H1 tablecloth demo."""
    if not np.isfinite(args_cli.pull_speed) or args_cli.pull_speed <= 0.0:
        raise ValueError("--pull_speed must be finite and positive")
    if args_cli.video and "none" in (args_cli.visualizer or []):
        raise ValueError("--video requires a capture-capable visualizer; omit --visualizer none")
    if args_cli.video and "kit" in (args_cli.visualizer or []):
        args_cli.enable_cameras = True
    max_steps = VIDEO_STEPS if args_cli.video and args_cli.max_steps < 0 else args_cli.max_steps

    physics_cfg = NewtonCfg(
        num_substeps=SUBSTEPS,
        # Robot, tableware, and cloth all move quickly during the pull, so collide every substep.
        collision_decimation=1,
        default_shape_cfg=NewtonShapeCfg(gap=RIGID_GAP, ke=1.0e4, kd=1.0e1, mu=0.35),
        # Material mixing keeps the cloth planted on the high-friction table while
        # allowing it to slide under the lower-friction tableware. The same table
        # material gives the objects more grip after the cloth clears them.
        soft_contact_cfg=NewtonSoftContactCfg(soft_contact_ke=1.0e3, soft_contact_kd=1.0e-2, soft_contact_mu=0.35),
        collision_cfg=NewtonCollisionPipelineCfg(
            broad_phase="sap",
            soft_contact_margin=0.008,
            enable_rigid_soft_full_surface_contact=True,
        ),
        solver_cfg=VBDSolverCfg(
            iterations=15,
            rigid_compliant_alm=True,
            rigid_body_contact_buffer_size=512,
            rigid_body_particle_contact_buffer_size=8192,
            rigid_joint_linear_ke=1.0e6,
            rigid_joint_angular_ke=1.0e6,
            rigid_joint_linear_kd=1.0e2,
            rigid_joint_angular_kd=1.0e2,
        ),
    )
    with launch_simulation(cfg=physics_cfg, launcher_args=args_cli) as resolved_physics_cfg:
        from isaaclab_newton.physics import NewtonManager  # noqa: PLC0415

        from isaaclab.scene import InteractiveScene  # noqa: PLC0415

        default_visualizer_cfg = None
        if args_cli.video and "kit" in (args_cli.visualizer or []):
            from isaaclab_visualizers.kit import KitVisualizerCfg  # noqa: PLC0415

            default_visualizer_cfg = KitVisualizerCfg(
                window_width=VIDEO_RESOLUTION[0], window_height=VIDEO_RESOLUTION[1]
            )

        h1_usd_path = _convert_h1_asset()
        sim = sim_utils.SimulationContext(
            sim_utils.SimulationCfg(
                dt=1.0 / FPS,
                device=args_cli.device,
                physics=resolved_physics_cfg,
                default_visualizer_cfg=default_visualizer_cfg,
            )
        )
        sim.set_camera_view(eye=(-2.435, -2.70, 1.725), target=(-0.08, 0.0, 0.93))
        scene = InteractiveScene(
            H1TableclothSceneCfg(
                num_envs=1,
                env_spacing=1.0,
                robot=_h1_articulation_cfg(h1_usd_path),
            )
        )
        sim.reset()
        sim_dt = sim.get_physics_dt()

        robot: Articulation = scene["robot"]
        controller = _H1TableclothController(robot, NewtonManager.get_model(), sim, args_cli.pull_speed)
        video_recorder = create_video_recorder(
            sim,
            enabled=args_cli.video,
            output_dir=VIDEO_OUTPUT_DIR,
            filename_prefix="newton_tablecloth_h1",
            video_length=max_steps,
            fps=FPS,
        )
        print("[INFO]: Setup complete. H1 Isaac Lab tablecloth state machine is ready.", flush=True)
        step = 0
        try:
            while sim.is_headless_or_exist_active_visualizer() and (max_steps < 0 or step < max_steps):
                controller.step()
                scene.write_data_to_sim()
                sim.step()
                scene.update(sim_dt)
                if video_recorder is not None:
                    video_recorder.step()
                step += 1
        finally:
            if video_recorder is not None:
                video_recorder.close()


if __name__ == "__main__":
    main()
