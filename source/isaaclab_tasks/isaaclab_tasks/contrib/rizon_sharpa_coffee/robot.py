# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rizon4s/Sharpa asset and joint contract shared with the workstation RJ45 demo.

The USD and textures remain in the user's asset cache because the upstream
Fabrics-Sim asset is NVIDIA proprietary. See this task's README for provenance.
"""

from __future__ import annotations

import hashlib
import math
import os
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.assets import ArticulationCfg

ASSET_REVISION = "d0dbd1ddaefc4996db546949a7dfb37e39afcbeb"
ASSET_BUNDLE_SHA256 = "ae5d22792b44fb6d29a7691d4276bc061a5529132f01e7a0eb5795a482595d63"
ASSET_USD_NAME = "rizon4s_sharpa_no_spheres_generated.usd"
ASSET_ROOT_ENV = "ISAACLAB_FABRICS_SIM_RIZON_SHARPA_ROOT"
BASE_BODY_NAME = "base_link"
PALM_BODY_NAME = "r_palm_ctrl"
"""Canonical palm: +X toward the knuckles, +Z out of the palm."""

DEFAULT_SPOON_OFFSET_POS = (0.19638407055658647, -0.0036995580810314636, 0.07643507665596544)
DEFAULT_SPOON_OFFSET_ROT = (1.0, 0.0, 0.0, 0.0)

ARM_JOINT_NAMES = tuple(f"joint{index}" for index in range(1, 8))
ARM_HOME_POSITIONS = (0.0, -0.698, 0.0, 1.571, 0.0, 0.698, 0.0)
"""Reference arm home used by the workstation RJ45 environment [rad]."""
COFFEE_READY_POSITIONS = (
    -1.081903255791,
    0.013804070751,
    0.268143828426,
    2.645703260349,
    2.465847724197,
    0.413775564614,
    -0.366468701795,
)
"""Measured IK branch with the level spoon at ``(0.508, -0.12, 0.22)`` m.

The spoon uses ``DEFAULT_SPOON_OFFSET_POS`` and 180 degrees about palm X.
This places the held utensil above the source bowl, ready to descend.
"""
HAND_JOINT_NAMES = (
    "right_thumb_CMC_FE",
    "right_thumb_CMC_AA",
    "right_thumb_MCP_FE",
    "right_thumb_MCP_AA",
    "right_thumb_IP",
    "right_index_MCP_FE",
    "right_index_MCP_AA",
    "right_index_PIP",
    "right_index_DIP",
    "right_middle_MCP_FE",
    "right_middle_MCP_AA",
    "right_middle_PIP",
    "right_middle_DIP",
    "right_ring_MCP_FE",
    "right_ring_MCP_AA",
    "right_ring_PIP",
    "right_ring_DIP",
    "right_pinky_CMC",
    "right_pinky_MCP_FE",
    "right_pinky_MCP_AA",
    "right_pinky_PIP",
    "right_pinky_DIP",
)
HAND_JOINT_LIMITS = (
    (-0.1745, 1.9199),
    (-0.3491, 0.3491),
    (-0.5236, 1.3963),
    (-0.3491, 0.3491),
    (0.0, 1.7453),
    (-0.17453293, 1.5708),
    (-0.3491, 0.3491),
    (0.0, 1.7453),
    (0.0, 1.3963),
    (-0.17453293, 1.5708),
    (-0.3491, 0.3491),
    (0.0, 1.7453),
    (0.0, 1.3963),
    (-0.17453293, 1.5708),
    (-0.3491, 0.3491),
    (0.0, 1.7453),
    (0.0, 1.3963),
    (0.0, 0.2618),
    (-0.17453293, 1.5708),
    (-0.3491, 0.3491),
    (0.0, 1.7453),
    (0.0, 1.3963),
)
HAND_HOLD_POSITIONS = tuple(
    math.radians(value)
    for value in (
        57.326863284,
        -6.056199517,
        39.987930283,
        -12.580265177,
        64.864070603,
        31.665550149,
        8.050335071,
        84.817118264,
        52.919252691,
        43.862671108,
        6.597351456,
        75.055080927,
        63.272558488,
        86.912901978,
        1.054959321,
        81.046617336,
        52.093328444,
        6.662629932,
        69.243995079,
        3.461576450,
        79.104594737,
        51.017046556,
    )
)
"""Collision-screened pinch at one handle cross-section [rad].

The index distal surface and thumb elastomer oppose each other near the handle
rear, with 0.8 mm cross-finger clearance and no tool penetration. Frictional
retention is not certified; the demo continues to use its mounted utensil.
"""


def configured_asset_path(*, verify: bool = True) -> Path:
    """Resolve the existing RJ45 robot USD, optionally checking all pinned textures."""
    configured = os.environ.get(ASSET_ROOT_ENV)
    root = (
        Path(configured).expanduser()
        if configured
        else Path.home() / ".cache/isaaclab/fabrics-sim/rizon4s-sharpa/sha256" / ASSET_BUNDLE_SHA256
    )
    if root.suffix.lower() in (".usd", ".usda", ".usdc"):
        root = root.parent
    path = root / ASSET_USD_NAME
    if verify:
        _verify_asset_bundle(root.resolve())
    return path


def get_robot_cfg(*, contact_grasp: bool = False) -> ArticulationCfg:
    """Return the 29-joint robot at a tabletop-height base with a curled hand."""
    from isaaclab_newton.sim.schemas import MujocoJointCfg, NewtonArticulationCfg

    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg

    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(_prepared_robot_path(configured_asset_path())),
            articulation_props=NewtonArticulationCfg(self_collision_enabled=False),
            joint_drive_props={"/Physics/.*": [MujocoJointCfg(actuatorgravcomp=True)]},
            activate_contact_sensors=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0),
            joint_pos={
                **dict(zip(ARM_JOINT_NAMES, COFFEE_READY_POSITIONS, strict=True)),
                **dict(zip(HAND_JOINT_NAMES, HAND_HOLD_POSITIONS, strict=True)),
            },
        ),
        actuators={
            "shoulder": ImplicitActuatorCfg(
                joint_names_expr=["joint[1-2]"],
                joint_effort_limit=123.0,
                joint_velocity_limit=2.094,
                stiffness=6000.0,
                damping=108.5,
            ),
            "elbow": ImplicitActuatorCfg(
                joint_names_expr=["joint[3-4]"],
                joint_effort_limit=64.0,
                joint_velocity_limit=2.443,
                stiffness=4200.0,
                damping=90.7,
            ),
            "wrist": ImplicitActuatorCfg(
                joint_names_expr=["joint[5-7]"],
                joint_effort_limit=39.0,
                joint_velocity_limit=4.887,
                stiffness=1500.0,
                damping=54.2,
            ),
            "hand": ImplicitActuatorCfg(
                joint_names_expr=["right_.*"],
                joint_effort_limit=3.3,
                joint_velocity_limit=6.0,
                stiffness=24.0,
                damping=0.2 if contact_grasp else 1.2,
            ),
        },
        soft_joint_pos_limit_factor=0.95,
    )


@lru_cache(maxsize=4)
def _prepared_robot_path(source: Path) -> Path:
    """Reference the licensed robot, omitting malformed visual corner normals.

    Two meshes in the pinned bundle have face-varying normal arrays that do not
    match their corner counts. Newton's USD reader rejects them. Blocking only
    those normals lets the importer recompute them from the original geometry;
    topology, UVs, materials, collisions and the source bundle remain unchanged.
    """
    from pxr import Usd, UsdGeom

    location = hashlib.sha256(str(source.resolve()).encode()).hexdigest()[:12]
    directory = Path.home() / ".cache/isaaclab/rizon-sharpa-coffee/robot-normals-v1" / ASSET_BUNDLE_SHA256
    destination = directory / f"{location}.usda"
    if destination.is_file():
        return destination
    original = Usd.Stage.Open(str(source))
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=directory) as temporary:
        prepared = Path(temporary) / "robot.usda"
        stage = Usd.Stage.CreateNew(str(prepared))
        stage.GetRootLayer().subLayerPaths = [str(source.resolve())]
        stage.SetDefaultPrim(stage.GetPrimAtPath(original.GetDefaultPrim().GetPath()))
        for prim in original.Traverse():
            if not prim.IsA(UsdGeom.Mesh):
                continue
            mesh = UsdGeom.Mesh(prim)
            normals = mesh.GetNormalsAttr().Get()
            if (
                normals is not None
                and mesh.GetNormalsInterpolation() == UsdGeom.Tokens.faceVarying
                and len(normals) != len(mesh.GetFaceVertexIndicesAttr().Get())
            ):
                UsdGeom.Mesh(stage.OverridePrim(prim.GetPath())).GetNormalsAttr().Block()
        stage.GetRootLayer().Save()
        prepared.replace(destination)
    return destination


@lru_cache(maxsize=4)
def _verify_asset_bundle(root: Path) -> None:
    """Check the exact bundle used by the cable task without copying licensed assets."""
    texture_names = [
        "t_Rizon4s_EmissiveMask.1001.png",
        "t_Rizon4s_EmissiveMask.1002.png",
        *[f"t_Rizon4s_{channel}.{tile}.png" for channel in ("alb", "nor", "orm") for tile in (1001, 1002, 1003)],
        "t_Sharpa_alb.png",
        "t_Sharpa_nor.png",
        "t_Sharpa_orm.png",
    ]
    digest = hashlib.sha256()
    for relative_path in [ASSET_USD_NAME, *[f"textures/{name}" for name in texture_names]]:
        path = root / relative_path
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing Rizon/Sharpa asset: {path}. Set {ASSET_ROOT_ENV} to the existing RJ45 USD bundle."
            )
        with path.open("rb") as stream:
            file_digest = hashlib.file_digest(stream, "sha256").hexdigest()
        digest.update(f"{file_digest}  {relative_path}\n".encode())
    if digest.hexdigest() != ASSET_BUNDLE_SHA256:
        raise ValueError(f"Rizon/Sharpa asset bundle at {root} differs from the pinned RJ45 asset.")
