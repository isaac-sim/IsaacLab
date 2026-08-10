# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Topple an NVIDIA-logo domino layout with Newton viewer dragging.

Right-click and drag the black trigger slab on the right to start the cascade
across all rows.

.. code-block:: bash

    uv run python scripts/demos/newton_viewer_dominoes.py
"""

import argparse
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description="NVIDIA-logo domino dragging demo (XPBD).")
parser.add_argument("--max_steps", type=int, default=-1, help="Stop after this many steps; negative runs forever.")
add_launcher_args(parser)
parser.set_defaults(visualizer=["newton_gl"])
args_cli = parser.parse_args()

import torch
from isaaclab_newton.physics import NewtonCfg, NewtonManager, NewtonShapeCfg, XPBDSolverCfg

from pxr import Gf, UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils.configclass import configclass

DOMINO_SIZE = (0.12, 0.032, 0.36)
DOMINO_SPACING = 0.12
LOGO_FOOTPRINT = (29.4, 8.4)
NVIDIA_GREEN = (0.24, 0.50, 0.0)

_POSES_PATH = Path(__file__).with_name("assets") / "nvidia_logo_domino_poses.pth"
_POSES = torch.load(_POSES_PATH, map_location="cpu", weights_only=True).tolist()
LOGO_DOMINO_POSES = [(tuple(pose[:3]), tuple(pose[3:])) for pose in _POSES]


def _set_display_color(prim_path: str, color: tuple[float, float, float]) -> None:
    """Set a mesh display color for the Newton model builder."""
    mesh = sim_utils.get_current_stage().GetPrimAtPath(f"{prim_path}/geometry/mesh")
    UsdGeom.Gprim(mesh).CreateDisplayColorAttr().Set([Gf.Vec3f(*color)])


def _domino_cfg(position: tuple[float, float, float], orientation: tuple[float, float, float, float]) -> RigidObjectCfg:
    """Return one domino at a saved pose."""
    return RigidObjectCfg(
        prim_path="",
        spawn=sim_utils.CuboidCfg(
            size=DOMINO_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(density=580.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.15,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=position, rot=orientation),
    )


def _trigger_cfg() -> RigidObjectCfg:
    """Place a low-mass black trigger slab across every row on the right."""
    right_edge = max(position[0] for position, _ in LOGO_DOMINO_POSES)
    return RigidObjectCfg(
        prim_path="/World/Dominoes/Trigger",
        spawn=sim_utils.CuboidCfg(
            size=(DOMINO_SIZE[1], LOGO_FOOTPRINT[1], DOMINO_SIZE[2]),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(density=20.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.15,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(right_edge + DOMINO_SPACING, 0.0, DOMINO_SIZE[2] / 2.0)),
    )


@configclass
class DominoSceneCfg(InteractiveSceneCfg):
    """White floor, saved domino poses, and the trigger slab."""

    floor: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/Floor",
        spawn=sim_utils.CuboidCfg(
            size=(LOGO_FOOTPRINT[0] + 4.0, LOGO_FOOTPRINT[1] + 4.0, 0.10),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.15,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.05)),
    )
    dominoes: RigidObjectCollectionCfg = RigidObjectCollectionCfg(
        rigid_objects={
            f"domino_{index:04d}": _domino_cfg(position, orientation).replace(
                prim_path=f"/World/Dominoes/Domino{index:04d}"
            )
            for index, (position, orientation) in enumerate(LOGO_DOMINO_POSES)
        }
    )
    trigger: RigidObjectCfg = _trigger_cfg()


def _apply_display_colors() -> None:
    """Apply viewer colors after InteractiveScene has authored the prims."""
    _set_display_color("/World/Floor", (1.0, 1.0, 1.0))
    for index in range(len(LOGO_DOMINO_POSES)):
        _set_display_color(f"/World/Dominoes/Domino{index:04d}", NVIDIA_GREEN)
    _set_display_color("/World/Dominoes/Trigger", (0.02, 0.02, 0.02))


def run_simulator(sim: sim_utils.SimulationContext) -> None:
    """Run until the viewer closes or the optional step limit is reached."""
    step_count = 0
    while sim.is_headless_or_exist_active_visualizer() and (args_cli.max_steps < 0 or step_count < args_cli.max_steps):
        sim.step()
        step_count += 1


def main() -> None:
    """Launch the Newton XPBD domino dragging demo."""
    physics_cfg = NewtonCfg(
        num_substeps=10,
        collision_decimation=1,
        default_shape_cfg=NewtonShapeCfg(gap=0.001, ke=1.0e4, kd=0.0, mu=1.0),
        solver_cfg=XPBDSolverCfg(iterations=20, enable_restitution=True),
    )
    with launch_simulation(cfg=physics_cfg, launcher_args=args_cli) as resolved_physics_cfg:
        sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args_cli.device, physics=resolved_physics_cfg)
        sim = sim_utils.SimulationContext(sim_cfg)
        sim.set_camera_view(eye=(0.0, -18.0, 15.0), target=(0.0, 0.0, 0.0))
        _scene = InteractiveScene(DominoSceneCfg(num_envs=1, env_spacing=1.0))
        _apply_display_colors()
        if NewtonManager._builder is None:
            NewtonManager.instantiate_builder_from_stage()
        NewtonManager._builder.rigid_gap = 0.001
        sim.reset()
        print(
            f"[INFO]: Setup complete with {len(LOGO_DOMINO_POSES)} green dominoes. "
            "Right-click and drag the black trigger slab on the right to topple the logo.",
            flush=True,
        )
        run_simulator(sim)


if __name__ == "__main__":
    main()
