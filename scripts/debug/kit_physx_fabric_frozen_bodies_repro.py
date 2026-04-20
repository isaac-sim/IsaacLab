# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Minimal repro: PhysX + Fabric + Kit visualizer — articulation meshes frozen while simulation runs.

On some Isaac Sim **develop / TOT** builds, the Kit viewport stops updating rigid articulation transforms
when Fabric is enabled and USD sync is off (default for performance), even though PhysX continues to
simulate. Debug geometry driven by tensors (e.g. :class:`~isaaclab.markers.VisualizationMarkers`) can
still move because it writes USD directly.

**What you should see on a failing build**

- Cartpole (or similar) **mesh pose frozen** in the viewport.
- **Red arrow** above the base follows the simulated root position.
- **Console** prints changing root Z (tensor readout) every ``--log-every`` steps.
- **Play/pause** or **viewport switch** refreshes the mesh once (same class of issue as IsaacLab
  issue 4279 / ``PhysxManager._re_sync_fabric()``, where Fabric can stop pushing articulation poses
  until something forces a re-sync).

**Baselines**

- ``--no-kit``: no :class:`~isaaclab_visualizers.kit.KitVisualizer`; uses default render path only.
- ``--no-fabric``: disables Fabric and turns USD sync on (may avoid the bug; slower).

**Kit / viewport**

:class:`~isaaclab.app.AppLauncher` forces **headless** mode unless you pass ``--viz …`` on the command line
*or* attach ``visualizer_intent`` (see :func:`isaaclab_tasks.utils.sim_launcher._compute_visualizer_intent`).
This script sets ``visualizer_intent`` so the **full Isaac Sim GUI** starts and ``omni.kit.viewport`` can load.
If you pass ``--viz none`` explicitly, that intent is not applied.

The Kit visualizer needs ``omni.kit.viewport``. If that module is still missing after a GUI launch, the
script **falls back to ``--no-kit``** unless you pass ``--require-kit`` (then it exits with an error).

**Usage** (from the Isaac Lab repo root)::

    ./isaaclab.sh -p scripts/debug/kit_physx_fabric_frozen_bodies_repro.py
    ./isaaclab.sh -p scripts/debug/kit_physx_fabric_frozen_bodies_repro.py --no-kit
    ./isaaclab.sh -p scripts/debug/kit_physx_fabric_frozen_bodies_repro.py --no-fabric

"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Minimal Kit + PhysX + Fabric frozen-body repro.")
parser.add_argument("--no-kit", action="store_true", help="Do not attach KitVisualizer (default viewport only).")
parser.add_argument(
    "--require-kit",
    action="store_true",
    help="Exit with an error if Kit visualizer cannot initialize (default: fall back to --no-kit).",
)
parser.add_argument("--no-fabric", action="store_true", help="Disable Fabric (enable USD physics sync).")
parser.add_argument(
    "--no-arrow-marker",
    action="store_true",
    help="Skip the red arrow PointInstancer (avoids Fabric point-instancer warnings on some builds).",
)
parser.add_argument(
    "--log-every",
    type=int,
    default=120,
    help="Print simulated root Z from tensors every N steps (0 = disable).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Start the full Isaac Sim SimulationApp (GUI), not headless: without visualizer_intent, AppLauncher assumes
# no Kit visualizer in upstream config and sets headless=True — then omni.kit.viewport is never available.
# Training scripts set this via isaaclab_tasks.utils.sim_launcher; standalone scripts must set it here.
# Do not override when the user explicitly passed --viz/--visualizer (e.g. --viz none).
if not getattr(args_cli, "visualizer_explicit", False):
    setattr(
        args_cli,
        "visualizer_intent",
        {"has_any_visualizers": True, "has_kit_visualizer": True},
    )

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import logging
import torch
import warp as wp

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import RED_ARROW_X_MARKER_CFG
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab_physx.physics import PhysxCfg

logger = logging.getLogger(__name__)

try:
    from isaaclab_visualizers.kit import KitVisualizerCfg
except ImportError as exc:
    KitVisualizerCfg = None  # type: ignore[misc, assignment]
    _kit_import_error = exc
else:
    _kit_import_error = None


def _kit_viewport_extension_available() -> bool:
    """True when the running Kit app exposes omni.kit.viewport (full GUI / viewport stack)."""
    try:
        import omni.kit.viewport.utility  # noqa: F401

        return True
    except ImportError:
        return False


def design_scene() -> tuple[dict[str, Articulation], list[list[float]]]:
    """Ground, dome light, one cartpole."""
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    origins = [[0.0, 0.0, 0.0]]
    sim_utils.create_prim("/World/Origin1", "Xform", translation=origins[0])

    from isaaclab_assets import CARTPOLE_CFG

    cartpole_cfg = CARTPOLE_CFG.copy()
    cartpole_cfg.prim_path = "/World/Origin.*/Robot"
    cartpole = Articulation(cfg=cartpole_cfg)
    return {"robot": cartpole}, origins


def run_loop(
    sim: SimulationContext,
    robot: Articulation,
    origins: torch.Tensor,
    vel_marker: VisualizationMarkers | None,
    log_every: int,
):
    sim_dt = sim.get_physics_dt()
    count = 0
    while simulation_app.is_running():
        if count % 500 == 0:
            count = 0
            root_pose = wp.to_torch(robot.data.default_root_pose).clone()
            root_pose[:, :3] += origins
            robot.write_root_pose_to_sim_index(root_pose=root_pose)
            root_vel = wp.to_torch(robot.data.default_root_vel).clone()
            robot.write_root_velocity_to_sim_index(root_velocity=root_vel)
            joint_pos = wp.to_torch(robot.data.default_joint_pos).clone()
            joint_vel = wp.to_torch(robot.data.default_joint_vel).clone()
            joint_pos += torch.rand_like(joint_pos) * 0.1
            robot.write_joint_position_to_sim_index(position=joint_pos)
            robot.write_joint_velocity_to_sim_index(velocity=joint_vel)
            robot.reset()
            logger.info("[repro] Reset robot state.")

        efforts = torch.randn_like(wp.to_torch(robot.data.joint_pos)) * 5.0
        robot.set_joint_effort_target_index(target=efforts)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim_dt)

        root_w = wp.to_torch(robot.data.root_pos_w)
        if vel_marker is not None:
            arrow_pos = root_w.clone()
            arrow_pos[:, 2] += 0.65
            ident = torch.zeros(1, 4, device=sim.device, dtype=torch.float32)
            ident[:, 3] = 1.0
            vel_marker.visualize(translations=arrow_pos, orientations=ident)

        if log_every > 0 and count % log_every == 0:
            z = float(root_w[0, 2].item())
            logger.info("[repro] tensor root_pos_w[0].z = %.5f (if this moves but the mesh does not, repro succeeded).", z)
        count += 1


def main():
    use_kit = not args_cli.no_kit
    if use_kit:
        if KitVisualizerCfg is None:
            msg = (
                "Kit visualizer requested but isaaclab_visualizers[kit] is not available: "
                f"{_kit_import_error!s}. Install or use --no-kit."
            )
            if args_cli.require_kit:
                raise RuntimeError(msg)
            logger.warning("%s Running without Kit.", msg)
            use_kit = False
        elif not _kit_viewport_extension_available():
            msg = (
                "omni.kit.viewport is not available in this Kit process. "
                "Use the full Isaac Sim GUI app (not a minimal/headless python that omits viewport extensions), "
                "or pass --no-kit. "
                "If you use ./isaaclab.sh, avoid forcing --visualizer kit unless the viewport stack is present."
            )
            if args_cli.require_kit:
                raise RuntimeError(msg)
            logger.warning("%s Falling back to no Kit visualizer.", msg)
            use_kit = False

    viz_cfg = KitVisualizerCfg() if use_kit else []

    sim_cfg = SimulationCfg(
        device=args_cli.device,
        physics=PhysxCfg(),
        use_fabric=not args_cli.no_fabric,
        visualizer_cfgs=viz_cfg,
    )
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])

    entities, origin_list = design_scene()
    origins = torch.tensor(origin_list, device=sim.device)
    robot = entities["robot"]

    sim.reset()
    # Create PointInstancer markers after reset so Fabric has already attached to the stage (fewer prototype warnings).
    vel_marker: VisualizationMarkers | None = None
    if not args_cli.no_arrow_marker:
        marker_cfg = RED_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/repro_velocity_arrow")
        vel_marker = VisualizationMarkers(marker_cfg)

    logger.info(
        "[repro] use_fabric=%s  kit_visualizer=%s  arrow_marker=%s  (compare with --no-fabric / --no-kit)",
        sim.cfg.use_fabric,
        use_kit,
        vel_marker is not None,
    )
    run_loop(sim, robot, origins, vel_marker, args_cli.log_every)


if __name__ == "__main__":
    main()
    simulation_app.close()
