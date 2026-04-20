# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Minimal PhysX + Fabric frozen-articulation repro using **Isaac Sim core APIs only** (no Isaac Lab).

This mirrors the scenario in ``kit_physx_fabric_frozen_bodies_repro.py`` but avoids
:class:`~isaaclab.sim.SimulationContext`, :class:`~isaaclab.assets.Articulation`, and task/env code.

It launches :class:`isaacsim.SimulationApp`, builds a scene with :class:`~isaacsim.core.api.world.World`,
spawns a **Cartpole** articulation from Omniverse Nucleus, enables ``omni.physx.fabric`` with USD sync
disabled (same idea as :meth:`isaaclab_physx.physics.physx_manager.PhysxManager._load_fabric`), and steps
physics in a loop.

**Run** (from Isaac Lab repo, same Python as Isaac Sim)::

    ./isaaclab.sh -p scripts/debug/kit_physx_fabric_frozen_bodies_repro_isaacsim.py

**Options**

- ``--no-fabric``: disable Fabric and turn USD physics sync on (baseline).
- ``--headless``: headless SimulationApp (viewport / ``omni.kit.viewport`` not loaded).

**Nucleus**

Requires a reachable Omniverse assets root (same as other Isaac Sim samples). Cartpole is resolved from
common paths under ``get_assets_root_path()``; if loading fails, check Nucleus login / VPN.

**Compare**

- Isaac Lab–based repro: ``scripts/debug/kit_physx_fabric_frozen_bodies_repro.py``
"""

from __future__ import annotations

import argparse
import logging

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# SimulationApp must be constructed before most other isaacsim/omni imports.
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Isaac-Sim-only PhysX + Fabric frozen-body repro.")
parser.add_argument("--headless", action="store_true", help="Run SimulationApp headless.")
parser.add_argument(
    "--no-fabric",
    action="store_true",
    help="Disable omni.physx.fabric and enable USD physics sync (baseline).",
)
parser.add_argument(
    "--log-every",
    type=int,
    default=120,
    help="Print root world Z from articulation view every N steps (0 = disable).",
)
args_cli = parser.parse_args()

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp(
    {
        "headless": args_cli.headless,
        "width": 1280,
        "height": 720,
    }
)


def _configure_physx_fabric(*, use_fabric: bool) -> tuple[object | None, object | None]:
    """Enable/disable Fabric and USD sync flags (aligned with Isaac Lab PhysXManager._load_fabric)."""
    import carb
    from omni.kit.app import get_app

    settings = carb.settings.get_settings()
    ext_mgr = get_app().get_extension_manager()

    fabric = None
    update_fabric = None

    if use_fabric:
        if not ext_mgr.is_extension_enabled("omni.physx.fabric"):
            ext_mgr.set_extension_enabled_immediate("omni.physx.fabric", True)
        from omni.physxfabric import get_physx_fabric_interface

        fabric = get_physx_fabric_interface()
        update_fabric = getattr(fabric, "force_update", fabric.update)
        for key in (
            "updateToUsd",
            "updateParticlesToUsd",
            "updateVelocitiesToUsd",
            "updateForceSensorsToUsd",
            "updateResidualsToUsd",
        ):
            settings.set_bool(f"/physics/{key}", False)
        settings.set_bool("/physics/visualizationDisplaySimulationOutput", False)
    else:
        if ext_mgr.is_extension_enabled("omni.physx.fabric"):
            ext_mgr.set_extension_enabled_immediate("omni.physx.fabric", False)
        for key in (
            "updateToUsd",
            "updateParticlesToUsd",
            "updateVelocitiesToUsd",
            "updateForceSensorsToUsd",
            "updateResidualsToUsd",
        ):
            settings.set_bool(f"/physics/{key}", True)
        settings.set_bool("/physics/visualizationDisplaySimulationOutput", False)

    return fabric, update_fabric


def _get_current_stage_id() -> int:
    """USD stage id for omni.physics.tensors (same idea as Isaac Lab get_current_stage_id)."""
    import omni.usd
    from pxr import UsdUtils

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return -1
    cache = UsdUtils.StageCache.Get()
    stage_id = cache.GetId(stage).ToLongInt()
    if stage_id < 0:
        if not stage.GetRootLayer():
            return -1
        stage_id = cache.Insert(stage).ToLongInt()
    return stage_id


def _resolve_cartpole_usd_path() -> str:
    import isaacsim.core.utils.nucleus as nucleus_utils

    root = nucleus_utils.get_assets_root_path()
    if root is None:
        raise RuntimeError(
            "Nucleus / assets root is not set (get_assets_root_path() returned None). "
            "Log in to Omniverse Nucleus or configure Isaac Sim assets per NVIDIA docs."
        )
    rel_candidates = (
        "/Isaac/IsaacLab/Robots/Classic/Cartpole/cartpole.usd",
        "/Isaac/Robots/Classic/Cartpole/cartpole.usd",
    )
    for rel in rel_candidates:
        path = f"{root.rstrip('/')}{rel}"
        try:
            import omni.client

            result, _ = omni.client.stat(path)
            if result == omni.client.Result.OK:
                logger.info("[repro] Using cartpole USD: %s", path)
                return path
        except Exception:
            continue
    fallback = f"{root.rstrip('/')}{rel_candidates[0]}"
    logger.warning("[repro] Could not stat cartpole candidates; trying default: %s", fallback)
    return fallback


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    import omni.kit.app
    import omni.timeline
    import isaacsim.core.utils.prims as prim_utils
    import isaacsim.core.utils.nucleus as nucleus_utils
    from isaacsim.core.api.world import World
    from isaacsim.core.prims import Articulation
    from isaacsim.core.utils.viewports import set_camera_view

    if nucleus_utils.get_assets_root_path() is None:
        raise RuntimeError(
            "Unable to resolve Omniverse assets root. See Isaac Sim docs for Nucleus / content setup."
        )

    use_fabric = not args_cli.no_fabric
    fabric, update_fabric = _configure_physx_fabric(use_fabric=use_fabric)
    logger.info("[repro] fabric=%s", use_fabric)

    physics_dt = 1.0 / 60.0
    rendering_dt = 1.0 / 60.0
    world = World(physics_dt=physics_dt, rendering_dt=rendering_dt, backend="torch", device="cuda:0")

    world.scene.add_default_ground_plane()
    prim_utils.create_prim("/World/Light", "SphereLight", translation=(4.5, 3.5, 10.0))

    usd_path = _resolve_cartpole_usd_path()
    prim_utils.create_prim(
        "/World/Cartpole",
        usd_path=usd_path,
        translation=(0.0, 0.0, 0.6),
    )
    robot = world.scene.add(Articulation("/World/Cartpole", name="cartpole"))

    set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 0.0])

    world.reset()

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()

    # One tensor simulation view for articulation kinematics → Fabric (see PhysxManager.forward).
    sim_view = None
    if use_fabric:
        import omni.physics.tensors as physx_tensors

        sid = _get_current_stage_id()
        if sid > 0:
            sim_view = physx_tensors.create_simulation_view("warp", stage_id=sid)
            if sim_view is not None:
                sim_view.set_subspace_roots("/")

    step = 0
    while simulation_app.is_running():
        if world.is_stopped():
            break

        world.step(render=True)

        # Match Isaac Lab PhysxManager.forward(): articulation kinematics, then Fabric update for Hydra.
        if use_fabric and sim_view is not None and update_fabric is not None and timeline.is_playing():
            sim_view.update_articulations_kinematic()
            update_fabric(0.0, 0.0)

        if args_cli.log_every > 0 and step % args_cli.log_every == 0:
            pos_w, _ = robot.get_world_poses()
            z = float(pos_w[0, 2].item()) if pos_w.numel() >= 3 else float("nan")
            logger.info(
                "[repro] step=%d tensor root z=%.5f (if z moves but the mesh is frozen, repro succeeded).",
                step,
                z,
            )

        # Keep Kit responsive (similar in spirit to KitVisualizer pumping).
        omni.kit.app.get_app().update()
        step += 1


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
