# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compare contact-driven bimanual tablecloth pulls at five speeds.

InteractiveScene declares fixed-base H2 Plus robots with SHARPA hands, cloths, and tableware.
Newton VBD simulates the articulations, cloth, and tableware together; Newton IK and a Warp
state machine run inside the physics CUDA graph. This is scripted manipulation, not an RL task.

.. code-block:: bash

    uv run isaaclab example tablecloth --device cuda:0
    uv run --extra importers isaaclab example tablecloth --robot h1 --device cuda:0
"""

from __future__ import annotations

import argparse
import logging
import math
import tempfile
from contextlib import nullcontext
from functools import partial

import numpy as np

from isaaclab.app import add_launcher_args, launch_simulation

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Five side-by-side GPU humanoid tablecloth pulls.")
parser.add_argument(
    "--robot", choices=("h2", "h1"), default="h2", help="H2 Plus / SHARPA (default), or the H1 fallback."
)
parser.add_argument(
    "--speeds",
    nargs="+",
    type=float,
    default=[4.0, 2.0, 1.5, 0.75, 0.5],
    help="Peak pull speeds, ordered left to right from fastest to slowest [m/s].",
)
parser.add_argument(
    "--max_steps",
    type=int,
    default=0,
    help="Frame limit; zero completes all pulls and a final hold, negative runs forever.",
)
add_launcher_args(parser)
parser.set_defaults(visualizer=["newton_gl"])
args_cli = parser.parse_args()
if not args_cli.device.startswith("cuda"):
    parser.error("This graph-captured example requires a CUDA device.")
if any(not math.isfinite(speed) or speed <= 0 for speed in args_cli.speeds):
    parser.error("--speeds must contain positive finite values.")

from isaaclab_newton.physics import (
    NewtonCfg,
    NewtonCollisionPipelineCfg,
    NewtonShapeCfg,
    NewtonSoftContactCfg,
    VBDSolverCfg,
)
from isaaclab_newton.sim.spawners.materials import NewtonSurfaceDeformableBodyMaterialCfg

import isaaclab.sim as sim_utils
from isaaclab import cloner
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import instantiate
from isaaclab.utils.version import standalone_importers_available
from isaaclab.visualizers import VisualizerCfg

from isaaclab_contrib.assets.tablecloth import (
    CLOTH_FRICTION,
    FPS,
    GROUND_CFG,
    H1_CFG,
    H2_CFG,
    LIGHT_CFG,
    RIGID_GAP,
    SUBSTEPS,
    TABLECLOTH_ASSETS,
    prepare_native_h2_contacts,
)


def main() -> None:
    """Construct the scene, capture GPU control, and run the complete comparison."""
    speeds = tuple(args_cli.speeds)
    step_dt = 1.0 / FPS
    is_h2 = args_cli.robot == "h2"
    # Resolve the native SHARPA pinch during the fastest pull without changing the H1 fallback.
    substeps = 64 if is_h2 else SUBSTEPS
    collision_cfg = NewtonCollisionPipelineCfg(
        broad_phase="sap",
        # Identical, separated lanes need linear capacity, not shapes * particles.
        soft_contact_max=65_536 * len(speeds),
        soft_contact_margin=0.008,
        enable_rigid_soft_full_surface_contact=True,
    )
    solver_cfg = VBDSolverCfg(
        iterations=60 if is_h2 else 30,
        friction_epsilon=1.0e-3 if is_h2 else 1.0e-2,
        rigid_compliant_alm=True,
        rigid_joint_linear_ke=1.0e7,
        rigid_joint_angular_ke=1.0e6 if is_h2 else 1.0e7,
        rigid_body_contact_buffer_size=256,
        rigid_body_particle_contact_buffer_size=8192,
    )
    camera_distance = max(3.0, 1.6 * len(speeds))
    sim_cfg = sim_utils.SimulationCfg(
        dt=step_dt / substeps if is_h2 else step_dt,
        render_interval=substeps if is_h2 else 1,
        physics=NewtonCfg(
            num_substeps=1 if is_h2 else substeps,
            collision_decimation=0 if is_h2 else 1,
            default_shape_cfg=NewtonShapeCfg(gap=RIGID_GAP, ke=1.0e4, kd=1.0e1, mu=0.35),
            soft_contact_cfg=NewtonSoftContactCfg(
                soft_contact_ke=1.0e3,
                soft_contact_kd=1.0e-2,
                soft_contact_mu=CLOTH_FRICTION,
            ),
            collision_cfg=collision_cfg,
            solver_cfg=solver_cfg,
        ),
        default_visualizer_cfg=VisualizerCfg(
            eye=(camera_distance, -1.5, 1.0 + 0.35 * camera_distance), lookat=(-0.3, 0.0, 1.0)
        ),
    )
    args_cli.require_kit = args_cli.robot == "h1" and not standalone_importers_available()
    with (
        launch_simulation(cfg=sim_cfg, launcher_args=args_cli),
        tempfile.TemporaryDirectory(prefix="isaaclab-tablecloth-") as asset_dir,
    ):
        from isaaclab_newton.cloner import newton_builder_world_hook  # noqa: PLC0415
        from isaaclab_newton.physics import NewtonManager  # noqa: PLC0415

        from isaaclab.scene import InteractiveScene  # noqa: PLC0415
        from isaaclab.utils.assets import retrieve_file_path  # noqa: PLC0415

        from isaaclab_contrib.controllers.tablecloth import GraspMonitor, TableclothController  # noqa: PLC0415

        if args_cli.robot == "h1":
            import newton.utils  # noqa: PLC0415

            from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg  # noqa: PLC0415

            usd_path = MjcfConverter(
                MjcfConverterCfg(
                    asset_path=str(newton.utils.download_asset("unitree_h1") / "mjcf/h1_with_hand.xml"),
                    usd_dir=asset_dir,
                    usd_file_name="h1_with_hand.usda",
                    fix_base=True,
                    self_collision=False,
                    robot_type="Humanoid",
                    run_asset_transformer=False,
                    run_multi_physics_conversion=False,
                )
            ).usd_path
        else:
            usd_path = retrieve_file_path(H2_CFG.spawn.usd_path)
        sim = sim_utils.SimulationContext(sim_cfg)
        scene = InteractiveScene(InteractiveSceneCfg(num_envs=len(speeds), env_spacing=1.5))
        positions = np.zeros((len(speeds), 3), dtype=np.float32)
        positions[:, 1] = (np.arange(len(speeds), dtype=np.float32) - 0.5 * (len(speeds) - 1)) * 1.5
        # Each trial has its own physics world, but their visual origins form one row.
        sim_utils.standardize_xform_ops(
            scene.stage.GetPrimAtPath(scene.env_prim_paths[0]), translation=tuple(map(float, positions[0]))
        )
        cfgs = {name: cfg.copy() for name, cfg in TABLECLOTH_ASSETS.items()}
        if args_cli.robot == "h1":
            # H1's smaller hands use the original corner-grasp arrangement.
            cfgs["cloth"].spawn.size = (0.48, 0.80)
        else:
            # Put the front hem between the prepositioned native index and thumb surfaces.
            cloth_pos = cfgs["cloth"].init_state.pos
            cfgs["cloth"].init_state.pos = (-0.06, cloth_pos[1], cloth_pos[2])
            # Thin cloth does not need the H1 configuration's heavy membrane damping.
            material = cfgs["cloth"].spawn.physics_material
            defaults = NewtonSurfaceDeformableBodyMaterialCfg()
            material.tri_ke, material.tri_ka, material.tri_kd = defaults.tri_ke, defaults.tri_ka, defaults.tri_kd
        cfgs["robot"] = (H2_CFG if args_cli.robot == "h2" else H1_CFG).copy()
        cfgs["robot"].spawn.usd_path = usd_path
        cfgs.update(ground=GROUND_CFG.copy(), light=LIGHT_CFG.copy())
        plan = cloner.clone_plan_from_env_0(scene.cfg.clone_cfg, cfgs.values(), len(speeds), 1.5, positions=positions)
        scene.articulations["robot"] = instantiate(cfgs["robot"])
        scene.deformable_objects["cloth"] = instantiate(cfgs["cloth"])
        for name in ("bowl", "glass", "fork"):
            scene.rigid_objects[name] = instantiate(cfgs[name])
        for name, cfg in cfgs.items():
            if name not in ("robot", "cloth", "bowl", "glass", "fork"):
                scene.extras[name] = instantiate(cfg)
        contact_hook = partial(prepare_native_h2_contacts, device=sim.device, mesh_sdf_cache={})
        with newton_builder_world_hook(contact_hook) if is_h2 else nullcontext():
            cloner.replicate(plan, replicate_physics=scene.cfg.replicate_physics)
        sim.reset()
        robot = scene["robot"]
        robot.write_joint_state_to_sim_index(
            position=robot.data.default_joint_pos.warp,
            velocity=robot.data.default_joint_vel.warp,
        )
        # Fixed roots already have the cloned row origins; defaults are environment-local.
        sim.forward()
        controller = TableclothController(robot, usd_path, step_dt, speeds, robot_type=args_cli.robot)
        if args_cli.robot == "h2":
            controller.observe_scene(NewtonManager.get_model(), NewtonManager.get_state_0(), scene.env_prim_paths)
        controller.compute()
        if is_h2:
            controller.bind_arm_substeps(
                NewtonManager.get_model(),
                NewtonManager.get_state_0(),
                NewtonManager.get_control(),
                scene.env_prim_paths,
                substeps,
            )
            NewtonManager.activate_newton_actuator_path()
            NewtonManager.register_post_actuator_callback(controller.interpolate_arm_targets)
            NewtonManager.set_decimation(substeps)
        NewtonManager.register_post_step_callback(controller.compute)
        grasp = GraspMonitor(NewtonManager.get_model(), NewtonManager.get_state_0(), controller, scene.env_prim_paths)
        NewtonManager.register_post_step_callback(grasp.compute)
        complete_steps = math.ceil(TableclothController.duration(speeds, step_dt, args_cli.robot) / step_dt)
        max_steps = args_cli.max_steps or complete_steps
        logger.info("GPU %s tablecloth pulls ready: %s m/s.", args_cli.robot.upper(), speeds)
        try:
            step = 0
            while sim.is_running() and (max_steps < 0 or step < max_steps):
                sim.step()
                scene.update(step_dt)
                step += 1
            if step >= complete_steps:
                reports = grasp.report()
                logger.info("Physical cloth grips: %s", reports)
                if not all(report["retained"] for report in reports):
                    raise RuntimeError("A robot lost its physical cloth grasp; see the material-point gaps above.")
        finally:
            NewtonManager.unregister_post_step_callback(grasp.compute)
            NewtonManager.unregister_post_step_callback(controller.compute)


if __name__ == "__main__":
    main()
