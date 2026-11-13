# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Move a kinematic sphere through settled sand using GPU particle sleeping.

.. code-block:: bash

    uv run --no-sync isaaclab example mpm-sleeping
    uv run --no-sync isaaclab example mpm-sleeping --paging
    uv run --no-sync isaaclab example mpm-sleeping --no-sleeping

Requires the companion Newton sleeping/working-set implementation and Warp's
built conditional-allocation support. The default pinned dependencies do not
provide these experimental APIs yet. All particle history stays in GPU memory;
optional paging groups evaluation slots rather than providing out-of-core storage.
Settling, allocation, and final validation occur outside the captured animation.
All modes use the same guarded sparse runtime; --no-sleeping evaluates the full bed.
Sleeping is approximate and can suppress slow creep; validate interaction forces
against full MPM for your task.
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import warp as wp

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--max_steps", type=int, default=480, help="Frames; negative runs until the viewer closes.")
parser.add_argument("--settle-time", type=float, default=10.0, help="Initial settling duration [s].")
parser.add_argument("--sleeping", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--paging", action="store_true", help="Use 32-particle evaluation pages instead of individual IDs.")
add_launcher_args(parser)
parser.set_defaults(visualizer=["newton_gl"])
args_cli = parser.parse_args()
if args_cli.settle_time < 0.0 or not math.isfinite(args_cli.settle_time):
    parser.error("--settle-time must be finite and nonnegative")
if args_cli.paging and not args_cli.sleeping:
    parser.error("--paging requires sleeping")

FPS = 60.0
SUBSTEPS = 4
VOXEL_SIZE = 0.04
BALL_RADIUS = 0.12


@wp.kernel(enable_backward=False)
def _move_ball(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    time: wp.array[float],
    enabled: wp.array[int],
    dt: float,
    height: float,
):
    if enabled[0] == 0:
        return
    omega = 2.0 * wp.pi / 8.0
    t = time[0] + dt
    phase = omega * t
    position = wp.vec3(0.65 * wp.sin(phase), 0.55 * wp.sin(2.0 * phase), height)
    velocity = omega * wp.vec3(0.65 * wp.cos(phase), 1.1 * wp.cos(2.0 * phase), 0.0)
    body_q[0] = wp.transform(position, wp.quat_identity())
    body_qd[0] = wp.spatial_vector(velocity, wp.vec3(0.0))
    time[0] = t


def create_sim_cfg():
    """Keep the kinematic controller inside the solver's captured substep."""
    from isaaclab_newton.physics import MPMSleepingCfg, MPMSolver, MPMSolverCfg, MPMWorkingSetCfg, NewtonCfg
    from isaaclab_visualizers.newton import NewtonGLBackendCfg

    import isaaclab.sim as sim_utils
    from isaaclab.visualizers import VisualizerCfg

    from isaaclab_tasks.utils.presets import MultiBackendVisualizerCfg

    class MovingBallSolver(MPMSolver):
        def build(self, model, mode, env_template):
            if not model.device.is_cuda:
                raise ValueError("This example requires a CUDA device.")
            solver = super().build(model, mode, env_template)
            self.motion_time = wp.zeros(1, dtype=float, device=model.device)
            self.motion_enabled = wp.zeros(1, dtype=int, device=model.device)
            self.motion_height = 1.0
            return solver

        def step(self, state_in, state_out, control, contacts, dt, *, observables=None):
            wp.launch(
                _move_ball,
                dim=1,
                inputs=[
                    state_in.body_q,
                    state_in.body_qd,
                    self.motion_time,
                    self.motion_enabled,
                    dt,
                    self.motion_height,
                ],
                device=self.native.model.device,
            )
            super().step(state_in, state_out, control, contacts, dt, observables=observables)

    return sim_utils.SimulationCfg(
        dt=1.0 / FPS,
        device=args_cli.device,
        visualizer_cfgs=VisualizerCfg(
            backend=MultiBackendVisualizerCfg(newton_gl=NewtonGLBackendCfg(particle_color=(0.76, 0.48, 0.18)))
        ),
        physics=NewtonCfg(
            num_substeps=SUBSTEPS,
            solver_cfg=MPMSolverCfg(
                class_type=MovingBallSolver,
                voxel_size=VOXEL_SIZE,
                grid_type="sparse",
                reuse_grid_topology=True,
                max_active_cell_count=32768,
                max_leaf_node_count=1024,
                max_iterations=25,
                tolerance=1.0e-10,
                velocity_basis="Q1",
                strain_basis="P0",
                collider_basis="S2",
                collider_velocity_mode="forward",
                sleeping=MPMSleepingCfg(stress_filter_time=0.1) if args_cli.sleeping else None,
                working_set=MPMWorkingSetCfg(particle_capacity=16384, page_size=32 if args_cli.paging else 0)
                if args_cli.sleeping
                else None,
            ),
        ),
    )


def create_scene_cfg():
    """Use the same declarative ground and MPM assets as the other examples."""
    from isaaclab_newton.assets import MPMObjectCfg
    from isaaclab_newton.sim.schemas import NewtonCollisionCfg
    from isaaclab_newton.sim.spawners.mpm import MPMGridCfg, MPMParticleMaterialCfg

    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sim.spawners.materials import UsdPhysicsRigidBodyMaterialCfg
    from isaaclab.utils import configclass

    @configclass
    class SleepingSceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(
            prim_path="/World/Ground",
            spawn=sim_utils.GroundPlaneCfg(
                physics_material=UsdPhysicsRigidBodyMaterialCfg(static_friction=0.7, dynamic_friction=0.7)
            ),
        )
        light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2500.0))
        sand = MPMObjectCfg(
            prim_path="{ENV_REGEX_NS}/Sand",
            spawn=MPMGridCfg(
                lower=(-1.0, -1.0, 0.0),
                upper=(1.0, 1.0, 0.08),
                voxel_size=VOXEL_SIZE,
                particles_per_cell=2.0,
                particle_placement="cell_center",
                jitter=0.0,
                material=MPMParticleMaterialCfg(density=1500.0, young_modulus=1.0e7, friction=0.7),
                visual_color=(0.76, 0.48, 0.18),
            ),
        )
        ball = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Ball",
            spawn=sim_utils.SphereCfg(
                radius=BALL_RADIUS,
                rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(rigid_body_enabled=True, kinematic_enabled=True),
                collision_props=[
                    sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True),
                    NewtonCollisionCfg(contact_margin=0.0),
                ],
                mass_props=sim_utils.MassCfg(mass=1.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.18, 0.62, 0.36)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
        )

    return SleepingSceneCfg(num_envs=1, env_spacing=0.0)


def run_simulator(sim, scene):
    """Settle with full buffers, classify activity, then capture the bounded layout."""
    from isaaclab_newton.physics import configure_mpm_sleeping, configure_mpm_working_set, defer_mpm_status_checks

    manager = sim.backend_manager
    backend = manager.backend
    cfg = sim.cfg.physics.solver_cfg
    dt = sim.get_physics_dt()
    for _ in range(math.ceil(args_cli.settle_time / dt)):
        sim.step(render=False)
        scene.update(dt)
    configure_mpm_sleeping(manager, cfg.sleeping)
    for _ in range(math.ceil(0.6 / dt)):
        sim.step(render=False)
        scene.update(dt)
    # This setup-only download places the sphere relative to the settled surface.
    surface_height = float(np.quantile(backend.state_0.particle_q.numpy()[:, 2], 0.99))
    backend.solver_ops.motion_height = surface_height + BALL_RADIUS - 0.02
    backend.invalidate_graph()  # The controller height is captured by value, including in the full reference.
    configure_mpm_working_set(manager, cfg.working_set)
    sim.step(render=False)  # Warm and capture the final layout with the controller disabled.
    scene.update(dt)
    backend.solver_ops.motion_enabled.fill_(1)
    print("[INFO]: Setup complete. Particle activity and sphere motion now replay on GPU.", flush=True)
    with defer_mpm_status_checks(manager):
        frames = 0
        while sim.is_running() and (args_cli.max_steps < 0 or frames < args_cli.max_steps):
            sim.step(render=False)
            scene.update(dt)
            if sim.is_rendering:
                sim.render()
            frames += 1
    if frames:
        np.testing.assert_allclose(backend.state_0.body_q.numpy()[0, 2], backend.solver_ops.motion_height, atol=1.0e-6)
    for values in (backend.state_0.particle_q, backend.state_0.particle_qd):
        if not np.isfinite(values.numpy()).all():
            raise RuntimeError("The rollout produced nonfinite particle state.")


def main():
    """Launch one settled bed with optional compact or paged evaluation buffers."""
    from isaaclab_tasks.utils import resolve_presets

    sim_cfg = resolve_presets(create_sim_cfg(), overrides={"visualizer_cfgs.backend": args_cli.visualizer})
    with launch_simulation(sim_cfg, args_cli):
        import isaaclab.sim as sim_utils
        from isaaclab.scene import InteractiveScene

        sim = sim_utils.SimulationContext(sim_cfg)
        sim.set_camera_view(eye=(3.0, -4.0, 2.5), target=(0.0, 0.0, 0.1))
        scene = InteractiveScene(create_scene_cfg())
        sim.reset()
        run_simulator(sim, scene)


if __name__ == "__main__":
    main()
