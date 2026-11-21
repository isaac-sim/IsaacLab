# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Roll an OvPhysX rigid ball through Newton implicit-MPM sand.

The ball runs in a standalone :class:`~isaaclab_ov.physics.OvPhysxBackend` with its
own small USD stage. Every step its pose drives a kinematic sphere in the Isaac Lab
scene, which Newton's MPM solver treats as a moving collider. The coupling is
one-way: the sand never pushes back on the ball. A new ball rolls down a
different lane every two seconds, and the sand resets once every lane is plowed.

.. code-block:: bash

    uvx isaaclab example mpm-ovphysx-one-way-coupling
"""

from __future__ import annotations

import argparse

from isaaclab_visualizers.newton import NewtonGLBackendCfg

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.visualizers import VisualizerCfg

from isaaclab_tasks.utils import resolve_presets
from isaaclab_tasks.utils.presets import MultiBackendVisualizerCfg

parser = argparse.ArgumentParser(description="OvPhysX rigid ball to Newton MPM sand one-way coupling example.")
parser.add_argument("--max_steps", type=int, default=-1, help="Stop after this many steps; negative runs forever.")
parser.add_argument("--voxel_size", type=float, default=0.025, help="MPM grid voxel size [m].")
add_launcher_args(parser)
parser.set_defaults(visualizer=["newton"])
args_cli = parser.parse_args()


DT = 1.0 / 120.0
GRAVITY = 9.81
PARTICLES_PER_VOXEL_AXIS = 2.0
PARTICLE_COLOR = (0.76, 0.58, 0.32)
SAND_LOWER = (-0.45, -0.45, 0.0)
SAND_UPPER = (0.95, 0.45, 0.10)
BALL_RADIUS = 0.12
BALL_START = (-0.75, 0.0, BALL_RADIUS)
BALL_SPEED = 3.0
LANES = (0.0, -0.25, 0.25)
"""Lateral offsets [m] cycled by successive throws."""
STEPS_PER_THROW = 240
CAMERA_EYE = (1.3, -1.3, 0.7)
CAMERA_TARGET = (0.3, 0.0, 0.0)


def ball_stage(lane: float, device: str) -> str:
    """Return the OvPhysX stage for one throw: a ground slab and a launched ball."""
    gpu = device.startswith("cuda")
    x, y, z = BALL_START
    return f"""#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1
    upAxis = "Z"
)
def Xform "World" {{
    def PhysicsScene "physicsScene" (prepend apiSchemas = ["PhysxSceneAPI"]) {{
        vector3f physics:gravityDirection = (0, 0, -1)
        float physics:gravityMagnitude = {GRAVITY}
        bool physxScene:enableGPUDynamics = {str(gpu).lower()}
        token physxScene:broadphaseType = "{"GPU" if gpu else "MBP"}"
    }}
    def Cube "Ground" (prepend apiSchemas = ["PhysicsCollisionAPI"]) {{
        double size = 1
        double3 xformOp:translate = (0, 0, -0.5)
        float3 xformOp:scale = (20, 20, 1)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
    }}
    def Sphere "Ball" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsCollisionAPI", "PhysicsMassAPI"]) {{
        double radius = {BALL_RADIUS}
        double3 xformOp:translate = ({x}, {y + lane}, {z})
        uniform token[] xformOpOrder = ["xformOp:translate"]
        float physics:mass = 5
        vector3f physics:velocity = ({BALL_SPEED}, 0, 0)
    }}
}}
"""


def create_sim_cfg():
    """Create the Newton MPM simulation that owns the sand."""
    from isaaclab_newton.physics import MPMSolverCfg, NewtonCfg

    import isaaclab.sim as sim_utils

    return sim_utils.SimulationCfg(
        dt=DT,
        device=args_cli.device,
        gravity=(0.0, 0.0, -GRAVITY),
        visualizer_cfgs=VisualizerCfg(
            backend=MultiBackendVisualizerCfg(newton_gl=NewtonGLBackendCfg(particle_color=PARTICLE_COLOR)),
        ),
        physics=NewtonCfg(
            solver_cfg=MPMSolverCfg(
                voxel_size=args_cli.voxel_size,
                # A fixed grid lets the MPM step run as a CUDA graph.
                grid_type="fixed",
                grid_padding=48,
                max_active_cell_count=1 << 17,
                max_iterations=50,
                transfer_scheme="pic",
                # The kinematic ball's velocity is never written, so infer it from successive poses.
                collider_velocity_mode="backward",
            )
        ),
    )


def create_scene_cfg():
    """Create the sand bed and the kinematic sphere that mirrors the OvPhysX ball."""
    from isaaclab_newton.assets import MPMObjectCfg
    from isaaclab_newton.sim.spawners.mpm import MPMGridCfg

    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sim.spawners.materials import UsdPhysicsRigidBodyMaterialCfg
    from isaaclab.utils import configclass

    @configclass
    class SandSceneCfg(InteractiveSceneCfg):
        """Scene containing MPM sand and a kinematic ball collider."""

        # A thick, wide floor: a thin plane lets pressed sand tunnel, and the ball flings a few grains far.
        floor = AssetBaseCfg(
            prim_path="/World/Floor",
            spawn=sim_utils.CuboidCfg(
                size=(40.0, 40.0, 0.2),
                collision_props=sim_utils.UsdPhysicsCollisionCfg(),
                physics_material=UsdPhysicsRigidBodyMaterialCfg(static_friction=0.6, dynamic_friction=0.6),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.12, 0.14, 0.18)),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.1)),
        )
        dome_light = AssetBaseCfg(
            prim_path="/World/DomeLight",
            spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.78, 0.78, 0.78)),
        )
        ball = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Ball",
            spawn=sim_utils.SphereCfg(
                radius=BALL_RADIUS,
                rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True),
                collision_props=sim_utils.UsdPhysicsCollisionCfg(),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=BALL_START),
        )
        sand = MPMObjectCfg(
            prim_path="{ENV_REGEX_NS}/Sand",
            spawn=MPMGridCfg(
                lower=SAND_LOWER,
                upper=SAND_UPPER,
                voxel_size=args_cli.voxel_size,
                particles_per_cell=PARTICLES_PER_VOXEL_AXIS,
                particle_placement="cell_center",
                jitter=args_cli.voxel_size / PARTICLES_PER_VOXEL_AXIS,
                visual_color=PARTICLE_COLOR,
            ),
        )

    return SandSceneCfg(num_envs=1, env_spacing=0.0)


def run_simulator(sim, scene, rigid) -> None:
    """Step OvPhysX, copy the ball pose into Newton, then step the sand."""
    ball, sand = scene["ball"], scene["sand"]
    step = 0
    while sim.is_running() and (args_cli.max_steps < 0 or step < args_cli.max_steps):
        if step % STEPS_PER_THROW == 0:
            throw = step // STEPS_PER_THROW
            if throw > 0 and throw % len(LANES) == 0:
                # Smooth the bed again once every lane has been plowed.
                sand.reset()
            rigid.load(ball_stage(LANES[throw % len(LANES)], sim.device))
            rigid.warmup()
        rigid.step(DT)
        # Both sides store poses as wp.transformf [m, xyzw] on the simulation device.
        ball.write_root_pose_to_sim_mask(root_pose=rigid.scene_data.transforms.transforms)
        sim.step(render=False)
        scene.update(DT)
        if sim.is_rendering:
            sim.render()
        step += 1


def main() -> None:
    """Launch the OvPhysX-to-MPM one-way coupling example."""
    sim_cfg = resolve_presets(create_sim_cfg(), overrides={"visualizer_cfgs.backend": args_cli.visualizer})
    with launch_simulation(sim_cfg, args_cli):
        from isaaclab_ov.physics import OvPhysxBackend, OvPhysxBackendCfg

        import isaaclab.sim as sim_utils
        from isaaclab.scene import InteractiveScene

        sim = sim_utils.SimulationContext(sim_cfg)
        sim.set_camera_view(eye=CAMERA_EYE, target=CAMERA_TARGET)
        scene = InteractiveScene(create_scene_cfg())
        sim.reset()
        sand = scene["sand"]
        particle_count = sand.num_instances * sand.particles_per_object
        print(f"[INFO]: OvPhysX to MPM example ready. Spawned {particle_count} particles.", flush=True)
        rigid = OvPhysxBackend(OvPhysxBackendCfg(device=sim.device))
        try:
            run_simulator(sim, scene, rigid)
        finally:
            rigid.close()


if __name__ == "__main__":
    main()
