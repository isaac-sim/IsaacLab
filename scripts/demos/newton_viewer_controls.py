# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Exercise Newton viewer dragging, pause, and single-step controls.

The scene uses only the Newton MJWarp rigid-body solver and contains three
dynamic cubes. Right-click and drag any cube to apply a force, press ``Space``
to pause or resume physics, and press ``.`` to advance one physics step.

.. code-block:: bash

    uv run python scripts/demos/newton_viewer_controls.py
"""

import argparse

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description="Newton viewer controls with three draggable MJWarp cubes.")
parser.add_argument("--max_steps", type=int, default=-1, help="Stop after this many steps; negative runs forever.")
add_launcher_args(parser)
parser.set_defaults(visualizer=["newton"])
args_cli = parser.parse_args()

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils.configclass import configclass


def cube_cfg(name: str, position: tuple[float, float, float]) -> RigidObjectCfg:
    """Create one draggable cube configuration."""
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 0.5, 0.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=position),
    )


@configclass
class ViewerControlsSceneCfg(InteractiveSceneCfg):
    """Ground plane and three dynamic MJWarp cubes."""

    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.GroundPlaneCfg(size=(6.0, 6.0), color=(0.25, 0.25, 0.25)),
    )
    left_cube = cube_cfg("LeftCube", (-0.75, 0.0, 0.5))
    center_cube = cube_cfg("CenterCube", (0.0, 0.0, 0.5))
    right_cube = cube_cfg("RightCube", (0.75, 0.0, 0.5))


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene) -> None:
    """Run until the viewer closes or the optional step limit is reached."""
    sim_dt = sim.get_physics_dt()
    step_count = 0
    while sim.is_headless_or_exist_active_visualizer() and (args_cli.max_steps < 0 or step_count < args_cli.max_steps):
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        step_count += 1


def main() -> None:
    """Launch the MJWarp viewer-controls demo."""
    physics_cfg = NewtonCfg(solver_cfg=MJWarpSolverCfg())
    with launch_simulation(cfg=physics_cfg, launcher_args=args_cli) as resolved_physics_cfg:
        sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 100.0, device=args_cli.device, physics=resolved_physics_cfg)
        sim = sim_utils.SimulationContext(sim_cfg)
        sim.set_camera_view(eye=(3.0, -4.0, 2.5), target=(0.0, 0.0, 0.5))
        scene = InteractiveScene(ViewerControlsSceneCfg(num_envs=1, env_spacing=1.0))
        sim.reset()
        print("[INFO]: Right-click and drag any cube. Space pauses; '.' advances one step.", flush=True)
        run_simulator(sim, scene)


if __name__ == "__main__":
    main()
