# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Visualization script for finger collision isolation test with Newton physics.

This script visualizes a sphere colliding with an Allegro hand fingertip.
Use this to debug collision behavior and verify finger deflection.

Usage:
    cd /home/zhengyuz/Projects/isaaclab
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate newton_isaaclab

    # Run with visualization (headless=False)
    python scripts/demos/finger_collision_visualization.py --target_finger index

    # Run headless with debug output
    python scripts/demos/finger_collision_visualization.py --target_finger thumb --headless
"""

import argparse
import torch
import warp as wp

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.sim import build_simulation_context
from isaaclab.sim._impl.newton_manager_cfg import NewtonCfg
from isaaclab.sim._impl.solvers_cfg import MJWarpSolverCfg
from isaaclab.sim.simulation_cfg import SimulationCfg

# Import hand configuration
from isaaclab_assets.robots.allegro import ALLEGRO_HAND_CFG

##
# Configuration
##

# Finger tip positions relative to hand root in default orientation
ALLEGRO_FINGERTIP_OFFSETS = {
    "index": (-0.052, -0.252, 0.052),
    "middle": (-0.001, -0.252, 0.052),
    "ring": (0.054, -0.252, 0.052),
    "thumb": (-0.168, -0.039, 0.080),
}

# Joint names for each finger
ALLEGRO_FINGER_JOINTS = {
    "index": ["index_joint_0", "index_joint_1", "index_joint_2", "index_joint_3"],
    "middle": ["middle_joint_0", "middle_joint_1", "middle_joint_2", "middle_joint_3"],
    "ring": ["ring_joint_0", "ring_joint_1", "ring_joint_2", "ring_joint_3"],
    "thumb": ["thumb_joint_0", "thumb_joint_1", "thumb_joint_2", "thumb_joint_3"],
}

# Newton solver configuration
SOLVER_CFG = MJWarpSolverCfg(
    njmax=100,
    nconmax=100,
    ls_iterations=20,
    cone="elliptic",
    impratio=100,
    ls_parallel=True,
    integrator="implicit",
)

NEWTON_CFG = NewtonCfg(
    solver_cfg=SOLVER_CFG,
    num_substeps=1,
    debug_mode=False,
    use_cuda_graph=False,
)


def main():
    parser = argparse.ArgumentParser(description="Visualize finger collision isolation test")
    parser.add_argument("--target_finger", type=str, default="index", choices=["index", "middle", "ring", "thumb"])
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    parser.add_argument("--device", type=str, default="cuda:0", choices=["cuda:0", "cpu"])
    parser.add_argument("--sphere_type", type=str, default="primitive", choices=["primitive", "mesh"])
    args = parser.parse_args()

    target_finger = args.target_finger
    device = args.device
    sim_dt = 1.0 / 240.0
    drop_steps = 480  # 2 seconds

    # Hand position
    hand_pos = (0.0, 0.0, 0.5)

    # Zero gravity globally - ball will have initial downward velocity
    sim_cfg = SimulationCfg(
        dt=sim_dt,
        create_stage_in_memory=False,
        newton_cfg=NEWTON_CFG,
        device=device,
        gravity=(0.0, 0.0, 0.0),
    )

    print(f"\n{'='*60}")
    print(f"Finger Collision Visualization")
    print(f"{'='*60}")
    print(f"Target finger: {target_finger}")
    print(f"Device: {device}")
    print(f"Sphere type: {args.sphere_type}")
    print(f"Headless: {args.headless}")
    print(f"Gravity: (0, 0, 0) - using initial velocity instead")
    print(f"{'='*60}\n")

    with build_simulation_context(sim_cfg=sim_cfg, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None

        # Create hand configuration
        hand_cfg = ALLEGRO_HAND_CFG.copy()
        hand_cfg.prim_path = "/World/Hand"
        hand_cfg.init_state.pos = hand_pos

        # Create ground plane
        ground_cfg = sim_utils.GroundPlaneCfg()
        ground_cfg.func("/World/ground", ground_cfg)

        # Create the hand
        hand = Articulation(hand_cfg)

        # Get fingertip offset for target finger
        fingertip_offset = ALLEGRO_FINGERTIP_OFFSETS[target_finger]
        print(f"Fingertip offset for '{target_finger}': {fingertip_offset}")

        # Position sphere above fingertip
        drop_height = 0.10  # 10cm above
        sphere_pos = (
            hand_pos[0] + fingertip_offset[0],
            hand_pos[1] + fingertip_offset[1],
            hand_pos[2] + fingertip_offset[2] + drop_height,
        )
        print(f"Sphere initial position: {sphere_pos}")

        # Create sphere
        if args.sphere_type == "primitive":
            sphere_spawn = sim_utils.SphereCfg(
                radius=0.035,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    linear_damping=0.0,
                    angular_damping=0.0,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            )
        else:
            sphere_spawn = sim_utils.MeshSphereCfg(
                radius=0.035,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    linear_damping=0.0,
                    angular_damping=0.0,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            )

        sphere_cfg = RigidObjectCfg(
            prim_path="/World/DropSphere",
            spawn=sphere_spawn,
            init_state=RigidObjectCfg.InitialStateCfg(pos=sphere_pos),
        )
        drop_sphere = RigidObject(sphere_cfg)

        # Reset simulation
        sim.reset()
        hand.reset()
        drop_sphere.reset()

        # Let hand settle
        print("\nSettling hand...")
        settle_steps = 30
        for _ in range(settle_steps):
            hand.write_data_to_sim()
            # Always use render=False for Newton physics (no omni.kit dependency)
            sim.step(render=False)
            hand.update(sim_dt)

        # Reset sphere and give initial velocity
        drop_sphere.reset()
        initial_velocity = torch.tensor([[0.0, 0.0, -1.5, 0.0, 0.0, 0.0]], device=device)
        drop_sphere.write_root_velocity_to_sim(initial_velocity)
        print(f"Sphere initial velocity: {initial_velocity[0, :3].tolist()} m/s")

        # Record initial joint positions
        initial_joint_pos = wp.to_torch(hand.data.joint_pos).clone()
        joint_names = hand.data.joint_names

        print(f"\nJoint names: {joint_names}")
        print(f"\nInitial joint positions: {initial_joint_pos[0].tolist()}")

        # Track deflection
        peak_deflection = {finger: 0.0 for finger in ["index", "middle", "ring", "thumb"]}

        print(f"\nRunning simulation for {drop_steps} steps ({drop_steps * sim_dt:.2f}s)...")
        print("-" * 60)

        # Run simulation
        for step in range(drop_steps):
            hand.write_data_to_sim()
            drop_sphere.write_data_to_sim()
            # Always use render=False for Newton physics (no omni.kit dependency)
            sim.step(render=False)
            hand.update(sim_dt)
            drop_sphere.update(sim_dt)

            # Track deflection
            current_joint_pos = wp.to_torch(hand.data.joint_pos)[0]
            for finger_name in ["index", "middle", "ring", "thumb"]:
                finger_deflection = 0.0
                for joint_name in ALLEGRO_FINGER_JOINTS[finger_name]:
                    if joint_name in joint_names:
                        idx = joint_names.index(joint_name)
                        finger_deflection += abs(current_joint_pos[idx].item() - initial_joint_pos[0, idx].item())
                peak_deflection[finger_name] = max(peak_deflection[finger_name], finger_deflection)

            # Print progress every 100 steps
            if step % 100 == 0 or step == drop_steps - 1:
                sphere_pos = wp.to_torch(drop_sphere.data.root_pos_w)[0]
                sphere_vel = wp.to_torch(drop_sphere.data.root_lin_vel_w)[0]
                print(
                    f"Step {step:4d}: "
                    f"sphere_z={sphere_pos[2]:.4f}, vel_z={sphere_vel[2]:.4f}, "
                    f"deflections: {', '.join([f'{k}={v:.4f}' for k, v in peak_deflection.items()])}"
                )

        # Print results
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        target_peak = peak_deflection[target_finger]
        print(f"Target finger: {target_finger}")
        print(f"Target peak deflection: {target_peak:.6f}")
        print("\nAll finger deflections:")
        for finger_name, deflection in peak_deflection.items():
            marker = " <-- TARGET" if finger_name == target_finger else ""
            print(f"  {finger_name:8s}: {deflection:.6f}{marker}")

        # Check test conditions
        print("\nTest conditions:")
        if target_peak > 0.01:
            print(f"  [PASS] Target finger deflected > 0.01 ({target_peak:.6f})")
        else:
            print(f"  [FAIL] Target finger deflection too small ({target_peak:.6f} <= 0.01)")

        for finger_name in ["index", "middle", "ring", "thumb"]:
            if finger_name != target_finger:
                if target_peak >= peak_deflection[finger_name]:
                    print(f"  [PASS] {target_finger} >= {finger_name} ({target_peak:.4f} >= {peak_deflection[finger_name]:.4f})")
                else:
                    print(f"  [FAIL] {target_finger} < {finger_name} ({target_peak:.4f} < {peak_deflection[finger_name]:.4f})")

        print("=" * 60)


if __name__ == "__main__":
    main()
