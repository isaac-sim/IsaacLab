import argparse
from isaaclab.app import AppLauncher

# CLI arguments
parser = argparse.ArgumentParser(description="Load X7 Duo dual-arm robot inside IsaacLab.")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Launch GUI
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


# --- Imports AFTER simulation app launched ---
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
import math
from isaaclab.utils import math as math_utils



# =====================================================================
#  1) CONFIGURE THE X7-DUO ROBOT (14 DOF)
# =====================================================================

X7_DUO_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="X7_Robot/assets/X7/x7_duo.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            fix_root_link=True,
        ),
    ),

    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={f"Joint{i}": 0.0 for i in range(1, 15)},
        pos=(0.0, 0.0, 1.8),
        # 关键：绕 Y 轴 -90° 旋转（单位：弧度）
        rot=tuple(
            math_utils.quat_from_euler_xyz(
                torch.tensor([0.0]),                 # roll  = 0
                torch.tensor([-math.pi / 2]),        # pitch = -90°
                torch.tensor([0.0]),                 # yaw   = 0
            )[0].tolist()
        ),
    ),

    actuators={
        # Left Arm Joint1~Joint7
        "A1": ImplicitActuatorCfg(
            joint_names_expr=["Joint1"],
            stiffness=3000.0,
            damping=80.0
        ),
        "A2": ImplicitActuatorCfg(
            joint_names_expr=["Joint2"],
            stiffness=3000.0,
            damping=80.0
        ),
        "A3": ImplicitActuatorCfg(
            joint_names_expr=["Joint3"],
            stiffness=3000.0,
            damping=80.0
        ),
        "A4": ImplicitActuatorCfg(
            joint_names_expr=["Joint4"],
            stiffness=3000.0,
            damping=80.0
        ),
        "A5": ImplicitActuatorCfg(
            joint_names_expr=["Joint5"],
            stiffness=3000.0,
            damping=80.0
        ),
        "A6": ImplicitActuatorCfg(
            joint_names_expr=["Joint6"],
            stiffness=3000.0,
            damping=80.0
        ),
        "A7": ImplicitActuatorCfg(
            joint_names_expr=["Joint7"],
            stiffness=3000.0,
            damping=80.0
        ),

        # Right Arm Joint8~Joint14
        "B1": ImplicitActuatorCfg(
            joint_names_expr=["Joint8"],
            stiffness=3000.0,
            damping=80.0
        ),
        "B2": ImplicitActuatorCfg(
            joint_names_expr=["Joint9"],
            stiffness=3000.0,
            damping=80.0
        ),
        "B3": ImplicitActuatorCfg(
            joint_names_expr=["Joint10"],
            stiffness=3000.0,
            damping=80.0
        ),
        "B4": ImplicitActuatorCfg(
            joint_names_expr=["Joint11"],
            stiffness=3000.0,
            damping=80.0
        ),
        "B5": ImplicitActuatorCfg(
            joint_names_expr=["Joint12"],
            stiffness=3000.0,
            damping=80.0
        ),
        "B6": ImplicitActuatorCfg(
            joint_names_expr=["Joint13"],
            stiffness=3000.0,
            damping=80.0
        ),
        "B7": ImplicitActuatorCfg(
            joint_names_expr=["Joint14"],
            stiffness=3000.0,
            damping=80.0
        ),
    },
)



# =====================================================================
#  2) DEFINE SCENE (WITH GROUND + ROBOT)
# =====================================================================

class X7DuoSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg()
    )

    robot = X7_DUO_CFG.replace(
        prim_path="{ENV_REGEX_NS}/X7_duo"
    )


# =====================================================================
#  3) SIMULATION LOOP
# =====================================================================

def run(sim, scene):
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    print("[INFO] Running simulator ...")

    while simulation_app.is_running():

        # Every 500 frames → reset robot
        if count % 500 == 0:
            print("[INFO] Reset robot state ...")

            # Reset root pose
            root = scene["robot"].data.default_root_state.clone()
            root[:, :3] += scene.env_origins  # place at env origin
            scene["robot"].write_root_pose_to_sim(root[:, :7])
            scene["robot"].write_root_velocity_to_sim(root[:, 7:])

            # Reset joints
            jp = scene["robot"].data.default_joint_pos.clone()
            jv = scene["robot"].data.default_joint_vel.clone()
            scene["robot"].write_joint_state_to_sim(jp, jv)

            scene.reset()

        # ----------------------------------------------------
        # ACTION: make both arms swing periodically
        # ----------------------------------------------------
        action = scene["robot"].data.default_joint_pos.clone()

        # Left Arm (Joint1~7)
        action[:, 0:7] = 0.4 * np.sin(2 * np.pi * 0.5 * sim_time)

        # Right Arm (Joint8~14)
        action[:, 7:14] = 0.4 * np.sin(2 * np.pi * 0.5 * sim_time)

        # Send command
        scene["robot"].set_joint_position_target(action)

        # Step
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

        count += 1
        sim_time += sim_dt


# =====================================================================
#  4) MAIN
# =====================================================================

def main():
    sim_cfg = sim_utils.SimulationCfg(device=args.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([4.0, 0.0, 2.5], [0, 0, 0.5])

    scene_cfg = X7DuoSceneCfg(args.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()

    print("[INFO] Scene ready.")
    run(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
