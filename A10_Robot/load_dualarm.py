import argparse

from isaaclab.app import AppLauncher

# ========= CLI =========
parser = argparse.ArgumentParser(description="Load A10 dual-arm robot inside IsaacLab.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# ========= 启动 GUI / SimulationApp =========
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ========= 在 app 启动后再导入 =========
from isaaclab.scene import InteractiveScene
import isaaclab.sim as sim_utils
from scene.a10_scene_cfg import A10SceneCfg
from control.controller import simple_swing_control

# Ensure URDF importer is enabled (required for UrdfFileCfg)
try:
    from omni.kit.app import get_app  # type: ignore
    ext_mgr = get_app().get_extension_manager()
    if not ext_mgr.is_extension_enabled("omni.importer.urdf"):
        ext_mgr.set_extension_enabled_immediate("omni.importer.urdf", True)
except Exception:
    pass


def run(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    robot = scene["robot"]
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    print("[INFO] Running simulator ...")

    while simulation_app.is_running():
        # Reset every 500 steps
        if count % 500 == 0:
            count = 0
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])

        # 简单双摆控制，验证关节驱动
        simple_swing_control(scene, sim_time)

        # step sim
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        count += 1
        sim_time += sim_dt


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([4.0, 0.0, 2.5], [0.0, 0.0, 0.5])  # type: ignore
    scene_cfg = A10SceneCfg(num_envs=args.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    print("[INFO]: A10 setup complete...")

    run(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
