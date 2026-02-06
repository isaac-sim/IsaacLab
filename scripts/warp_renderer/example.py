import contextlib
import argparse
import time

from isaaclab.app import AppLauncher

@contextlib.contextmanager
def measure_time(title: str, iterations: int = 1):
    start_time = time.perf_counter_ns()
    yield
    end_time = time.perf_counter_ns()
    print(f"[BENCHMARK] {title}: {(end_time - start_time) / iterations / 1e6:.2f} ms")


@contextlib.contextmanager
def launch_app(args):
    with measure_time("App start time"):
        app_launcher = AppLauncher(args)
    yield app_launcher.app
    if app_launcher.app:
        app_launcher.app.close()


parser = argparse.ArgumentParser(description="Benchmark Warp Raytrace")
parser.add_argument("--num_envs", type=int, default=32, help="Number of robots to simulate")
parser.add_argument("--save_images", action="store_true", help="Save Sensor Images")
parser.add_argument("--steps", type=int, default=2000, help="Number of steps to simulate")

AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()

app_launcher = AppLauncher(args)

with measure_time("Imports time"):
    import torch

    import isaaclab.sim as isaaclab_sim
    from isaaclab.sensors import TiledCamera, TiledCameraCfg
    from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.utils import configclass
    from isaaclab_assets import ANYMAL_D_CFG
    from isaaclab.renderers import NewtonWarpRenderer as NewtonWarpRenderer


@configclass
class SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=isaaclab_sim.GroundPlaneCfg())
    dome_light = AssetBaseCfg(prim_path="/World/Light", spawn=isaaclab_sim.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)))
    robot: ArticulationCfg = ANYMAL_D_CFG
    robot.prim_path = "{ENV_REGEX_NS}/Robot"


def run_simulator(sim: isaaclab_sim.SimulationContext, scene: InteractiveScene, num_steps: int, save_images: bool):
    robot: Articulation = scene["robot"]
    for step in range(num_steps):
        if step % 500 == 0:
            # reset the scene entities root state we offset the root state by the origin since the states are 
            # written in simulation world frame if this is not done, then the robots will be spawned at 
            # the (0, 0, 0) of the simulation world

            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])
            # set joint positions with some noise
            joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
            joint_pos += torch.rand_like(joint_pos) * 0.1
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            # clear internal buffers
            scene.reset()

        # Apply random action
        efforts = torch.randn_like(robot.data.joint_pos) * 5.0
        robot.set_joint_effort_target(efforts)

        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())

        if save_images:
            if isinstance(scene.sensors["tiled_camera"], TiledCamera):
                if isinstance(scene.sensors["tiled_camera"].renderer, NewtonWarpRenderer):
                    scene.sensors["tiled_camera"].renderer.camera_manager.save_images(f"__warp_renderer/%s_rgb.{step:04d}.png")
        scene.sensors["tiled_camera"].data


def main():
    with measure_time("Simulation Context creation time"):
        sim_cfg = isaaclab_sim.SimulationCfg(device="cuda:0")
        sim = isaaclab_sim.SimulationContext(sim_cfg)

    sim.set_camera_view((2.5, 0.0, 4.0), (0.0, 0.0, 2.0))

    with measure_time("Scene creation time"):
        scene_cfg = SceneCfg(num_envs=args.num_envs, env_spacing=2.0)
        scene = InteractiveScene(scene_cfg)

        tiled_camera_cfg: TiledCameraCfg = TiledCameraCfg(
            prim_path="/World/envs/env_.*/Camera",
            offset=TiledCameraCfg.OffsetCfg(pos=(-3.0, 0.0, 1.0), rot=(0.0, 0.0, 0.0, 1.0), convention="world"),
            data_types=["rgb"],
            spawn=isaaclab_sim.PinholeCameraCfg(focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)),
            width=400,
            height=300,
            renderer="newton"
        )
        scene.sensors["tiled_camera"] = TiledCamera(tiled_camera_cfg, scene)

        # stage = isaaclab_sim.get_current_stage()
        # stage.Export("/home/dhasenbring/development/isaac/IsaacLab/stage.usda")

    with measure_time("Sim start time"):
        sim.reset()

    with measure_time("Average sim step time", iterations=args.steps):
        run_simulator(sim, scene, args.steps, args.save_images)


if __name__ == "__main__":
    main()
    app_launcher.app.close()