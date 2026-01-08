import argparse

from isaaclab.app import AppLauncher
import torch

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
from control.controller import simple_swing_control, pi_control
from openpi_client import websocket_client_policy
from control.robot_reset import reset_robot

# ========= WebSocket client（必须全局初始化一次）=========
from openpi_client import websocket_client_policy



def run(sim: sim_utils.SimulationContext, scene: InteractiveScene, client: websocket_client_policy.WebsocketClientPolicy):
    robot=scene["robot"]
    
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    action_chunk = None

    print("[INFO] Running simulator ...")

    robot=scene["robot"]

    while simulation_app.is_running():
        # Reset
        if count > 0 and count % 500 == 0:
            reset_robot(scene)  # 根姿态 + 关节位姿 + scene.reset()
            scene.write_data_to_sim()
            # 给一小步让状态落地
            sim.step()
            scene.update(sim_dt)
            # 重新计时，避免首帧就再触发
            count = 0
            sim_time = 0.0
            # 跳过本轮控制，避免立刻被新动作覆盖
            continue

        # 控制器：双臂简单双摆
        action_chunk = pi_control(scene, sim_time, client)
        for i in range(10):
            a = action_chunk[i]    # 14-dim
            #print(f"Action step {i}: {a}")

            # -------------- ALOHA 动作结构 --------------
            left_arm   = a[0:6]
            #left_grip  = a[6]
            right_arm  = a[7:13]
            #right_grip = a[13]
            # ---------------------------------------------

            # 4）当前机器人状态
            cur_q = scene["robot"].data.joint_pos[0].clone()  
            #print(f" Current qpos: {cur_q}")
            
            # 5）更新目标动作：这里用「增量控制」更稳定
            cur_q[0:6]  += torch.tensor(left_arm,  device=cur_q.device)
            cur_q[6:12] += torch.tensor(right_arm, device=cur_q.device)
           

            # 6）发给 A10 双臂
            scene["robot"].set_joint_position_target(cur_q.unsqueeze(0))

            # 7）步进1次仿真
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_dt)
            #sim.render()
            count += 1
            sim_time += sim_dt


def main():
    # 初始化仿真
    sim_cfg = sim_utils.SimulationCfg(device=args.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([4.0, 0.0, 2.5], [0.0, 0.0, 0.5]) # type: ignore
    scene_cfg = A10SceneCfg(num_envs=args.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    print("[INFO]: Setup complete...")

    client = websocket_client_policy.WebsocketClientPolicy(host="localhost", port=8000)
 
    run(sim, scene, client)


if __name__ == "__main__":
    main()
    simulation_app.close()