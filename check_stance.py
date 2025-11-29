import argparse
from isaaclab.app import AppLauncher

# 1. 配置启动参数
parser = argparse.ArgumentParser(description="Check robot initial stance.")
parser.add_argument("--task", type=str, default="Isaac-Velocity-MyBiped-v0", help="Task name.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 2. 启动 Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import gymnasium as gym
import isaaclab_tasks  # 注册你的任务
from isaaclab_tasks.utils import parse_env_cfg

def main():
    # 3. 加载并修改配置
    print(f"[INFO] Loading configuration for task: {args_cli.task}")
    
    # 注意：新版移除了 use_gpu 参数，如果有 device 参数就加上，没有就去掉
    # 这里我们只传 num_envs
    cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    
    # 4. 创建环境
    env = gym.make(args_cli.task, cfg=cfg)
    
    print("[INFO] Environment created. Resetting...")
    env.reset()

    print("[INFO] Running simulation with ZERO actions (Holding Default Pose)...")
    
    # 5. 循环运行
    while simulation_app.is_running():
        # ⚠️ 关键修改：通过 unwrapped 获取 device
        device = env.unwrapped.device
        
        # 发送全 0 动作 = 保持默认姿态 (Default Joint Pos)
        zero_actions = torch.zeros(env.action_space.shape, device=device)
        
        # 步进环境
        env.step(zero_actions)

    env.close()

if __name__ == "__main__":
    main()