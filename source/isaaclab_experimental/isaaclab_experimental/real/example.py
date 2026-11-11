# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run a stock manager environment against fake joint telemetry and report unsupported surfaces.

Run ``uv run python -m isaaclab_experimental.real.example --steps 30`` from a source checkout.
This example does not connect to hardware or use IsaacDeploy's C++ inference runtime.
"""

from __future__ import annotations

import argparse
import json

import torch

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg, mdp
from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from .backend_cfg import RealCfg
from .session import FakeSession


@configclass
class JointSceneCfg(InteractiveSceneCfg):
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=None,
        actuators={},
        init_state=ArticulationCfg.InitialStateCfg(joint_pos={"shoulder": 0.1, "elbow": -0.2}),
    )


@configclass
class PolicyCfg(ObservationGroupCfg):
    position = ObservationTermCfg(func=mdp.joint_pos_rel)
    velocity = ObservationTermCfg(func=mdp.joint_vel_rel)


@configclass
class ActionsCfg:
    joints = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=["elbow", "shoulder"], preserve_order=True, scale=0.5
    )


@configclass
class ObservationsCfg:
    policy = PolicyCfg()


def make_env_cfg() -> ManagerBasedEnvCfg:
    """Configure stock actions/observations and explicitly remove simulation-only reset events."""
    return ManagerBasedEnvCfg(
        sim=SimulationCfg(
            physics=RealCfg(session_factory=FakeSession),
            device="cpu",
            dt=0.02,
            render_interval=2,
            create_stage_in_memory=True,
            visualizer_cfgs=[],
        ),
        scene=JointSceneCfg(num_envs=1, env_spacing=0.0),
        decimation=2,
        seed=0,
        actions=ActionsCfg(),
        observations=ObservationsCfg(),
        events={},
        ui_window_class_type=None,
    )


def probe_unsupported(env: ManagerBasedEnv) -> dict[str, str]:
    """Exercise real caller boundaries and report the concrete missing capabilities."""
    robot = env.scene["robot"]
    probes = {
        "scene_state_recording": lambda: env.scene.get_state(),
        "floating_base_velocity": lambda: mdp.base_lin_vel(env),
        "teleport_joint_state": lambda: robot.write_joint_position_to_sim(torch.zeros(1, 2)),
        "velocity_commands": lambda: robot.set_joint_velocity_target_index(torch.zeros(1, 2)),
    }
    results = {}
    for name, operation in probes.items():
        try:
            operation()
        except (AttributeError, NotImplementedError) as error:
            results[name] = f"{type(error).__name__}: {error}"
        else:
            results[name] = "supported"
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=30)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    env = ManagerBasedEnv(make_env_cfg())
    try:
        initial, _ = env.reset()
        for _ in range(args.steps):
            observations, _ = env.step(torch.tensor([[0.6, -0.8]]))
        robot = env.scene["robot"]
        print(
            json.dumps(
                {
                    "backend": env.sim.backend_manager.backend_name,
                    "runtime": "FakeSession (logical time; no hardware)",
                    "joint_names": robot.joint_names,
                    "initial_observations": initial["policy"].tolist(),
                    "final_observations": observations["policy"].tolist(),
                    "measured_position": robot.data.joint_pos.torch.tolist(),
                    "position_target": robot.data.joint_pos_target.torch.tolist(),
                    "session_time": robot.timestamp,
                    "unsupported": probe_unsupported(env),
                },
                indent=2,
            )
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
