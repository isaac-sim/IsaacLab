from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.stage import add_reference_to_stage

# Optional: For Reinforcement Learning or environment setups
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
import haw_ur5_cfg

import torch


# Define the DirectRL environment configuration
@configclass
class MyRobotEnvCfg(DirectRLEnvCfg):
    # Simulation Configuration
    sim: SimulationCfg = SimulationCfg(dt=1 / 120.0, render_interval=2)

    # Scene Configuration
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1, env_spacing=4.0, replicate_physics=True
    )

    # Robot Configuration
    robot_usd_path: str = (
        "omniverse://localhost/MyAssets/haw_ur5_arm_rg6/haw_ur5_gr6.usd"  # Path to USD file
    )
    robot_prim_path: str = "/World/UR5"  # Where to spawn the robot in the scene


class MyRobotEnv(DirectRLEnv):
    cfg: MyRobotEnvCfg

    def __init__(self, cfg: MyRobotEnvCfg, **kwargs):
        super().__init__(cfg, **kwargs)
        self.robot = None

    def _setup_scene(self):
        # Load the robot USD file into the stage
        add_reference_to_stage(
            usd_path=self.cfg.robot_usd_path, prim_path=self.cfg.robot_prim_path
        )

        # Create the robot as an articulation
        self.robot = Articulation(prim_path=self.cfg.robot_prim_path)

        # Optionally, add ground plane or other objects
        ground_plane = DynamicCuboid(
            prim_path="/World/groundPlane",
            position=[0.0, 0.0, 0.0],
            size=[100.0, 100.0, 0.1],
            color=[0.5, 0.5, 0.5],
            mass=0.0,
            name="groundPlane",
        )

        # Add the robot to the scene
        self.scene.add(self.robot)
        self.scene.add(ground_plane)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # Handle actions here
        pass

    def _get_observations(self) -> dict:
        # Return observations as a dictionary
        return {}

    def _get_rewards(self) -> torch.Tensor:
        # Compute and return the reward
        return torch.tensor(0.0)

    def _get_dones(self) -> torch.Tensor:
        # Return done flags
        return torch.tensor(False)

    def _reset_idx(self, env_ids):
        # Reset the environment based on env_ids
        pass


if __name__ == "__main__":
    # Load your environment configuration
    cfg = MyRobotEnvCfg()

    # Initialize your custom environment (this will manage the simulation context)
    env = MyRobotEnv(cfg=cfg)

    # Reset the environment before starting
    env.reset()

    # Simulation loop
    for _ in range(1000):  # Run for a set number of steps
        # Get actions from your policy or set them manually for testing
        actions = torch.zeros((1,))  # Dummy actions

        # Step the simulation
        env.step(actions)

    # Finalize and close the simulation
    env.simulation_app.close()
