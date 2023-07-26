import torch
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from .velocity.locomotion_env import LocomotionEnv


def terrain_levels_vel(env: "LocomotionEnv", env_ids: Sequence[int]) -> torch.Tensor:
    """
    If the robot walked more than half the terrain length if moves to a harder level.
    Else if it walked less than half of the distance required by the commanded velocity, it goes to a simpler level
    """
    distance = torch.norm(env.robot.data.root_pos_w[env_ids, :2] - env.terrain_importer.env_origins[env_ids, :2], dim=1)
    # robots that walked far enough progress to harder terrains
    move_up = distance > env.cfg.terrain.terrain_generator.size[0] / 2
    # robots that walked less than half of their required distance go to simpler terrains
    move_down = (
        distance < torch.norm(env._command_manager.command[env_ids, :2], dim=1) * env.max_episode_length * env.dt * 0.5
    )
    move_down *= ~move_up
    # update terrain levels
    env.terrain_importer.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(env.terrain_importer.terrain_levels.float())
