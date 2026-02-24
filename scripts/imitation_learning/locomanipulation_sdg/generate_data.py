# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to replay demonstrations with Isaac Lab environments."""

"""Launch Isaac Sim Simulator first."""


import argparse
import os

from isaaclab.app import AppLauncher

# Launch Isaac Lab
parser = argparse.ArgumentParser(description="Locomanipulation SDG")
parser.add_argument("--task", type=str, help="The Isaac Lab locomanipulation SDG task to load for data generation.")
parser.add_argument("--dataset", type=str, help="The static manipulation dataset recorded via teleoperation.")
parser.add_argument("--output_file", type=str, help="The file name for the generated output dataset.")
parser.add_argument(
    "--lift_step",
    type=int,
    help=(
        "The step index in the input recording where the robot is ready to lift the object.  Aka, where the grasp is"
        " finished."
    ),
)
parser.add_argument(
    "--navigate_step",
    type=int,
    help=(
        "The step index in the input recording where the robot is ready to navigate.  Aka, where it has finished"
        " lifting the object"
    ),
)
parser.add_argument("--demo", type=str, default=None, help="The demo in the input dataset to use.")
parser.add_argument("--num_runs", type=int, default=1, help="The number of trajectories to generate.")
parser.add_argument(
    "--draw_visualization", type=bool, default=False, help="Draw the occupancy map and path planning visualization."
)
parser.add_argument(
    "--angular_gain",
    type=float,
    default=2.0,
    help=(
        "The angular gain to use for determining an angular control velocity when driving the robot during navigation."
    ),
)
parser.add_argument(
    "--linear_gain",
    type=float,
    default=1.0,
    help="The linear gain to use for determining the linear control velocity when driving the robot during navigation.",
)
parser.add_argument(
    "--linear_max", type=float, default=1.0, help="The maximum linear control velocity allowable during navigation."
)
parser.add_argument(
    "--distance_threshold",
    type=float,
    default=0.2,
    help="The distance threshold in meters to perform state transitions between navigation and manipulation tasks.",
)
parser.add_argument(
    "--following_offset",
    type=float,
    default=0.6,
    help=(
        "The target point offset distance used for local path following during navigation.  A larger value will result"
        " in smoother trajectories, but may cut path corners."
    ),
)
parser.add_argument(
    "--angle_threshold",
    type=float,
    default=0.2,
    help=(
        "The angle threshold in radians to determine when the robot can move forward or transition between navigation"
        " and manipulation tasks."
    ),
)
parser.add_argument(
    "--approach_distance",
    type=float,
    default=0.5,
    help="An offset distance added to the destination to allow a buffer zone for reliably approaching the goal.",
)
parser.add_argument(
    "--randomize_placement",
    type=bool,
    default=True,
    help="Whether or not to randomize the placement of fixtures in the scene upon environment initialization.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import enum
import random

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler

import isaaclab_mimic.locomanipulation_sdg.envs  # noqa: F401
from isaaclab_mimic.locomanipulation_sdg.data_classes import LocomanipulationSDGOutputData
from isaaclab_mimic.locomanipulation_sdg.envs.locomanipulation_sdg_env import LocomanipulationSDGEnv
from isaaclab_mimic.locomanipulation_sdg.occupancy_map_utils import (
    OccupancyMap,
    merge_occupancy_maps,
    occupancy_map_add_to_stage,
)
from isaaclab_mimic.locomanipulation_sdg.path_utils import ParameterizedPath, plan_path
from isaaclab_mimic.locomanipulation_sdg.scene_utils import RelativePose, place_randomly
from isaaclab_mimic.locomanipulation_sdg.transform_utils import transform_inv, transform_mul, transform_relative_pose

from isaaclab_tasks.utils import parse_env_cfg


class LocomanipulationSDGDataGenerationState(enum.IntEnum):
    """States for the locomanipulation SDG data generation state machine."""

    GRASP_OBJECT = 0
    """Robot grasps object at start position"""

    LIFT_OBJECT = 1
    """Robot lifts object while stationary"""

    NAVIGATE = 2
    """Robot navigates to approach position with object"""

    APPROACH = 3
    """Robot approaches final goal position"""

    DROP_OFF_OBJECT = 4
    """Robot places object at end position"""

    DONE = 5
    """Task completed"""


@configclass
class LocomanipulationSDGControlConfig:
    """Configuration for navigation control parameters."""

    angular_gain: float = 2.0
    """Proportional gain for angular velocity control"""

    linear_gain: float = 1.0
    """Proportional gain for linear velocity control"""

    linear_max: float = 1.0
    """Maximum allowed linear velocity (m/s)"""

    distance_threshold: float = 0.1
    """Distance threshold for state transitions (m)"""

    following_offset: float = 0.6
    """Look-ahead distance for path following (m)"""

    angle_threshold: float = 0.2
    """Angular threshold for orientation control (rad)"""

    approach_distance: float = 1.0
    """Buffer distance from final goal (m)"""


def compute_navigation_velocity(
    current_pose: torch.Tensor, target_xy: torch.Tensor, config: LocomanipulationSDGControlConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute linear and angular velocities for navigation control.

    Args:
        current_pose: Current robot pose [x, y, yaw]
        target_xy: Target position [x, y]
        config: Navigation control configuration

    Returns:
        Tuple of (linear_velocity, angular_velocity)
    """
    current_xy = current_pose[:2]
    current_yaw = current_pose[2]

    # Compute position and orientation errors
    delta_xy = target_xy - current_xy
    delta_distance = torch.sqrt(torch.sum(delta_xy**2))

    target_yaw = torch.arctan2(delta_xy[1], delta_xy[0])
    delta_yaw = target_yaw - current_yaw
    # Normalize angle to [-π, π]
    delta_yaw = (delta_yaw + torch.pi) % (2 * torch.pi) - torch.pi

    # Compute control commands
    angular_velocity = config.angular_gain * delta_yaw
    linear_velocity = torch.clip(config.linear_gain * delta_distance, 0.0, config.linear_max) / (
        1 + torch.abs(angular_velocity)
    )

    return linear_velocity, angular_velocity


def load_and_transform_recording_data(
    env: LocomanipulationSDGEnv,
    input_episode_data: EpisodeData,
    recording_step: int,
    reference_pose: torch.Tensor,
    target_pose: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load recording data and transform hand targets to current reference frame.

    Args:
        env: The locomanipulation SDG environment
        input_episode_data: Input episode data from static manipulation
        recording_step: Current step in the recording
        reference_pose: Original reference pose for the hand targets
        target_pose: Current target pose to transform to

    Returns:
        Tuple of transformed (left_hand_pose, right_hand_pose)
    """
    recording_item = env.load_input_data(input_episode_data, recording_step)
    if recording_item is None:
        return None, None

    left_hand_pose = transform_relative_pose(recording_item.left_hand_pose_target, reference_pose, target_pose)[0]
    right_hand_pose = transform_relative_pose(recording_item.right_hand_pose_target, reference_pose, target_pose)[0]

    return left_hand_pose, right_hand_pose


def setup_navigation_scene(
    env: LocomanipulationSDGEnv,
    input_episode_data: EpisodeData,
    approach_distance: float,
    randomize_placement: bool = True,
) -> tuple[OccupancyMap, ParameterizedPath, RelativePose, RelativePose]:
    """Set up the navigation scene with occupancy map and path planning.

    Args:
        env: The locomanipulation SDG environment
        input_episode_data: Input episode data
        approach_distance: Buffer distance from final goal
        randomize_placement: Whether to randomize fixture placement

    Returns:
        Tuple of (occupancy_map, path_helper, base_goal, base_goal_approach)
    """
    # Create base occupancy map
    occupancy_map = merge_occupancy_maps(
        [
            OccupancyMap.make_empty(start=(-7, -7), end=(7, 7), resolution=0.05),
            env.get_start_fixture().get_occupancy_map(),
        ]
    )

    # Randomize fixture placement if enabled
    if randomize_placement:
        fixtures = [env.get_end_fixture()] + env.get_obstacle_fixtures()
        for fixture in fixtures:
            place_randomly(fixture, occupancy_map.buffered_meters(1.0))
            occupancy_map = merge_occupancy_maps([occupancy_map, fixture.get_occupancy_map()])

    # Compute goal poses from initial state
    initial_state = env.load_input_data(input_episode_data, 0)
    base_goal = RelativePose(
        relative_pose=transform_mul(transform_inv(initial_state.fixture_pose), initial_state.base_pose),
        parent=env.get_end_fixture(),
    )
    base_goal_approach = RelativePose(
        relative_pose=torch.tensor([-approach_distance, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]), parent=base_goal
    )

    # Plan navigation path
    base_path = plan_path(
        start=env.get_base(), end=base_goal_approach, occupancy_map=occupancy_map.buffered_meters(0.15)
    )
    base_path_helper = ParameterizedPath(base_path)

    return occupancy_map, base_path_helper, base_goal, base_goal_approach


def handle_grasp_state(
    env: LocomanipulationSDGEnv,
    input_episode_data: EpisodeData,
    recording_step: int,
    lift_step: int,
    output_data: LocomanipulationSDGOutputData,
) -> tuple[int, LocomanipulationSDGDataGenerationState]:
    """Handle the GRASP_OBJECT state logic.

    Args:
        env: The environment
        input_episode_data: Input episode data
        recording_step: Current recording step
        lift_step: Step to transition to lift phase
        output_data: Output data to populate

    Returns:
        Tuple of (next_recording_step, next_state)
    """
    recording_item = env.load_input_data(input_episode_data, recording_step)

    # Set control targets - robot stays stationary during grasping
    output_data.data_generation_state = int(LocomanipulationSDGDataGenerationState.GRASP_OBJECT)
    output_data.recording_step = recording_step
    output_data.base_velocity_target = torch.tensor([0.0, 0.0, 0.0])

    # Transform hand poses relative to object
    output_data.left_hand_pose_target = transform_relative_pose(
        recording_item.left_hand_pose_target, recording_item.object_pose, env.get_object().get_pose()
    )[0]
    output_data.right_hand_pose_target = transform_relative_pose(
        recording_item.right_hand_pose_target, recording_item.base_pose, env.get_base().get_pose()
    )[0]
    output_data.left_hand_joint_positions_target = recording_item.left_hand_joint_positions_target
    output_data.right_hand_joint_positions_target = recording_item.right_hand_joint_positions_target

    # Update state

    next_recording_step = recording_step + 1
    next_state = (
        LocomanipulationSDGDataGenerationState.LIFT_OBJECT
        if next_recording_step > lift_step
        else LocomanipulationSDGDataGenerationState.GRASP_OBJECT
    )

    return next_recording_step, next_state


def handle_lift_state(
    env: LocomanipulationSDGEnv,
    input_episode_data: EpisodeData,
    recording_step: int,
    navigate_step: int,
    output_data: LocomanipulationSDGOutputData,
) -> tuple[int, LocomanipulationSDGDataGenerationState]:
    """Handle the LIFT_OBJECT state logic.

    Args:
        env: The environment
        input_episode_data: Input episode data
        recording_step: Current recording step
        navigate_step: Step to transition to navigation phase
        output_data: Output data to populate

    Returns:
        Tuple of (next_recording_step, next_state)
    """
    recording_item = env.load_input_data(input_episode_data, recording_step)

    # Set control targets - robot stays stationary during lifting
    output_data.data_generation_state = int(LocomanipulationSDGDataGenerationState.LIFT_OBJECT)
    output_data.recording_step = recording_step
    output_data.base_velocity_target = torch.tensor([0.0, 0.0, 0.0])

    # Transform hand poses relative to base
    output_data.left_hand_pose_target = transform_relative_pose(
        recording_item.left_hand_pose_target, recording_item.base_pose, env.get_base().get_pose()
    )[0]
    output_data.right_hand_pose_target = transform_relative_pose(
        recording_item.right_hand_pose_target, recording_item.object_pose, env.get_object().get_pose()
    )[0]
    output_data.left_hand_joint_positions_target = recording_item.left_hand_joint_positions_target
    output_data.right_hand_joint_positions_target = recording_item.right_hand_joint_positions_target

    # Update state
    next_recording_step = recording_step + 1
    next_state = (
        LocomanipulationSDGDataGenerationState.NAVIGATE
        if next_recording_step > navigate_step
        else LocomanipulationSDGDataGenerationState.LIFT_OBJECT
    )

    return next_recording_step, next_state


def handle_navigate_state(
    env: LocomanipulationSDGEnv,
    input_episode_data: EpisodeData,
    recording_step: int,
    base_path_helper: ParameterizedPath,
    base_goal_approach: RelativePose,
    config: LocomanipulationSDGControlConfig,
    output_data: LocomanipulationSDGOutputData,
) -> LocomanipulationSDGDataGenerationState:
    """Handle the NAVIGATE state logic.

    Args:
        env: The environment
        input_episode_data: Input episode data
        recording_step: Current recording step
        base_path_helper: Parameterized path for navigation
        base_goal_approach: Approach pose goal
        config: Navigation control configuration
        output_data: Output data to populate

    Returns:
        Next state
    """
    recording_item = env.load_input_data(input_episode_data, recording_step)
    current_pose = env.get_base().get_pose_2d()[0]

    # Find target point along path using pure pursuit algorithm
    _, nearest_path_length, _, _ = base_path_helper.find_nearest(current_pose[:2])
    target_xy = base_path_helper.get_point_by_distance(distance=nearest_path_length + config.following_offset)

    # Compute navigation velocities
    linear_velocity, angular_velocity = compute_navigation_velocity(current_pose, target_xy, config)

    # Set control targets
    output_data.data_generation_state = int(LocomanipulationSDGDataGenerationState.NAVIGATE)
    output_data.recording_step = recording_step
    output_data.base_velocity_target = torch.tensor([linear_velocity, 0.0, angular_velocity])

    # Transform hand poses relative to base
    output_data.left_hand_pose_target = transform_relative_pose(
        recording_item.left_hand_pose_target, recording_item.base_pose, env.get_base().get_pose()
    )[0]
    output_data.right_hand_pose_target = transform_relative_pose(
        recording_item.right_hand_pose_target, recording_item.base_pose, env.get_base().get_pose()
    )[0]
    output_data.left_hand_joint_positions_target = recording_item.left_hand_joint_positions_target
    output_data.right_hand_joint_positions_target = recording_item.right_hand_joint_positions_target

    # Check if close enough to approach goal to transition
    goal_xy = base_goal_approach.get_pose_2d()[0, :2]
    distance_to_goal = torch.sqrt(torch.sum((current_pose[:2] - goal_xy) ** 2))

    return (
        LocomanipulationSDGDataGenerationState.APPROACH
        if distance_to_goal < config.distance_threshold
        else LocomanipulationSDGDataGenerationState.NAVIGATE
    )


def handle_approach_state(
    env: LocomanipulationSDGEnv,
    input_episode_data: EpisodeData,
    recording_step: int,
    base_goal: RelativePose,
    config: LocomanipulationSDGControlConfig,
    output_data: LocomanipulationSDGOutputData,
) -> LocomanipulationSDGDataGenerationState:
    """Handle the APPROACH state logic.

    Args:
        env: The environment
        input_episode_data: Input episode data
        recording_step: Current recording step
        base_goal: Final goal pose
        config: Navigation control configuration
        output_data: Output data to populate

    Returns:
        Next state
    """
    recording_item = env.load_input_data(input_episode_data, recording_step)
    current_pose = env.get_base().get_pose_2d()[0]

    # Navigate directly to final goal position
    goal_xy = base_goal.get_pose_2d()[0, :2]
    linear_velocity, angular_velocity = compute_navigation_velocity(current_pose, goal_xy, config)

    # Set control targets
    output_data.data_generation_state = int(LocomanipulationSDGDataGenerationState.APPROACH)
    output_data.recording_step = recording_step
    output_data.base_velocity_target = torch.tensor([linear_velocity, 0.0, angular_velocity])

    # Transform hand poses relative to base
    output_data.left_hand_pose_target = transform_relative_pose(
        recording_item.left_hand_pose_target, recording_item.base_pose, env.get_base().get_pose()
    )[0]
    output_data.right_hand_pose_target = transform_relative_pose(
        recording_item.right_hand_pose_target, recording_item.base_pose, env.get_base().get_pose()
    )[0]
    output_data.left_hand_joint_positions_target = recording_item.left_hand_joint_positions_target
    output_data.right_hand_joint_positions_target = recording_item.right_hand_joint_positions_target

    # Check if close enough to final goal to start drop-off
    distance_to_goal = torch.sqrt(torch.sum((current_pose[:2] - goal_xy) ** 2))

    return (
        LocomanipulationSDGDataGenerationState.DROP_OFF_OBJECT
        if distance_to_goal < config.distance_threshold
        else LocomanipulationSDGDataGenerationState.APPROACH
    )


def handle_drop_off_state(
    env: LocomanipulationSDGEnv,
    input_episode_data: EpisodeData,
    recording_step: int,
    base_goal: RelativePose,
    config: LocomanipulationSDGControlConfig,
    output_data: LocomanipulationSDGOutputData,
) -> tuple[int, LocomanipulationSDGDataGenerationState | None]:
    """Handle the DROP_OFF_OBJECT state logic.

    Args:
        env: The environment
        input_episode_data: Input episode data
        recording_step: Current recording step
        base_goal: Final goal pose
        config: Navigation control configuration
        output_data: Output data to populate

    Returns:
        Tuple of (next_recording_step, next_state)
    """
    recording_item = env.load_input_data(input_episode_data, recording_step)
    if recording_item is None:
        return recording_step, None

    # Compute orientation control to face target orientation
    current_pose = env.get_base().get_pose_2d()[0]
    target_pose = base_goal.get_pose_2d()[0]
    current_yaw = current_pose[2]
    target_yaw = target_pose[2]
    delta_yaw = target_yaw - current_yaw
    delta_yaw = (delta_yaw + torch.pi) % (2 * torch.pi) - torch.pi

    angular_velocity = config.angular_gain * delta_yaw
    linear_velocity = 0.0  # Stay in place while orienting

    # Set control targets
    output_data.data_generation_state = int(LocomanipulationSDGDataGenerationState.DROP_OFF_OBJECT)
    output_data.recording_step = recording_step
    output_data.base_velocity_target = torch.tensor([linear_velocity, 0.0, angular_velocity])

    # Transform hand poses relative to end fixture
    output_data.left_hand_pose_target = transform_relative_pose(
        recording_item.left_hand_pose_target,
        recording_item.fixture_pose,
        env.get_end_fixture().get_pose(),
    )[0]
    output_data.right_hand_pose_target = transform_relative_pose(
        recording_item.right_hand_pose_target,
        recording_item.fixture_pose,
        env.get_end_fixture().get_pose(),
    )[0]
    output_data.left_hand_joint_positions_target = recording_item.left_hand_joint_positions_target
    output_data.right_hand_joint_positions_target = recording_item.right_hand_joint_positions_target

    # Continue playback if orientation is within threshold
    next_recording_step = recording_step + 1 if abs(delta_yaw) < config.angle_threshold else recording_step

    return next_recording_step, LocomanipulationSDGDataGenerationState.DROP_OFF_OBJECT


def populate_output_data(
    env: LocomanipulationSDGEnv,
    output_data: LocomanipulationSDGOutputData,
    base_goal: RelativePose,
    base_goal_approach: RelativePose,
    base_path: torch.Tensor,
) -> None:
    """Populate remaining output data fields.

    Args:
        env: The environment
        output_data: Output data to populate
        base_goal: Final goal pose
        base_goal_approach: Approach goal pose
        base_path: Planned navigation path
    """
    output_data.base_pose = env.get_base().get_pose()
    output_data.object_pose = env.get_object().get_pose()
    output_data.start_fixture_pose = env.get_start_fixture().get_pose()
    output_data.end_fixture_pose = env.get_end_fixture().get_pose()
    output_data.base_goal_pose = base_goal.get_pose()
    output_data.base_goal_approach_pose = base_goal_approach.get_pose()
    output_data.base_path = base_path

    # Collect obstacle poses
    obstacle_poses = []
    for obstacle in env.get_obstacle_fixtures():
        obstacle_poses.append(obstacle.get_pose())
    if obstacle_poses:
        output_data.obstacle_fixture_poses = torch.cat(obstacle_poses, dim=0)[None, :]
    else:
        output_data.obstacle_fixture_poses = torch.empty((1, 0, 7))  # Empty tensor with correct shape


def replay(
    env: LocomanipulationSDGEnv,
    input_episode_data: EpisodeData,
    lift_step: int,
    navigate_step: int,
    draw_visualization: bool = False,
    angular_gain: float = 2.0,
    linear_gain: float = 1.0,
    linear_max: float = 1.0,
    distance_threshold: float = 0.1,
    following_offset: float = 0.6,
    angle_threshold: float = 0.2,
    approach_distance: float = 1.0,
    randomize_placement: bool = True,
) -> None:
    """Replay a locomanipulation SDG episode with state machine control.

    This function implements a state machine for locomanipulation SDG, where the robot:
    1. Grasps an object at the start position
    2. Lifts the object while stationary
    3. Navigates with the object to an approach position
    4. Approaches the final goal position
    5. Places the object at the end position

    Args:
        env: The locomanipulation SDG environment
        input_episode_data: Static manipulation episode data to replay
        lift_step: Recording step where lifting phase begins
        navigate_step: Recording step where navigation phase begins
        draw_visualization: Whether to visualize occupancy map and path
        angular_gain: Proportional gain for angular velocity control
        linear_gain: Proportional gain for linear velocity control
        linear_max: Maximum linear velocity (m/s)
        distance_threshold: Distance threshold for state transitions (m)
        following_offset: Look-ahead distance for path following (m)
        angle_threshold: Angular threshold for orientation control (rad)
        approach_distance: Buffer distance from final goal (m)
        randomize_placement: Whether to randomize obstacle placement
    """

    # Initialize environment to starting state
    env.reset_to(state=input_episode_data.get_initial_state(), env_ids=torch.tensor([0]), is_relative=True)

    # Create navigation control configuration
    config = LocomanipulationSDGControlConfig(
        angular_gain=angular_gain,
        linear_gain=linear_gain,
        linear_max=linear_max,
        distance_threshold=distance_threshold,
        following_offset=following_offset,
        angle_threshold=angle_threshold,
        approach_distance=approach_distance,
    )

    # Set up navigation scene and path planning
    occupancy_map, base_path_helper, base_goal, base_goal_approach = setup_navigation_scene(
        env, input_episode_data, approach_distance, randomize_placement
    )

    # Visualize occupancy map and path if requested
    if draw_visualization:
        occupancy_map_add_to_stage(
            occupancy_map,
            stage=sim_utils.get_current_stage(),
            path="/OccupancyMap",
            z_offset=0.01,
            draw_path=base_path_helper.points,
        )

    # Initialize state machine
    output_data = LocomanipulationSDGOutputData()
    current_state = LocomanipulationSDGDataGenerationState.GRASP_OBJECT
    recording_step = 0

    # Main simulation loop with state machine
    while simulation_app.is_running() and not simulation_app.is_exiting():
        print(f"Current state: {current_state.name}, Recording step: {recording_step}")

        # Execute state-specific logic using helper functions
        if current_state == LocomanipulationSDGDataGenerationState.GRASP_OBJECT:
            recording_step, current_state = handle_grasp_state(
                env, input_episode_data, recording_step, lift_step, output_data
            )

        elif current_state == LocomanipulationSDGDataGenerationState.LIFT_OBJECT:
            recording_step, current_state = handle_lift_state(
                env, input_episode_data, recording_step, navigate_step, output_data
            )

        elif current_state == LocomanipulationSDGDataGenerationState.NAVIGATE:
            current_state = handle_navigate_state(
                env, input_episode_data, recording_step, base_path_helper, base_goal_approach, config, output_data
            )

        elif current_state == LocomanipulationSDGDataGenerationState.APPROACH:
            current_state = handle_approach_state(
                env, input_episode_data, recording_step, base_goal, config, output_data
            )

        elif current_state == LocomanipulationSDGDataGenerationState.DROP_OFF_OBJECT:
            recording_step, next_state = handle_drop_off_state(
                env, input_episode_data, recording_step, base_goal, config, output_data
            )
            if next_state is None:  # End of episode data
                break
            current_state = next_state

        # Populate additional output data fields
        populate_output_data(env, output_data, base_goal, base_goal_approach, base_path_helper.points)

        # Attach output data to environment for recording
        env._locomanipulation_sdg_output_data = output_data

        # Build and execute action
        action = env.build_action_vector(
            base_velocity_target=output_data.base_velocity_target,
            left_hand_joint_positions_target=output_data.left_hand_joint_positions_target,
            right_hand_joint_positions_target=output_data.right_hand_joint_positions_target,
            left_hand_pose_target=output_data.left_hand_pose_target,
            right_hand_pose_target=output_data.right_hand_pose_target,
        )

        env.step(action)


if __name__ == "__main__":
    with torch.no_grad():
        # Create environment
        if args_cli.task is not None:
            env_name = args_cli.task.split(":")[-1]
        if env_name is None:
            raise ValueError("Task/env name was not specified nor found in the dataset.")

        env_cfg = parse_env_cfg(env_name, device=args_cli.device, num_envs=1)
        env_cfg.sim.device = "cpu"
        env_cfg.recorders.dataset_export_dir_path = os.path.dirname(args_cli.output_file)
        env_cfg.recorders.dataset_filename = os.path.basename(args_cli.output_file)

        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

        # Load input data
        input_dataset_file_handler = HDF5DatasetFileHandler()
        input_dataset_file_handler.open(args_cli.dataset)

        for i in range(args_cli.num_runs):
            if args_cli.demo is None:
                demo = random.choice(list(input_dataset_file_handler.get_episode_names()))
            else:
                demo = args_cli.demo

            input_episode_data = input_dataset_file_handler.load_episode(demo, args_cli.device)

            replay(
                env=env,
                input_episode_data=input_episode_data,
                lift_step=args_cli.lift_step,
                navigate_step=args_cli.navigate_step,
                draw_visualization=args_cli.draw_visualization,
                angular_gain=args_cli.angular_gain,
                linear_gain=args_cli.linear_gain,
                linear_max=args_cli.linear_max,
                distance_threshold=args_cli.distance_threshold,
                following_offset=args_cli.following_offset,
                angle_threshold=args_cli.angle_threshold,
                approach_distance=args_cli.approach_distance,
                randomize_placement=args_cli.randomize_placement,
            )

        env.reset()  # FIXME: hack to handle missing final recording
        env.close()

        simulation_app.close()
