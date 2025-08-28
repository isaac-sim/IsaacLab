# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to replay demonstrations with Isaac Lab environments."""

"""Launch Isaac Sim Simulator first."""


import argparse

import pinocchio

from isaaclab.app import AppLauncher

# Launch Isaac Lab
parser = argparse.ArgumentParser(description="Disjoint navigation")
parser.add_argument("--dataset", type=str, help="The static manipulation dataset recorded via teleoperation.")
parser.add_argument("--output_dir", type=str, help="The directory to output the generated dataset.")
parser.add_argument("--output_file_name", type=str, help="The file name for the generated output dataset.")
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
parser.add_argument("--demo", type=str, default="demo_0", help="The demo in the input dataset to use.")
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
    help="Whether or not to randomize the placement of fixtures in the scene upon scenario initialization.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import random
import torch

import omni.kit
from common import (
    DisjointNavRecording,
    DisjointNavScenario,
    RelativePose,
    place_randomly,
    plan_path,
    transform_inv,
    transform_mul,
    transform_relative_pose,
    DisjointNavReplayState,
    DisjointNavReplayTask
)
from scripts.imitation_learning.disjoint_navigation.g1_disjoint_nav_env import G1DisjointNavRecording, G1DisjointNavScenario
from occupancy_map import OccupancyMap, merge_occupancy_maps
from path_utils import PathHelper
from visualization import occupancy_map_add_to_stage



def replay(
    scenario: DisjointNavScenario,
    recording: DisjointNavRecording,
    lift_step: int,
    navigate_step: int,
    draw_visualization: bool = False,
    angular_gain=2.0,
    linear_gain=1.0,
    linear_max=1.0,
    distance_threshold=0.2,
    following_offset=0.6,
    angle_threshold=0.2,
    approach_distance: float = 0.5,
    randomize_placement: bool = True,
):

    scenario.reset(initial_state=recording.get_initial_state())

    occupancy_map = merge_occupancy_maps([
        OccupancyMap.make_empty(start=(-7, -7), end=(7, 7), resolution=0.05),
        scenario.get_start_fixture().get_occupancy_map(),
    ])

    if randomize_placement:

        fixtures = [scenario.get_end_fixture()] + scenario.get_obstacle_fixtures()

        for fixture in fixtures:
            place_randomly(fixture, occupancy_map.buffered_meters(1.0))

            occupancy_map = merge_occupancy_maps([occupancy_map, fixture.get_occupancy_map()])

    initial_state = recording.get_item(step=0)

    base_goal = RelativePose(
        relative_pose=transform_mul(transform_inv(initial_state.fixture_pose), initial_state.base_pose),
        parent=scenario.get_end_fixture(),
    )

    base_goal_approach = RelativePose(
        relative_pose=torch.tensor([-approach_distance, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]), parent=base_goal
    )

    base_path = plan_path(
        start=scenario.get_base(), end=base_goal_approach, occupancy_map=occupancy_map.buffered_meters(0.15)
    )

    base_path_helper = PathHelper(base_path)

    if draw_visualization:

        occupancy_map_add_to_stage(
            occupancy_map,
            stage=omni.usd.get_context().get_stage(),
            path="/OccupancyMap",
            z_offset=0.01,
            draw_path=base_path,
        )

    state = DisjointNavReplayState()
    task = DisjointNavReplayTask.GRASP_OBJECT
    recording_step = 0

    while simulation_app.is_running() and not simulation_app.is_exiting():

        print(task)

        if task == DisjointNavReplayTask.GRASP_OBJECT:

            recording_item = recording.get_item(recording_step)

            state.task = int(task)
            state.recording_step = recording_step
            state.base_velocity_target = torch.tensor([0.0, 0.0, 0.0])
            state.left_hand_pose_target = transform_relative_pose(
                recording_item.left_hand_pose_target, recording_item.object_pose, scenario.get_object().get_pose()
            )[0]
            state.right_hand_pose_target = transform_relative_pose(
                recording_item.right_hand_pose_target, recording_item.object_pose, scenario.get_object().get_pose()
            )[0]
            state.left_hand_joint_positions_target = recording_item.left_hand_joint_positions_target
            state.right_hand_joint_positions_target = recording_item.right_hand_joint_positions_target

            recording_step += 1

            if recording_step > lift_step:
                task = DisjointNavReplayTask.LIFT_OBJECT

        elif task == DisjointNavReplayTask.LIFT_OBJECT:

            recording_item = recording.get_item(recording_step)

            state.task = int(task)
            state.recording_step = recording_step
            state.base_velocity_target = torch.tensor([0.0, 0.0, 0.0])
            state.left_hand_pose_target = transform_relative_pose(
                recording_item.left_hand_pose_target, recording_item.base_pose, scenario.get_base().get_pose()
            )[0]
            state.right_hand_pose_target = transform_relative_pose(
                recording_item.right_hand_pose_target, recording_item.base_pose, scenario.get_base().get_pose()
            )[0]
            state.left_hand_joint_positions_target = recording_item.left_hand_joint_positions_target
            state.right_hand_joint_positions_target = recording_item.right_hand_joint_positions_target

            recording_step += 1

            if recording_step > navigate_step:
                task = DisjointNavReplayTask.NAVIGATE

        elif task == DisjointNavReplayTask.NAVIGATE:

            recording_item = recording.get_item(recording_step)

            # Compute base velocity
            current_pose = scenario.get_base().get_pose_2d()[0]
            current_xy = current_pose[:2]
            current_yaw = current_pose[2]

            _, nearest_path_point_length_along_path, _, _ = base_path_helper.find_nearest(current_xy)

            assert nearest_path_point_length_along_path is not None

            target_xy = base_path_helper.get_point_by_distance(
                distance=nearest_path_point_length_along_path + following_offset
            )

            delta_xy = target_xy - current_xy
            delta_distance = torch.sqrt(torch.sum(delta_xy**2))

            target_yaw = torch.arctan2(delta_xy[1], delta_xy[0])
            delta_yaw = target_yaw - current_yaw
            delta_yaw = (delta_yaw + torch.pi) % (2 * torch.pi) - torch.pi  # smallest angle

            angular_velocity = angular_gain * delta_yaw

            linear_velocity = torch.clip(linear_gain * delta_distance, 0.0, linear_max) / (
                1 + torch.abs(angular_velocity)
            )

            # Set commands

            recording_item = recording.get_item(recording_step)

            state.task = int(task)
            state.recording_step = recording_step
            state.base_velocity_target = torch.tensor([linear_velocity, 0.0, angular_velocity])
            state.left_hand_pose_target = transform_relative_pose(
                recording_item.left_hand_pose_target, recording_item.base_pose, scenario.get_base().get_pose()
            )[0]
            state.right_hand_pose_target = transform_relative_pose(
                recording_item.right_hand_pose_target, recording_item.base_pose, scenario.get_base().get_pose()
            )[0]
            state.left_hand_joint_positions_target = recording_item.left_hand_joint_positions_target
            state.right_hand_joint_positions_target = recording_item.right_hand_joint_positions_target

            # Update state

            goal_xy = base_goal_approach.get_pose_2d()[0, :2]
            distance_to_goal = torch.sqrt(torch.sum((current_xy - goal_xy) ** 2))

            if distance_to_goal < distance_threshold:
                task = DisjointNavReplayTask.APPROACH

        elif task == DisjointNavReplayTask.APPROACH:

            recording_item = recording.get_item(recording_step)

            # Compute base velocity
            current_pose = scenario.get_base().get_pose_2d()[0]
            current_xy = current_pose[:2]
            current_yaw = current_pose[2]

            target_xy = base_goal.get_pose_2d()[0, :2]

            delta_xy = target_xy - current_xy
            delta_distance = torch.sqrt(torch.sum(delta_xy**2))

            target_yaw = torch.arctan2(delta_xy[1], delta_xy[0])
            delta_yaw = target_yaw - current_yaw
            delta_yaw = (delta_yaw + torch.pi) % (2 * torch.pi) - torch.pi  # smallest angle

            angular_velocity = angular_gain * delta_yaw

            linear_velocity = torch.clip(linear_gain * delta_distance, 0.0, linear_max) / (
                1 + torch.abs(angular_velocity)
            )

            # Set commands

            recording_item = recording.get_item(recording_step)

            state.task = int(task)
            state.recording_step = recording_step
            state.base_velocity_target = torch.tensor([linear_velocity, 0.0, angular_velocity])
            state.left_hand_pose_target = transform_relative_pose(
                recording_item.left_hand_pose_target, recording_item.base_pose, scenario.get_base().get_pose()
            )[0]
            state.right_hand_pose_target = transform_relative_pose(
                recording_item.right_hand_pose_target, recording_item.base_pose, scenario.get_base().get_pose()
            )[0]
            state.left_hand_joint_positions_target = recording_item.left_hand_joint_positions_target
            state.right_hand_joint_positions_target = recording_item.right_hand_joint_positions_target
            
            # Update state

            goal_xy = base_goal.get_pose_2d()[0, :2]
            distance_to_goal = torch.sqrt(torch.sum((current_xy - goal_xy) ** 2))

            if distance_to_goal < distance_threshold:
                task = DisjointNavReplayTask.DROP_OFF_OBJECT

        elif task == DisjointNavReplayTask.DROP_OFF_OBJECT:

            # Calculate turn rate to face object
            current_pose = scenario.get_base().get_pose_2d()[0]
            target_pose = base_goal.get_pose_2d()[0]
            current_yaw = current_pose[2]
            target_yaw = target_pose[2]
            delta_yaw = target_yaw - current_yaw
            delta_yaw = (delta_yaw + torch.pi) % (2 * torch.pi) - torch.pi  # smallest angle
            angular_velocity = angular_gain * delta_yaw
            linear_velocity = 0.0

            # Set commands

            recording_item = recording.get_item(recording_step)

            if recording_item is None:
                return

            state.task = int(task)
            state.recording_step = recording_step
            state.base_velocity_target = torch.tensor([linear_velocity, 0.0, angular_velocity])
            state.left_hand_pose_target = transform_relative_pose(
                recording_item.left_hand_pose_target,
                recording_item.fixture_pose,
                scenario.get_end_fixture().get_pose(),
            )[0]
            state.right_hand_pose_target = transform_relative_pose(
                recording_item.right_hand_pose_target,
                recording_item.fixture_pose,
                scenario.get_end_fixture().get_pose(),
            )[0]
            state.left_hand_joint_positions_target = recording_item.left_hand_joint_positions_target
            state.right_hand_joint_positions_target = recording_item.right_hand_joint_positions_target

            # Continue playback if angle is within threshold
            if delta_yaw < angle_threshold:
                recording_step += 1

        # Populate remaining state items and attach to env so they get recorded by recorder manager
        state.base_pose = scenario.get_base().get_pose()
        state.object_pose = scenario.get_object().get_pose()
        state.start_fixture_pose = scenario.get_start_fixture().get_pose()
        state.end_fixture_pose = scenario.get_end_fixture().get_pose()
        state.base_goal_pose = base_goal.get_pose()
        state.base_goal_approach_pose = base_goal_approach.get_pose()
        state.base_path = base_path
        env = scenario.get_env()
        env._replay_state = state

        scenario.set_base_velocity_target(state.base_velocity_target)
        scenario.set_left_hand_joint_positions_target(state.left_hand_joint_positions_target)
        scenario.set_right_hand_joint_positions_target(state.right_hand_joint_positions_target)
        scenario.set_left_hand_pose_target(state.left_hand_pose_target)
        scenario.set_right_hand_pose_target(state.right_hand_pose_target)
        scenario.step()


if __name__ == "__main__":

    with torch.no_grad():
        recording = G1DisjointNavRecording(path=args_cli.dataset, demo=args_cli.demo, device=args_cli.device)

        scenario = G1DisjointNavScenario(output_dir=args_cli.output_dir, output_file_name=args_cli.output_file_name)

        for i in range(args_cli.num_runs):

            replay(
                scenario=scenario,
                recording=recording,
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

        scenario.close()

        simulation_app.close()
