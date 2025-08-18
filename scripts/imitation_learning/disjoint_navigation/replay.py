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
parser.add_argument("--dataset", type=str)
parser.add_argument("--lift_step", type=int)
parser.add_argument("--navigate_step", type=int)
parser.add_argument("--demo", type=str, default="demo_0")
parser.add_argument("--num_runs", type=int, default=1)
parser.add_argument("--draw_visualization", type=bool, default=False)
parser.add_argument("--angular_gain", type=float, default=2.0)
parser.add_argument("--linear_gain", type=float, default=1.0)
parser.add_argument("--linear_max", type=float, default=1.0)
parser.add_argument("--distance_threshold", type=float, default=0.2)
parser.add_argument("--following_offset", type=float, default=0.6)
parser.add_argument("--angle_threshold", type=float, default=0.2)
parser.add_argument("--approach_distance", type=float, default=0.5)
parser.add_argument("--randomize_placement", type=bool, default=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import random
import torch
import omni.kit

from enum import Enum

from occupancy_map import OccupancyMap, merge_occupancy_maps
from visualization import occupancy_map_add_to_stage
from path_utils import PathHelper

from common import (
    DisjointNavScenario, 
    DisjointNavRecording, 
    RelativePose, 
    transform_inv, 
    transform_mul, 
    transform_relative_pose,
    plan_path,
    SceneFixture,
    HasOccupancyMap,
    place_randomly
)
from g1_29dof import (
    G1DisjointNavRecording,
    G1DisjointNavScenario
)


class ReplayState(Enum):
    GRASP_OBJECT = 0
    LIFT_OBJECT = 1
    NAVIGATE = 2
    APPROACH = 3
    DROP_OFF_OBJECT = 4
    DONE = 5


def replay(
        scenario: DisjointNavScenario, 
        recording: DisjointNavRecording,
        lift_step: int,
        navigate_step: int,
        draw_visualization: bool = False,
        angular_gain = 2.0,
        linear_gain = 1.0,
        linear_max = 1.0,
        distance_threshold = 0.2,
        following_offset = 0.6,
        angle_threshold = 0.2,
        approach_distance: float = 0.5,
        randomize_placement: bool = True
    ):

    scenario.reset(
        initial_state=recording.get_initial_state()
    )

    occupancy_map = merge_occupancy_maps([
        OccupancyMap.make_empty(start=(-7, -7), end=(7, 7), resolution=0.05),
        scenario.get_start_fixture().get_occupancy_map()
    ])

    if randomize_placement:

        fixtures = [scenario.get_end_fixture()] + scenario.get_obstacle_fixtures()

        for fixture in fixtures:
            place_randomly(
                fixture,
                occupancy_map.buffered_meters(1.0)
            )

            occupancy_map = merge_occupancy_maps([
                occupancy_map,
                fixture.get_occupancy_map()
            ])


    initial_state = recording.get_item(step=0)
    
    base_goal = RelativePose(
        relative_pose=transform_mul(
            transform_inv(initial_state.fixture_pose),
            initial_state.base_pose
        ),
        parent=scenario.get_end_fixture()
    )

    base_goal_approach = RelativePose(
        relative_pose=torch.tensor([-approach_distance, 0., 0., 1.0, 0., 0., 0.]),
        parent=base_goal
    )

    base_path = plan_path(
        start=scenario.get_base(),
        end=base_goal_approach,
        occupancy_map=occupancy_map.buffered_meters(0.15)
    )

    base_path_helper = PathHelper(base_path)

    if draw_visualization:

        occupancy_map_add_to_stage(
            occupancy_map,
            stage=omni.usd.get_context().get_stage(),
            path="/OccupancyMap",
            z_offset=0.01,
            draw_path=base_path
        )

    state = ReplayState.GRASP_OBJECT

    recording_step = 0

    while simulation_app.is_running() and not simulation_app.is_exiting():

        print(state)

        if state == ReplayState.GRASP_OBJECT:

            recording_item = recording.get_item(recording_step)

            scenario.set_base_velocity_target(torch.tensor([0., 0., 0.]))
            scenario.set_left_hand_pose_target(
                transform_relative_pose(
                    recording_item.left_hand_pose_target,
                    recording_item.object_pose,
                    scenario.get_object().get_pose()
                )[0]
            )
            scenario.set_right_hand_pose_target(
                transform_relative_pose(
                    recording_item.right_hand_pose_target,
                    recording_item.object_pose,
                    scenario.get_object().get_pose()
                )[0]
            )
            scenario.set_left_hand_joint_positions_target(recording_item.left_hand_joint_positions_target)
            scenario.set_right_hand_joint_positions_target(recording_item.right_hand_joint_positions_target)

            recording_step += 1

            if recording_step > lift_step:
                state = ReplayState.LIFT_OBJECT

        elif state == ReplayState.LIFT_OBJECT:

            recording_item = recording.get_item(recording_step)

            scenario.set_base_velocity_target(torch.tensor([0., 0., 0.]))
            scenario.set_left_hand_pose_target(
                transform_relative_pose(
                    recording_item.left_hand_pose_target,
                    recording_item.base_pose,
                    scenario.get_base().get_pose()
                )[0]
            )
            scenario.set_right_hand_pose_target(
                transform_relative_pose(
                    recording_item.right_hand_pose_target,
                    recording_item.base_pose,
                    scenario.get_base().get_pose()
                )[0]
            )
            scenario.set_left_hand_joint_positions_target(recording_item.left_hand_joint_positions_target)
            scenario.set_right_hand_joint_positions_target(recording_item.right_hand_joint_positions_target)

            recording_step += 1

            if recording_step > navigate_step:
                state = ReplayState.NAVIGATE

        elif state == ReplayState.NAVIGATE:
            
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
            
            linear_velocity = torch.clip(linear_gain * delta_distance, 0., linear_max) / (1 + torch.abs(angular_velocity))

            # Set commands

            recording_item = recording.get_item(recording_step)

            scenario.set_base_velocity_target(torch.tensor([linear_velocity, 0., angular_velocity]))
            scenario.set_left_hand_pose_target(
                transform_relative_pose(
                    recording_item.left_hand_pose_target,
                    recording_item.base_pose,
                    scenario.get_base().get_pose()
                )[0]
            )
            scenario.set_right_hand_pose_target(
                transform_relative_pose(
                    recording_item.right_hand_pose_target,
                    recording_item.base_pose,
                    scenario.get_base().get_pose()
                )[0]
            )
            scenario.set_left_hand_joint_positions_target(recording_item.left_hand_joint_positions_target)
            scenario.set_right_hand_joint_positions_target(recording_item.right_hand_joint_positions_target)

            # Update state

            goal_xy = base_goal_approach.get_pose_2d()[0, :2]
            distance_to_goal = torch.sqrt(torch.sum((current_xy - goal_xy)**2))

            if distance_to_goal < distance_threshold:
                state = ReplayState.APPROACH

        elif state == ReplayState.APPROACH:

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
            
            linear_velocity = torch.clip(linear_gain * delta_distance, 0., linear_max) / (1 + torch.abs(angular_velocity))

            # Set commands

            recording_item = recording.get_item(recording_step)

            scenario.set_base_velocity_target(torch.tensor([linear_velocity, 0., angular_velocity]))
            scenario.set_left_hand_pose_target(
                transform_relative_pose(
                    recording_item.left_hand_pose_target,
                    recording_item.base_pose,
                    scenario.get_base().get_pose()
                )[0]
            )
            scenario.set_right_hand_pose_target(
                transform_relative_pose(
                    recording_item.right_hand_pose_target,
                    recording_item.base_pose,
                    scenario.get_base().get_pose()
                )[0]
            )
            scenario.set_left_hand_joint_positions_target(recording_item.left_hand_joint_positions_target)
            scenario.set_right_hand_joint_positions_target(recording_item.right_hand_joint_positions_target)

            # Update state

            goal_xy = base_goal.get_pose_2d()[0, :2]
            distance_to_goal = torch.sqrt(torch.sum((current_xy - goal_xy)**2))

            if distance_to_goal < distance_threshold:
                state = ReplayState.DROP_OFF_OBJECT

        elif state == ReplayState.DROP_OFF_OBJECT:

            # Calculate turn rate to face object
            current_pose = scenario.get_base().get_pose_2d()[0]
            target_pose = base_goal.get_pose_2d()[0]
            current_yaw = current_pose[2]
            target_yaw = target_pose[2]
            delta_yaw = target_yaw - current_yaw
            delta_yaw = (delta_yaw + torch.pi) % (2 * torch.pi) - torch.pi  # smallest angle
            angular_velocity = angular_gain * delta_yaw
            linear_velocity = 0.

            # Set commands

            recording_item = recording.get_item(recording_step)

            if recording_item is None:
                return

            scenario.set_base_velocity_target(torch.tensor([linear_velocity, 0., angular_velocity]))
            scenario.set_left_hand_pose_target(
                transform_relative_pose(
                    recording_item.left_hand_pose_target,
                    recording_item.fixture_pose,
                    scenario.get_end_fixture().get_pose()
                )[0]
            )
            scenario.set_right_hand_pose_target(
                transform_relative_pose(
                    recording_item.right_hand_pose_target,
                    recording_item.fixture_pose,
                    scenario.get_end_fixture().get_pose()
                )[0]
            )
            scenario.set_left_hand_joint_positions_target(recording_item.left_hand_joint_positions_target)
            scenario.set_right_hand_joint_positions_target(recording_item.right_hand_joint_positions_target)

            # Continue playback if angle is within threshold
            if delta_yaw < angle_threshold:
                recording_step += 1


        scenario.step()


if __name__ == "__main__":

    recording = G1DisjointNavRecording(
        path=args_cli.dataset,
        demo=args_cli.demo,
        device=args_cli.device
    )


    scenario = G1DisjointNavScenario()

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
            randomize_placement=args_cli.randomize_placement
        )


    scenario.close()

    simulation_app.close()
