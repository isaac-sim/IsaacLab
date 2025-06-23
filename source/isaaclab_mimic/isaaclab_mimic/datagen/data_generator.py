# Copyright (c) 2024-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
Base class for data generator.
"""
import asyncio
import numpy as np
import torch

import isaaclab.utils.math as PoseUtils
from isaaclab.envs import (
    ManagerBasedRLMimicEnv,
    MimicEnvCfg,
    SubTaskConstraintCoordinationScheme,
    SubTaskConstraintType,
)
from isaaclab.managers import TerminationTermCfg

from isaaclab_mimic.datagen.datagen_info import DatagenInfo
from isaaclab_mimic.datagen.selection_strategy import make_selection_strategy
from isaaclab_mimic.datagen.waypoint import MultiWaypoint, Waypoint, WaypointSequence, WaypointTrajectory

from .datagen_info_pool import DataGenInfoPool


def transform_source_data_segment_using_delta_object_pose(
    src_eef_poses: torch.Tensor,
    delta_obj_pose: torch.Tensor,
) -> torch.Tensor:
    """
    Transform a source data segment (object-centric subtask segment from source demonstration) using
    a delta object pose.

    Args:
        src_eef_poses: pose sequence (shape [T, 4, 4]) for the sequence of end effector control poses
            from the source demonstration
        delta_obj_pose: 4x4 delta object pose

    Returns:
        transformed_eef_poses: transformed pose sequence (shape [T, 4, 4])
    """
    return PoseUtils.pose_in_A_to_pose_in_B(
        pose_in_A=src_eef_poses,
        pose_A_in_B=delta_obj_pose[None],
    )


def transform_source_data_segment_using_object_pose(
    obj_pose: torch.Tensor,
    src_eef_poses: torch.Tensor,
    src_obj_pose: torch.Tensor,
) -> torch.Tensor:
    """
    Transform a source data segment (object-centric subtask segment from source demonstration) such that
    the relative poses between the target eef pose frame and the object frame are preserved. Recall that
    each object-centric subtask segment corresponds to one object, and consists of a sequence of
    target eef poses.

    Args:
        obj_pose: 4x4 object pose in current scene
        src_eef_poses: pose sequence (shape [T, 4, 4]) for the sequence of end effector control poses
            from the source demonstration
        src_obj_pose: 4x4 object pose from the source demonstration

    Returns:
        transformed_eef_poses: transformed pose sequence (shape [T, 4, 4])
    """

    # transform source end effector poses to be relative to source object frame
    src_eef_poses_rel_obj = PoseUtils.pose_in_A_to_pose_in_B(
        pose_in_A=src_eef_poses,
        pose_A_in_B=PoseUtils.pose_inv(src_obj_pose[None]),
    )

    # apply relative poses to current object frame to obtain new target eef poses
    transformed_eef_poses = PoseUtils.pose_in_A_to_pose_in_B(
        pose_in_A=src_eef_poses_rel_obj,
        pose_A_in_B=obj_pose[None],
    )
    return transformed_eef_poses


def get_delta_pose_with_scheme(
    src_obj_pose: torch.Tensor,
    cur_obj_pose: torch.Tensor,
    task_constraint: dict,
) -> torch.Tensor:
    """
    Get the delta pose with the given coordination scheme.

    Args:
        src_obj_pose: 4x4 object pose in source scene
        cur_obj_pose: 4x4 object pose in current scene
        task_constraint: task constraint dictionary

    Returns:
        delta_pose: 4x4 delta pose
    """
    coord_transform_scheme = task_constraint["coordination_scheme"]
    device = src_obj_pose.device
    if coord_transform_scheme == SubTaskConstraintCoordinationScheme.TRANSFORM:
        delta_pose = PoseUtils.get_delta_object_pose(cur_obj_pose, src_obj_pose)
        # add noise to delta pose position
    elif coord_transform_scheme == SubTaskConstraintCoordinationScheme.TRANSLATE:
        delta_pose = torch.eye(4, device=device)
        delta_pose[:3, 3] = cur_obj_pose[:3, 3] - src_obj_pose[:3, 3]
    elif coord_transform_scheme == SubTaskConstraintCoordinationScheme.REPLAY:
        delta_pose = torch.eye(4, device=device)
    else:
        raise ValueError(
            f"coordination coord_transform_scheme {coord_transform_scheme} not supported, only"
            f" {[e.value for e in SubTaskConstraintCoordinationScheme]} are supported"
        )

    pos_noise_scale = task_constraint["coordination_scheme_pos_noise_scale"]
    rot_noise_scale = task_constraint["coordination_scheme_rot_noise_scale"]
    if pos_noise_scale != 0.0 or rot_noise_scale != 0.0:
        pos = delta_pose[:3, 3]
        rot = delta_pose[:3, :3]
        pos_new, rot_new = PoseUtils.add_uniform_noise_to_pose(pos, rot, pos_noise_scale, rot_noise_scale)
        delta_pose = torch.eye(4, device=device)
        delta_pose[:3, 3] = pos_new
        delta_pose[:3, :3] = rot_new
    return delta_pose


class DataGenerator:
    """
    The main data generator class that generates new trajectories from source datasets.

    The data generator, inspired by the MimicGen, enables the generation of new datasets based on a few human
    collected source demonstrations.

    The data generator works by parsing demonstrations into object-centric subtask segments, stored in DataGenInfoPool.
    It then adapts these subtask segments to new scenes by transforming each segment according to the new scene’s context,
    stitching them into a coherent trajectory for a robotic end-effector to execute.
    """

    def __init__(
        self,
        env: ManagerBasedRLMimicEnv,
        src_demo_datagen_info_pool: DataGenInfoPool | None = None,
        dataset_path: str | None = None,
        demo_keys: list[str] | None = None,
    ):
        """
        Args:
            env: environment to use for data generation
            src_demo_datagen_info_pool: source demo datagen info pool
            dataset_path: path to hdf5 dataset to use for generation
            demo_keys: list of demonstration keys to use in file. If not provided, all demonstration keys
                will be used.
        """
        self.env = env
        self.env_cfg = env.cfg
        assert isinstance(self.env_cfg, MimicEnvCfg)
        self.dataset_path = dataset_path

        # sanity check on task spec offset ranges - final subtask should not have any offset randomization
        for subtask_configs in self.env_cfg.subtask_configs.values():
            assert subtask_configs[-1].subtask_term_offset_range[0] == 0
            assert subtask_configs[-1].subtask_term_offset_range[1] == 0

        self.demo_keys = demo_keys

        if src_demo_datagen_info_pool is not None:
            self.src_demo_datagen_info_pool = src_demo_datagen_info_pool
        elif dataset_path is not None:
            self.src_demo_datagen_info_pool = DataGenInfoPool(
                env=self.env, env_cfg=self.env_cfg, device=self.env.device
            )
            self.src_demo_datagen_info_pool.load_from_dataset_file(dataset_path, select_demo_keys=self.demo_keys)
        else:
            raise ValueError("Either src_demo_datagen_info_pool or dataset_path must be provided")

    def __repr__(self):
        """
        Pretty print this object.
        """
        msg = str(self.__class__.__name__)
        msg += " (\n\tdataset_path={}\n\tdemo_keys={}\n)".format(
            self.dataset_path,
            self.demo_keys,
        )
        return msg

    def randomize_subtask_boundaries(self) -> dict[str, np.ndarray]:
        """
        Apply random offsets to sample subtask boundaries according to the task spec.
        Recall that each demonstration is segmented into a set of subtask segments, and the
        end index of each subtask can have a random offset.
        """

        randomized_subtask_boundaries = {}

        for eef_name, subtask_boundaries in self.src_demo_datagen_info_pool.subtask_boundaries.items():
            # initial subtask start and end indices - shape (N, S, 2)
            subtask_boundaries = np.array(subtask_boundaries)

            # Randomize the start of the first subtask
            first_subtask_start_offsets = np.random.randint(
                low=self.env_cfg.subtask_configs[eef_name][0].first_subtask_start_offset_range[0],
                high=self.env_cfg.subtask_configs[eef_name][0].first_subtask_start_offset_range[0] + 1,
                size=subtask_boundaries.shape[0],
            )
            subtask_boundaries[:, 0, 0] += first_subtask_start_offsets

            # for each subtask (except last one), sample all end offsets at once for each demonstration
            # add them to subtask end indices, and then set them as the start indices of next subtask too
            for i in range(subtask_boundaries.shape[1] - 1):
                end_offsets = np.random.randint(
                    low=self.env_cfg.subtask_configs[eef_name][i].subtask_term_offset_range[0],
                    high=self.env_cfg.subtask_configs[eef_name][i].subtask_term_offset_range[1] + 1,
                    size=subtask_boundaries.shape[0],
                )
                subtask_boundaries[:, i, 1] = subtask_boundaries[:, i, 1] + end_offsets
                # don't forget to set these as start indices for next subtask too
                subtask_boundaries[:, i + 1, 0] = subtask_boundaries[:, i, 1]

            # ensure non-empty subtasks
            assert np.all((subtask_boundaries[:, :, 1] - subtask_boundaries[:, :, 0]) > 0), "got empty subtasks!"

            # ensure subtask indices increase (both starts and ends)
            assert np.all(
                (subtask_boundaries[:, 1:, :] - subtask_boundaries[:, :-1, :]) > 0
            ), "subtask indices do not strictly increase"

            # ensure subtasks are in order
            subtask_inds_flat = subtask_boundaries.reshape(subtask_boundaries.shape[0], -1)
            assert np.all((subtask_inds_flat[:, 1:] - subtask_inds_flat[:, :-1]) >= 0), "subtask indices not in order"

            randomized_subtask_boundaries[eef_name] = subtask_boundaries

        return randomized_subtask_boundaries

    def select_source_demo(
        self,
        eef_name: str,
        eef_pose: np.ndarray,
        object_pose: np.ndarray,
        src_demo_current_subtask_boundaries: np.ndarray,
        subtask_object_name: str,
        selection_strategy_name: str,
        selection_strategy_kwargs: dict | None = None,
    ) -> int:
        """
        Helper method to run source subtask segment selection.

        Args:
            eef_name: name of end effector
            eef_pose: current end effector pose
            object_pose: current object pose for this subtask
            src_demo_current_subtask_boundaries: start and end indices for subtask segment in source demonstrations of shape (N, 2)
            subtask_object_name: name of reference object for this subtask
            selection_strategy_name: name of selection strategy
            selection_strategy_kwargs: extra kwargs for running selection strategy

        Returns:
            selected_src_demo_ind: selected source demo index
        """
        if subtask_object_name is None:
            # no reference object - only random selection is supported
            assert selection_strategy_name == "random", selection_strategy_name

        # We need to collect the datagen info objects over the timesteps for the subtask segment in each source
        # demo, so that it can be used by the selection strategy.
        src_subtask_datagen_infos = []
        for i in range(len(self.src_demo_datagen_info_pool.datagen_infos)):
            # datagen info over all timesteps of the src trajectory
            src_ep_datagen_info = self.src_demo_datagen_info_pool.datagen_infos[i]

            # time indices for subtask
            subtask_start_ind = src_demo_current_subtask_boundaries[i][0]
            subtask_end_ind = src_demo_current_subtask_boundaries[i][1]

            # get subtask segment using indices
            src_subtask_datagen_infos.append(
                DatagenInfo(
                    eef_pose=src_ep_datagen_info.eef_pose[eef_name][subtask_start_ind:subtask_end_ind],
                    # only include object pose for relevant object in subtask
                    object_poses=(
                        {
                            subtask_object_name: src_ep_datagen_info.object_poses[subtask_object_name][
                                subtask_start_ind:subtask_end_ind
                            ]
                        }
                        if (subtask_object_name is not None)
                        else None
                    ),
                    # subtask termination signal is unused
                    subtask_term_signals=None,
                    target_eef_pose=src_ep_datagen_info.target_eef_pose[eef_name][subtask_start_ind:subtask_end_ind],
                    gripper_action=src_ep_datagen_info.gripper_action[eef_name][subtask_start_ind:subtask_end_ind],
                )
            )

        # make selection strategy object
        selection_strategy_obj = make_selection_strategy(selection_strategy_name)

        # run selection
        if selection_strategy_kwargs is None:
            selection_strategy_kwargs = dict()
        selected_src_demo_ind = selection_strategy_obj.select_source_demo(
            eef_pose=eef_pose,
            object_pose=object_pose,
            src_subtask_datagen_infos=src_subtask_datagen_infos,
            **selection_strategy_kwargs,
        )

        return selected_src_demo_ind

    def generate_trajectory(
        self,
        env_id: int,
        eef_name: str,
        subtask_ind: int,
        all_randomized_subtask_boundaries: dict[str, np.ndarray],
        runtime_subtask_constraints_dict: dict[tuple[str, int], dict],
        selected_src_demo_inds: dict[str, int | None],
        prev_executed_traj: dict[str, list[Waypoint] | None],
    ) -> list[Waypoint]:
        """
        Generate a trajectory for the given subtask.

        Args:
            env_id: environment index
            eef_name: name of end effector
            subtask_ind: index of subtask
            all_randomized_subtask_boundaries: randomized subtask boundaries
            runtime_subtask_constraints_dict: runtime subtask constraints
            selected_src_demo_inds: dictionary of selected source demo indices per eef, updated in place
            prev_executed_traj: dictionary of previously executed eef trajectories

        Returns:
            trajectory: generated trajectory
        """
        subtask_configs = self.env_cfg.subtask_configs[eef_name]
        # name of object for this subtask
        subtask_object_name = self.env_cfg.subtask_configs[eef_name][subtask_ind].object_ref
        subtask_object_pose = (
            self.env.get_object_poses(env_ids=[env_id])[subtask_object_name][0]
            if (subtask_object_name is not None)
            else None
        )

        is_first_subtask = subtask_ind == 0

        need_source_demo_selection = is_first_subtask or self.env_cfg.datagen_config.generation_select_src_per_subtask

        if not self.env_cfg.datagen_config.generation_select_src_per_arm:
            need_source_demo_selection = need_source_demo_selection and selected_src_demo_inds[eef_name] is None

        use_delta_transform = None
        coord_transform_scheme = None
        if (eef_name, subtask_ind) in runtime_subtask_constraints_dict:
            if runtime_subtask_constraints_dict[(eef_name, subtask_ind)]["type"] == SubTaskConstraintType.COORDINATION:
                # avoid selecting source demo if it has already been selected by the concurrent task
                concurrent_task_spec_key = runtime_subtask_constraints_dict[(eef_name, subtask_ind)][
                    "concurrent_task_spec_key"
                ]
                concurrent_subtask_ind = runtime_subtask_constraints_dict[(eef_name, subtask_ind)][
                    "concurrent_subtask_ind"
                ]
                concurrent_selected_src_ind = runtime_subtask_constraints_dict[
                    (concurrent_task_spec_key, concurrent_subtask_ind)
                ]["selected_src_demo_ind"]
                if concurrent_selected_src_ind is not None:
                    # the concurrent task has started, so we should use the same source demo
                    selected_src_demo_inds[eef_name] = concurrent_selected_src_ind
                    need_source_demo_selection = False
                    use_delta_transform = runtime_subtask_constraints_dict[
                        (concurrent_task_spec_key, concurrent_subtask_ind)
                    ]["transform"]
                else:
                    assert (
                        "transform" not in runtime_subtask_constraints_dict[(eef_name, subtask_ind)]
                    ), "transform should not be set for concurrent task"
                    # need to transform demo according to scheme
                    coord_transform_scheme = runtime_subtask_constraints_dict[(eef_name, subtask_ind)][
                        "coordination_scheme"
                    ]
                    if coord_transform_scheme != SubTaskConstraintCoordinationScheme.REPLAY:
                        assert (
                            subtask_object_name is not None
                        ), f"object reference should not be None for {coord_transform_scheme} coordination scheme"

        if need_source_demo_selection:
            selected_src_demo_inds[eef_name] = self.select_source_demo(
                eef_name=eef_name,
                eef_pose=self.env.get_robot_eef_pose(env_ids=[env_id], eef_name=eef_name)[0],
                object_pose=subtask_object_pose,
                src_demo_current_subtask_boundaries=all_randomized_subtask_boundaries[eef_name][:, subtask_ind],
                subtask_object_name=subtask_object_name,
                selection_strategy_name=self.env_cfg.subtask_configs[eef_name][subtask_ind].selection_strategy,
                selection_strategy_kwargs=self.env_cfg.subtask_configs[eef_name][subtask_ind].selection_strategy_kwargs,
            )

        assert selected_src_demo_inds[eef_name] is not None
        selected_src_demo_ind = selected_src_demo_inds[eef_name]

        if not self.env_cfg.datagen_config.generation_select_src_per_arm and need_source_demo_selection:
            for itrated_eef_name in self.env_cfg.subtask_configs.keys():
                selected_src_demo_inds[itrated_eef_name] = selected_src_demo_ind

        # selected subtask segment time indices
        selected_src_subtask_boundary = all_randomized_subtask_boundaries[eef_name][selected_src_demo_ind, subtask_ind]

        if (eef_name, subtask_ind) in runtime_subtask_constraints_dict:
            if runtime_subtask_constraints_dict[(eef_name, subtask_ind)]["type"] == SubTaskConstraintType.COORDINATION:
                # store selected source demo ind for concurrent task
                runtime_subtask_constraints_dict[(eef_name, subtask_ind)][
                    "selected_src_demo_ind"
                ] = selected_src_demo_ind
                concurrent_task_spec_key = runtime_subtask_constraints_dict[(eef_name, subtask_ind)][
                    "concurrent_task_spec_key"
                ]
                concurrent_subtask_ind = runtime_subtask_constraints_dict[(eef_name, subtask_ind)][
                    "concurrent_subtask_ind"
                ]
                concurrent_src_subtask_inds = all_randomized_subtask_boundaries[concurrent_task_spec_key][
                    selected_src_demo_ind, concurrent_subtask_ind
                ]
                subtask_len = selected_src_subtask_boundary[1] - selected_src_subtask_boundary[0]
                concurrent_subtask_len = concurrent_src_subtask_inds[1] - concurrent_src_subtask_inds[0]
                runtime_subtask_constraints_dict[(eef_name, subtask_ind)]["synchronous_steps"] = min(
                    subtask_len, concurrent_subtask_len
                )

        # TODO allow for different anchor selection strategies for each subtask

        # get subtask segment, consisting of the sequence of robot eef poses, target poses, gripper actions
        src_ep_datagen_info = self.src_demo_datagen_info_pool.datagen_infos[selected_src_demo_ind]
        src_subtask_eef_poses = src_ep_datagen_info.eef_pose[eef_name][
            selected_src_subtask_boundary[0] : selected_src_subtask_boundary[1]
        ]
        src_subtask_target_poses = src_ep_datagen_info.target_eef_pose[eef_name][
            selected_src_subtask_boundary[0] : selected_src_subtask_boundary[1]
        ]
        src_subtask_gripper_actions = src_ep_datagen_info.gripper_action[eef_name][
            selected_src_subtask_boundary[0] : selected_src_subtask_boundary[1]
        ]

        # get reference object pose from source demo
        src_subtask_object_pose = (
            src_ep_datagen_info.object_poses[subtask_object_name][selected_src_subtask_boundary[0]]
            if (subtask_object_name is not None)
            else None
        )

        if is_first_subtask or self.env_cfg.datagen_config.generation_transform_first_robot_pose:
            # Source segment consists of first robot eef pose and the target poses. This ensures that
            # we will interpolate to the first robot eef pose in this source segment, instead of the
            # first robot target pose.
            src_eef_poses = torch.cat([src_subtask_eef_poses[0:1], src_subtask_target_poses], dim=0)
            # account for extra timestep added to @src_eef_poses
            src_subtask_gripper_actions = torch.cat(
                [src_subtask_gripper_actions[0:1], src_subtask_gripper_actions], dim=0
            )
        else:
            # Source segment consists of just the target poses.
            src_eef_poses = src_subtask_target_poses.clone()
            src_subtask_gripper_actions = src_subtask_gripper_actions.clone()

        # Transform source demonstration segment using relevant object pose.
        if use_delta_transform is not None:
            # use delta transform from concurrent task
            transformed_eef_poses = transform_source_data_segment_using_delta_object_pose(
                src_eef_poses, use_delta_transform
            )

            # TODO: Uncomment below to support case of temporal concurrent but NOT does not require coordination. Need to decide if this case is necessary
            # if subtask_object_name is not None:
            #     transformed_eef_poses = PoseUtils.transform_source_data_segment_using_object_pose(
            #         cur_object_poses[task_spec_idx],
            #         src_eef_poses,
            #         src_subtask_object_pose,
            #     )

        else:
            if coord_transform_scheme is not None:
                delta_obj_pose = get_delta_pose_with_scheme(
                    src_subtask_object_pose,
                    subtask_object_pose,
                    runtime_subtask_constraints_dict[(eef_name, subtask_ind)],
                )
                transformed_eef_poses = transform_source_data_segment_using_delta_object_pose(
                    src_eef_poses, delta_obj_pose
                )
                runtime_subtask_constraints_dict[(eef_name, subtask_ind)]["transform"] = delta_obj_pose
            else:
                if subtask_object_name is not None:
                    transformed_eef_poses = transform_source_data_segment_using_object_pose(
                        subtask_object_pose,
                        src_eef_poses,
                        src_subtask_object_pose,
                    )
                else:
                    print(f"skipping transformation for {subtask_object_name}")
                    # skip transformation if no reference object is provided
                    transformed_eef_poses = src_eef_poses

        # We will construct a WaypointTrajectory instance to keep track of robot control targets
        # that will be executed and then execute it.
        traj_to_execute = WaypointTrajectory()

        if self.env_cfg.datagen_config.generation_interpolate_from_last_target_pose and (not is_first_subtask):
            # Interpolation segment will start from last target pose (which may not have been achieved).
            assert prev_executed_traj[eef_name] is not None
            last_waypoint = prev_executed_traj[eef_name][-1]
            init_sequence = WaypointSequence(sequence=[last_waypoint])
        else:
            # Interpolation segment will start from current robot eef pose.
            init_sequence = WaypointSequence.from_poses(
                poses=self.env.get_robot_eef_pose(env_ids=[env_id], eef_name=eef_name)[0].unsqueeze(0),
                gripper_actions=src_subtask_gripper_actions[0].unsqueeze(0),
                action_noise=subtask_configs[subtask_ind].action_noise,
            )
        traj_to_execute.add_waypoint_sequence(init_sequence)

        # Construct trajectory for the transformed segment.
        transformed_seq = WaypointSequence.from_poses(
            poses=transformed_eef_poses,
            gripper_actions=src_subtask_gripper_actions,
            action_noise=subtask_configs[subtask_ind].action_noise,
        )
        transformed_traj = WaypointTrajectory()
        transformed_traj.add_waypoint_sequence(transformed_seq)

        # Merge this trajectory into our trajectory using linear interpolation.
        # Interpolation will happen from the initial pose (@init_sequence) to the first element of @transformed_seq.
        traj_to_execute.merge(
            transformed_traj,
            num_steps_interp=self.env_cfg.subtask_configs[eef_name][subtask_ind].num_interpolation_steps,
            num_steps_fixed=self.env_cfg.subtask_configs[eef_name][subtask_ind].num_fixed_steps,
            action_noise=(
                float(self.env_cfg.subtask_configs[eef_name][subtask_ind].apply_noise_during_interpolation)
                * self.env_cfg.subtask_configs[eef_name][subtask_ind].action_noise
            ),
        )

        # We initialized @traj_to_execute with a pose to allow @merge to handle linear interpolation
        # for us. However, we can safely discard that first waypoint now, and just start by executing
        # the rest of the trajectory (interpolation segment and transformed subtask segment).
        traj_to_execute.pop_first()

        # Return the generated trajectory
        return traj_to_execute.get_full_sequence().sequence

    async def generate(
        self,
        env_id: int,
        success_term: TerminationTermCfg,
        env_reset_queue: asyncio.Queue | None = None,
        env_action_queue: asyncio.Queue | None = None,
        pause_subtask: bool = False,
        export_demo: bool = True,
    ) -> dict:
        """
        Attempt to generate a new demonstration.

        Args:
            env_id: environment index
            success_term: success function to check if the task is successful
            env_reset_queue: queue to store environment IDs for reset
            env_action_queue: queue to store actions for each environment
            pause_subtask: if True, pause after every subtask during generation, for debugging
            export_demo: if True, export the generated demonstration

        Returns:
            results: dictionary with the following items:
                initial_state (dict): initial simulator state for the executed trajectory
                states (list): simulator state at each timestep
                observations (list): observation dictionary at each timestep
                datagen_infos (list): datagen_info at each timestep
                actions (np.array): action executed at each timestep
                success (bool): whether the trajectory successfully solved the task or not
                src_demo_inds (list): list of selected source demonstration indices for each subtask
                src_demo_labels (np.array): same as @src_demo_inds, but repeated to have a label for each timestep of the trajectory
        """

        # reset the env to create a new task demo instance
        env_id_tensor = torch.tensor([env_id], dtype=torch.int64, device=self.env.device)
        self.env.recorder_manager.reset(env_ids=env_id_tensor)
        await env_reset_queue.put(env_id)
        await env_reset_queue.join()
        new_initial_state = self.env.scene.get_state(is_relative=True)

        # create runtime subtask constraint rules from subtask constraint configs
        runtime_subtask_constraints_dict = {}
        for subtask_constraint in self.env_cfg.task_constraint_configs:
            runtime_subtask_constraints_dict.update(subtask_constraint.generate_runtime_subtask_constraints())

        # save generated data in these variables
        generated_states = []
        generated_obs = []
        generated_actions = []
        generated_success = False

        # some eef-specific state variables used during generation
        current_eef_selected_src_demo_indices = {}
        current_eef_subtask_trajectories = {}
        current_eef_subtask_indices = {}
        current_eef_subtask_step_indices = {}
        eef_subtasks_done = {}
        for eef_name in self.env_cfg.subtask_configs.keys():
            current_eef_selected_src_demo_indices[eef_name] = None
            # prev_eef_executed_traj[eef_name] = None  # type of list of Waypoint
            current_eef_subtask_trajectories[eef_name] = None  # type of list of Waypoint
            current_eef_subtask_indices[eef_name] = 0
            current_eef_subtask_step_indices[eef_name] = None
            eef_subtasks_done[eef_name] = False

        prev_src_demo_datagen_info_pool_size = 0
        # While loop that runs per time step
        while True:
            async with self.src_demo_datagen_info_pool.asyncio_lock:
                if len(self.src_demo_datagen_info_pool.datagen_infos) > prev_src_demo_datagen_info_pool_size:
                    # src_demo_datagen_info_pool at this point may be updated with new demos,
                    # so we need to updaet subtask boundaries again
                    randomized_subtask_boundaries = (
                        self.randomize_subtask_boundaries()
                    )  # shape [N, S, 2], last dim is start and end action lengths
                    prev_src_demo_datagen_info_pool_size = len(self.src_demo_datagen_info_pool.datagen_infos)

                # generate trajectory for a subtask for the eef that is currently at the beginning of a subtask
                for eef_name, eef_subtask_step_index in current_eef_subtask_step_indices.items():
                    if eef_subtask_step_index is None:
                        # current_eef_selected_src_demo_indices will be updated in generate_trajectory
                        subtask_trajectory = self.generate_trajectory(
                            env_id,
                            eef_name,
                            current_eef_subtask_indices[eef_name],
                            randomized_subtask_boundaries,
                            runtime_subtask_constraints_dict,
                            current_eef_selected_src_demo_indices,
                            current_eef_subtask_trajectories,
                        )
                        current_eef_subtask_trajectories[eef_name] = subtask_trajectory
                        current_eef_subtask_step_indices[eef_name] = 0
                        # current_eef_selected_src_demo_indices[eef_name] = selected_src_demo_inds
                        # two_arm_trajectories[task_spec_idx] = subtask_trajectory
                        # prev_executed_traj[task_spec_idx] = subtask_trajectory

            # determine the next waypoint for each eef based on the current subtask constraints
            eef_waypoint_dict = {}
            for eef_name in sorted(self.env_cfg.subtask_configs.keys()):
                # handle constraints
                step_ind = current_eef_subtask_step_indices[eef_name]
                subtask_ind = current_eef_subtask_indices[eef_name]
                if (eef_name, subtask_ind) in runtime_subtask_constraints_dict:
                    task_constraint = runtime_subtask_constraints_dict[(eef_name, subtask_ind)]
                    if task_constraint["type"] == SubTaskConstraintType._SEQUENTIAL_LATTER:
                        min_time_diff = task_constraint["min_time_diff"]
                        if not task_constraint["fulfilled"]:
                            if (
                                min_time_diff == -1
                                or step_ind >= len(current_eef_subtask_trajectories[eef_name]) - min_time_diff
                            ):
                                if step_ind > 0:
                                    # wait at the same step
                                    step_ind -= 1
                                    current_eef_subtask_step_indices[eef_name] = step_ind

                    elif task_constraint["type"] == SubTaskConstraintType.COORDINATION:
                        synchronous_steps = task_constraint["synchronous_steps"]
                        concurrent_task_spec_key = task_constraint["concurrent_task_spec_key"]
                        concurrent_subtask_ind = task_constraint["concurrent_subtask_ind"]
                        concurrent_task_fulfilled = runtime_subtask_constraints_dict[
                            (concurrent_task_spec_key, concurrent_subtask_ind)
                        ]["fulfilled"]

                        if (
                            task_constraint["coordination_synchronize_start"]
                            and current_eef_subtask_indices[concurrent_task_spec_key] < concurrent_subtask_ind
                        ):
                            # the concurrent eef is not yet at the concurrent subtask, so wait at the first action
                            # this also makes sure that the concurrent task starts at the same time as this task
                            step_ind = 0
                            current_eef_subtask_step_indices[eef_name] = 0
                        else:
                            if (
                                not concurrent_task_fulfilled
                                and step_ind >= len(current_eef_subtask_trajectories[eef_name]) - synchronous_steps
                            ):
                                # trigger concurrent task
                                runtime_subtask_constraints_dict[(concurrent_task_spec_key, concurrent_subtask_ind)][
                                    "fulfilled"
                                ] = True

                            if not task_constraint["fulfilled"]:
                                if step_ind >= len(current_eef_subtask_trajectories[eef_name]) - synchronous_steps:
                                    if step_ind > 0:
                                        step_ind -= 1
                                        current_eef_subtask_step_indices[eef_name] = step_ind  # wait here

                waypoint = current_eef_subtask_trajectories[eef_name][step_ind]
                eef_waypoint_dict[eef_name] = waypoint
            multi_waypoint = MultiWaypoint(eef_waypoint_dict)

            # execute the next waypoints for all eefs
            exec_results = await multi_waypoint.execute(
                env=self.env,
                success_term=success_term,
                env_id=env_id,
                env_action_queue=env_action_queue,
            )

            # update execution state buffers
            if len(exec_results["states"]) > 0:
                generated_states.extend(exec_results["states"])
                generated_obs.extend(exec_results["observations"])
                generated_actions.extend(exec_results["actions"])
                generated_success = generated_success or exec_results["success"]

            for eef_name in self.env_cfg.subtask_configs.keys():
                current_eef_subtask_step_indices[eef_name] += 1
                subtask_ind = current_eef_subtask_indices[eef_name]
                if current_eef_subtask_step_indices[eef_name] == len(
                    current_eef_subtask_trajectories[eef_name]
                ):  # subtask done
                    if (eef_name, subtask_ind) in runtime_subtask_constraints_dict:
                        task_constraint = runtime_subtask_constraints_dict[(eef_name, subtask_ind)]
                        if task_constraint["type"] == SubTaskConstraintType._SEQUENTIAL_FORMER:
                            constrained_task_spec_key = task_constraint["constrained_task_spec_key"]
                            constrained_subtask_ind = task_constraint["constrained_subtask_ind"]
                            runtime_subtask_constraints_dict[(constrained_task_spec_key, constrained_subtask_ind)][
                                "fulfilled"
                            ] = True
                        elif task_constraint["type"] == SubTaskConstraintType.COORDINATION:
                            concurrent_task_spec_key = task_constraint["concurrent_task_spec_key"]
                            concurrent_subtask_ind = task_constraint["concurrent_subtask_ind"]
                            # concurrent_task_spec_idx = task_spec_keys.index(concurrent_task_spec_key)
                            task_constraint["finished"] = True
                            # check if concurrent task has been finished
                            assert (
                                runtime_subtask_constraints_dict[(concurrent_task_spec_key, concurrent_subtask_ind)][
                                    "finished"
                                ]
                                or current_eef_subtask_step_indices[concurrent_task_spec_key]
                                >= len(current_eef_subtask_trajectories[concurrent_task_spec_key]) - 1
                            )

                    if pause_subtask:
                        input(
                            f"Pausing after subtask {current_eef_subtask_indices[eef_name]} of {eef_name} execution."
                            " Press any key to continue..."
                        )
                    # This is a check to see if this arm has completed all the subtasks
                    if current_eef_subtask_indices[eef_name] == len(self.env_cfg.subtask_configs[eef_name]) - 1:
                        eef_subtasks_done[eef_name] = True
                        # If all subtasks done for this arm, repeat last waypoint to make sure this arm does not move
                        current_eef_subtask_trajectories[eef_name].append(
                            current_eef_subtask_trajectories[eef_name][-1]
                        )
                    else:
                        current_eef_subtask_step_indices[eef_name] = None
                        current_eef_subtask_indices[eef_name] += 1
            # Check if all eef_subtasks_done values are True
            if all(eef_subtasks_done.values()):
                break

        # merge numpy arrays
        if len(generated_actions) > 0:
            generated_actions = torch.cat(generated_actions, dim=0)

        # set success to the recorded episode data and export to file
        self.env.recorder_manager.set_success_to_episodes(
            env_id_tensor, torch.tensor([[generated_success]], dtype=torch.bool, device=self.env.device)
        )
        if export_demo:
            self.env.recorder_manager.export_episodes(env_id_tensor)

        results = dict(
            initial_state=new_initial_state,
            states=generated_states,
            observations=generated_obs,
            actions=generated_actions,
            success=generated_success,
        )
        return results
