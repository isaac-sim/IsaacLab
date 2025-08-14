# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import os
import sys
import torch
import trimesh

import warp as wp

print("Python Executable:", sys.executable)
print("Python Path:", sys.path)

from scipy.stats import norm

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.mixture import GaussianMixture

base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "."))
sys.path.append(base_dir)

from isaaclab.utils.assets import retrieve_file_path

"""
Initialization / Sampling
"""


def get_prev_success_init(held_asset_pose, fixed_asset_pose, success, N, device):
    """
    Randomly selects N held_asset_pose and corresponding fixed_asset_pose
    at indices where success is 1 and returns them as torch tensors.

    Args:
        held_asset_pose (np.ndarray): Numpy array of held asset poses.
        fixed_asset_pose (np.ndarray): Numpy array of fixed asset poses.
        success (np.ndarray): Numpy array of success values (1 for success, 0 for failure).
        N (int): Number of successful indices to select.
        device: torch device.

    Returns:
        tuple: (held_asset_poses, fixed_asset_poses) as torch tensors, or None if no success found.
    """
    # Get indices where success is 1
    success_indices = np.where(success == 1)[0]

    if success_indices.size == 0:
        return None  # No successful entries found

    # Select up to N random indices from successful indices
    selected_indices = np.random.choice(success_indices, min(N, len(success_indices)), replace=False)

    return torch.tensor(held_asset_pose[selected_indices], device=device), torch.tensor(
        fixed_asset_pose[selected_indices], device=device
    )


def model_succ_w_gmm(held_asset_pose, fixed_asset_pose, success):
    """
    Models the success rate distribution as a function of the relative position between the held and fixed assets
    using a Gaussian Mixture Model (GMM).

    Parameters:
        held_asset_pose (np.ndarray): Array of shape (N, 7) representing the positions of the held asset.
        fixed_asset_pose (np.ndarray): Array of shape (N, 7) representing the positions of the fixed asset.
        success (np.ndarray): Array of shape (N, 1) representing the success.

    Returns:
        GaussianMixture: The fitted GMM.

    Example:
        gmm = model_succ_dist_w_gmm(held_asset_pose, fixed_asset_pose, success)
        relative_pose = held_asset_pose - fixed_asset_pose
        # To compute the probability of each component for the given relative positions:
        probabilities = gmm.predict_proba(relative_pose)
    """
    # Compute the relative positions (held asset relative to fixed asset)
    relative_pos = held_asset_pose[:, :3] - fixed_asset_pose[:, :3]

    # Flatten the success array to serve as sample weights.
    # This way, samples with higher success contribute more to the model.
    sample_weights = success.flatten()

    # Initialize the Gaussian Mixture Model with the specified number of components.
    gmm = GaussianMixture(n_components=2, random_state=0)

    # Fit the GMM on the relative positions, using sample weights from the success metric.
    gmm.fit(relative_pos, sample_weight=sample_weights)

    return gmm


def sample_rel_pos_from_gmm(gmm, batch_size, device):
    """
    Samples a batch of relative poses (held_asset relative to fixed_asset)
    from a fitted GaussianMixture model.

    Parameters:
        gmm (GaussianMixture): A GaussianMixture model fitted on relative pose data.
        batch_size (int): The number of samples to generate.

    Returns:
        torch.Tensor: A tensor of shape (batch_size, 3) containing the sampled relative poses.
    """
    # Sample batch_size samples from the Gaussian Mixture Model.
    samples, _ = gmm.sample(batch_size)

    # Convert the numpy array to a torch tensor.
    samples_tensor = torch.from_numpy(samples).to(device)

    return samples_tensor


def model_succ_w_gp(held_asset_pose, fixed_asset_pose, success):
    """
    Models the success rate distribution given the relative position of the held asset
    from the fixed asset using a Gaussian Process classifier.

    Parameters:
        held_asset_pose (np.ndarray): Array of shape (N, 7) representing the held asset pose.
                                      Assumes the first 3 columns are the (x, y, z) positions.
        fixed_asset_pose (np.ndarray): Array of shape (N, 7) representing the fixed asset pose.
                                      Assumes the first 3 columns are the (x, y, z) positions.
        success (np.ndarray): Array of shape (N, 1) representing the success outcome (e.g., 0 for failure,
                              1 for success).

    Returns:
        GaussianProcessClassifier: A trained GP classifier that models the success rate.
    """
    # Compute the relative position (using only the translation components)
    relative_position = held_asset_pose[:, :3] - fixed_asset_pose[:, :3]

    # Flatten success array from (N, 1) to (N,)
    y = success.ravel()

    # Create and fit the Gaussian Process Classifier
    # gp = GaussianProcessClassifier(kernel=kernel, random_state=42)
    gp = GaussianProcessRegressor(random_state=42)
    gp.fit(relative_position, y)

    return gp


def propose_failure_samples_batch_from_gp(
    gp_model, candidate_points, batch_size, device, method="ucb", kappa=2.0, xi=0.01
):
    """
    Proposes a batch of candidate samples from failure-prone regions using one of three acquisition functions:
    'ucb' (Upper Confidence Bound), 'pi' (Probability of Improvement), or 'ei' (Expected Improvement).

    In this formulation, lower predicted success probability (closer to 0) is desired,
    so we invert the typical acquisition formulations.

    Parameters:
        gp_model: A trained Gaussian Process model (e.g., GaussianProcessRegressor) that supports
                  predictions with uncertainties via the 'predict' method (with return_std=True).
        candidate_points (np.ndarray): Array of shape (n_candidates, d) representing candidate relative positions.
        batch_size (int): Number of candidate samples to propose.
        method (str): Acquisition function to use: 'ucb', 'pi', or 'ei'. Default is 'ucb'.
        kappa (float): Exploration parameter for UCB. Default is 2.0.
        xi (float): Exploration parameter for PI and EI. Default is 0.01.

    Returns:
        best_candidates (np.ndarray): Array of shape (batch_size, d) containing the selected candidate points.
        acquisition (np.ndarray): Acquisition values computed for each candidate point.
    """
    # Obtain the predictive mean and standard deviation for each candidate point.
    mu, sigma = gp_model.predict(candidate_points, return_std=True)
    # mu, sigma = gp_model.predict(candidate_points)

    # Compute the acquisition values based on the chosen method.
    if method.lower() == "ucb":
        # Inversion: we want low success (i.e. low mu) and high uncertainty (sigma) to be attractive.
        acquisition = kappa * sigma - mu
    elif method.lower() == "pi":
        # Probability of Improvement: likelihood of the prediction falling below the target=0.0.
        Z = (-mu - xi) / (sigma + 1e-9)
        acquisition = norm.cdf(Z)
    elif method.lower() == "ei":
        # Expected Improvement
        Z = (-mu - xi) / (sigma + 1e-9)
        acquisition = (-mu - xi) * norm.cdf(Z) + sigma * norm.pdf(Z)
        # Set acquisition to 0 where sigma is nearly zero.
        acquisition[sigma < 1e-9] = 0.0
    else:
        raise ValueError("Unknown acquisition method. Please choose 'ucb', 'pi', or 'ei'.")

    # Select the indices of the top batch_size candidates (highest acquisition values).
    sorted_indices = np.argsort(acquisition)[::-1]  # sort in descending order
    best_indices = sorted_indices[:batch_size]
    best_candidates = candidate_points[best_indices]

    # Convert the numpy array to a torch tensor.
    best_candidates_tensor = torch.from_numpy(best_candidates).to(device)

    return best_candidates_tensor, acquisition


def propose_success_samples_batch_from_gp(
    gp_model, candidate_points, batch_size, device, method="ucb", kappa=2.0, xi=0.01
):
    """
    Proposes a batch of candidate samples from high success rate regions using one of three acquisition functions:
    'ucb' (Upper Confidence Bound), 'pi' (Probability of Improvement), or 'ei' (Expected Improvement).

    In this formulation, higher predicted success probability is desired.
    The GP model is assumed to provide predictions with uncertainties via its 'predict' method (using return_std=True).

    Parameters:
        gp_model: A trained Gaussian Process model (e.g., GaussianProcessRegressor) that supports
                  predictions with uncertainties.
        candidate_points (np.ndarray): Array of shape (n_candidates, d) representing candidate relative positions.
        batch_size (int): Number of candidate samples to propose.
        method (str): Acquisition function to use: 'ucb', 'pi', or 'ei'. Default is 'ucb'.
        kappa (float): Exploration parameter for UCB. Default is 2.0.
        xi (float): Exploration parameter for PI and EI. Default is 0.01.

    Returns:
        best_candidates (np.ndarray): Array of shape (batch_size, d) containing the selected candidate points.
        acquisition (np.ndarray): Acquisition values computed for each candidate point.
    """
    # Obtain the predictive mean and standard deviation for each candidate point.
    mu, sigma = gp_model.predict(candidate_points, return_std=True)

    # Compute the acquisition values based on the chosen method.
    if method.lower() == "ucb":
        # For maximization, UCB is defined as μ + kappa * σ.
        acquisition = mu + kappa * sigma
    elif method.lower() == "pi":
        # Probability of Improvement (maximization formulation).
        Z = (mu - 1.0 - xi) / (sigma + 1e-9)
        acquisition = norm.cdf(Z)
    elif method.lower() == "ei":
        # Expected Improvement (maximization formulation).
        Z = (mu - 1.0 - xi) / (sigma + 1e-9)
        acquisition = (mu - 1.0 - xi) * norm.cdf(Z) + sigma * norm.pdf(Z)
        # Handle nearly zero sigma values.
        acquisition[sigma < 1e-9] = 0.0
    else:
        raise ValueError("Unknown acquisition method. Please choose 'ucb', 'pi', or 'ei'.")

    # Sort candidates by acquisition value in descending order and select the top batch_size.
    sorted_indices = np.argsort(acquisition)[::-1]
    best_indices = sorted_indices[:batch_size]
    best_candidates = candidate_points[best_indices]

    # Convert the numpy array to a torch tensor.
    best_candidates_tensor = torch.from_numpy(best_candidates).to(device)

    return best_candidates_tensor, acquisition


"""
Util Functions
"""


def get_gripper_open_width(obj_filepath):

    retrieve_file_path(obj_filepath, download_dir="./")
    obj_mesh = trimesh.load_mesh(os.path.basename(obj_filepath))
    # obj_mesh = trimesh.load_mesh(obj_filepath)
    aabb = obj_mesh.bounds

    return min(0.04, (aabb[1][1] - aabb[0][1]) / 1.25)


"""
Imitation Reward
"""


def get_closest_state_idx(ref_traj, curr_ee_pos):
    """Find the index of the closest state in reference trajectory."""

    # ref_traj.shape = (num_trajs, traj_len, 3)
    traj_len = ref_traj.shape[1]
    num_envs = curr_ee_pos.shape[0]

    # dist_from_all_state.shape = (num_envs, num_trajs, traj_len, 1)
    dist_from_all_state = torch.cdist(ref_traj.unsqueeze(0), curr_ee_pos.reshape(-1, 1, 1, 3), p=2)

    # dist_from_all_state_flatten.shape = (num_envs, num_trajs * traj_len)
    dist_from_all_state_flatten = dist_from_all_state.reshape(num_envs, -1)

    # min_dist_per_env.shape = (num_envs)
    min_dist_per_env = torch.amin(dist_from_all_state_flatten, dim=-1)

    # min_dist_idx.shape = (num_envs)
    min_dist_idx = torch.argmin(dist_from_all_state_flatten, dim=-1)

    # min_dist_traj_idx.shape = (num_envs)
    # min_dist_step_idx.shape = (num_envs)
    min_dist_traj_idx = min_dist_idx // traj_len
    min_dist_step_idx = min_dist_idx % traj_len

    return min_dist_traj_idx, min_dist_step_idx, min_dist_per_env


def get_reward_mask(ref_traj, curr_ee_pos, tolerance):

    _, min_dist_step_idx, _ = get_closest_state_idx(ref_traj, curr_ee_pos)
    selected_steps = torch.index_select(
        ref_traj, dim=1, index=min_dist_step_idx
    )  # selected_steps.shape = (num_trajs, num_envs, 3)

    x_min = torch.amin(selected_steps[:, :, 0], dim=0) - tolerance
    x_max = torch.amax(selected_steps[:, :, 0], dim=0) + tolerance
    y_min = torch.amin(selected_steps[:, :, 1], dim=0) - tolerance
    y_max = torch.amax(selected_steps[:, :, 1], dim=0) + tolerance

    x_in_range = torch.logical_and(torch.lt(curr_ee_pos[:, 0], x_max), torch.gt(curr_ee_pos[:, 0], x_min))
    y_in_range = torch.logical_and(torch.lt(curr_ee_pos[:, 1], y_max), torch.gt(curr_ee_pos[:, 1], y_min))
    pos_in_range = torch.logical_and(x_in_range, y_in_range).int()

    return pos_in_range


def get_imitation_reward_from_dtw(ref_traj, curr_ee_pos, prev_ee_traj, criterion, device):
    """Get imitation reward based on dynamic time warping."""

    soft_dtw = torch.zeros((curr_ee_pos.shape[0]), device=device)
    prev_ee_pos = prev_ee_traj[:, 0, :]  # select the first ee pos in robot traj
    min_dist_traj_idx, min_dist_step_idx, min_dist_per_env = get_closest_state_idx(ref_traj, prev_ee_pos)

    for i in range(curr_ee_pos.shape[0]):
        traj_idx = min_dist_traj_idx[i]
        step_idx = min_dist_step_idx[i]
        curr_ee_pos_i = curr_ee_pos[i].reshape(1, 3)

        # NOTE: in reference trajectories, larger index -> closer to goal
        traj = ref_traj[traj_idx, step_idx:, :].reshape((1, -1, 3))

        _, curr_step_idx, _ = get_closest_state_idx(traj, curr_ee_pos_i)

        if curr_step_idx == 0:
            selected_pos = ref_traj[traj_idx, step_idx, :].reshape((1, 1, 3))
            selected_traj = torch.cat([selected_pos, selected_pos], dim=1)
        else:
            selected_traj = ref_traj[traj_idx, step_idx : (curr_step_idx + step_idx), :].reshape((1, -1, 3))
        eef_traj = torch.cat((prev_ee_traj[i, 1:, :], curr_ee_pos_i)).reshape((1, -1, 3))
        soft_dtw[i] = criterion(eef_traj, selected_traj)

    w_task_progress = 1 - (min_dist_step_idx / ref_traj.shape[1])

    # imitation_rwd = torch.exp(-soft_dtw)
    imitation_rwd = 1 - torch.tanh(soft_dtw)

    return imitation_rwd * w_task_progress


"""
Sampling-Based Curriculum (SBC)
"""


def get_new_max_disp(curr_success, cfg_task, curriculum_height_bound, curriculum_height_step, curr_max_disp):
    """Update max downward displacement of plug at beginning of episode, based on success rate."""

    if curr_success > cfg_task.curriculum_success_thresh:
        # If success rate is above threshold, increase min downward displacement until max value
        new_max_disp = torch.where(
            curr_max_disp + curriculum_height_step[:, 0] < curriculum_height_bound[:, 1],
            curr_max_disp + curriculum_height_step[:, 0],
            curriculum_height_bound[:, 1],
        )
    elif curr_success < cfg_task.curriculum_failure_thresh:
        # If success rate is below threshold, decrease min downward displacement until min value
        new_max_disp = torch.where(
            curr_max_disp + curriculum_height_step[:, 1] > curriculum_height_bound[:, 0],
            curr_max_disp + curriculum_height_step[:, 1],
            curriculum_height_bound[:, 0],
        )
    else:
        # Maintain current max downward displacement
        new_max_disp = curr_max_disp

    return new_max_disp


"""
Bonus and Success Checking
"""


def check_plug_close_to_socket(keypoints_plug, keypoints_socket, dist_threshold, progress_buf):
    """Check if plug is close to socket."""

    # Compute keypoint distance between plug and socket
    keypoint_dist = torch.norm(keypoints_socket - keypoints_plug, p=2, dim=-1)

    # Check if keypoint distance is below threshold
    is_plug_close_to_socket = torch.where(
        torch.mean(keypoint_dist, dim=-1) < dist_threshold,
        torch.ones_like(progress_buf),
        torch.zeros_like(progress_buf),
    )

    return is_plug_close_to_socket


def check_plug_inserted_in_socket(
    plug_pos, socket_pos, disassembly_dist, keypoints_plug, keypoints_socket, close_error_thresh, progress_buf
):
    """Check if plug is inserted in socket."""

    # Check if plug is within threshold distance of assembled state
    is_plug_below_insertion_height = plug_pos[:, 2] < socket_pos[:, 2] + disassembly_dist
    is_plug_above_table_height = plug_pos[:, 2] > socket_pos[:, 2]

    is_plug_height_success = torch.logical_and(is_plug_below_insertion_height, is_plug_above_table_height)

    # Check if plug is close to socket
    # NOTE: This check addresses edge case where plug is within threshold distance of
    # assembled state, but plug is outside socket
    is_plug_close_to_socket = check_plug_close_to_socket(
        keypoints_plug=keypoints_plug,
        keypoints_socket=keypoints_socket,
        dist_threshold=close_error_thresh,
        progress_buf=progress_buf,
    )

    # Combine both checks
    is_plug_inserted_in_socket = torch.logical_and(is_plug_height_success, is_plug_close_to_socket)

    return is_plug_inserted_in_socket


def get_curriculum_reward_scale(curr_max_disp, curriculum_height_bound):
    """Compute reward scale for SBC."""

    # Compute difference between max downward displacement at beginning of training (easiest condition)
    # and current max downward displacement (based on current curriculum stage)
    # NOTE: This number increases as curriculum gets harder
    curr_stage_diff = curr_max_disp - curriculum_height_bound[:, 0]

    # Compute difference between max downward displacement at beginning of training (easiest condition)
    # and min downward displacement (hardest condition)
    final_stage_diff = curriculum_height_bound[:, 1] - curriculum_height_bound[:, 0]

    # Compute reward scale
    reward_scale = curr_stage_diff / final_stage_diff + 1.0

    return reward_scale.mean()


"""
Warp Kernels
"""


# Transform points from source coordinate frame to destination coordinate frame
@wp.kernel
def transform_points(src: wp.array(dtype=wp.vec3), dest: wp.array(dtype=wp.vec3), xform: wp.transform):
    tid = wp.tid()

    p = src[tid]
    m = wp.transform_point(xform, p)

    dest[tid] = m


# Return interpenetration distances between query points (e.g., plug vertices in current pose)
# and mesh surfaces (e.g., of socket mesh in current pose)
@wp.kernel
def get_interpen_dist(
    queries: wp.array(dtype=wp.vec3),
    mesh: wp.uint64,
    interpen_dists: wp.array(dtype=wp.float32),
):
    tid = wp.tid()

    # Declare arguments to wp.mesh_query_point() that will not be modified
    q = queries[tid]  # query point
    max_dist = 1.5  # max distance on mesh from query point

    # Declare arguments to wp.mesh_query_point() that will be modified
    sign = float(
        0.0
    )  # -1 if query point inside mesh; 0 if on mesh; +1 if outside mesh (NOTE: Mesh must be watertight!)
    face_idx = int(0)  # index of closest face
    face_u = float(0.0)  # barycentric u-coordinate of closest point
    face_v = float(0.0)  # barycentric v-coordinate of closest point

    # Get closest point on mesh to query point
    closest_mesh_point_exists = wp.mesh_query_point(mesh, q, max_dist, sign, face_idx, face_u, face_v)

    # If point exists within max_dist
    if closest_mesh_point_exists:
        # Get 3D position of point on mesh given face index and barycentric coordinates
        p = wp.mesh_eval_position(mesh, face_idx, face_u, face_v)

        # Get signed distance between query point and mesh point
        delta = q - p
        signed_dist = sign * wp.length(delta)

        # If signed distance is negative
        if signed_dist < 0.0:
            # Store interpenetration distance
            interpen_dists[tid] = signed_dist
