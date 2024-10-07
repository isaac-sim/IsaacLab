# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to run an environment with a pick and lift state machine.

The state machine is implemented in the kernel function `infer_state_machine`.
It uses the `warp` library to run the state machine in parallel on the GPU.

.. code-block:: bash

    ./isaaclab.sh -p source/standalone/environments/state_machine/lift_cube_sm.py --num_envs 32

"""

"""Launch Omniverse Toolkit first."""

import argparse
import io
import os

import tqdm

import cv2
from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Pick and lift state machine for lift environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument(
    "--width", type=int, default=720, help="Width of the viewport and generated images. Defaults to 1280"
)
parser.add_argument(
    "--height", type=int, default=720, help="Height of the viewport and generated images. Defaults to 720"
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
args_cli.enable_cameras = True
# args_cli.height = 720
# args_cli.width = 720
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything else."""

import gymnasium as gym
import torch
from collections.abc import Sequence

import warp as wp

from omni.isaac.lab.assets.rigid_object.rigid_object_data import RigidObjectData

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg
from omni.isaac.lab_tasks.utils.parse_cfg import parse_env_cfg
import omni.isaac.lab.sim as sim_utils
import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.markers.config import RAY_CASTER_MARKER_CFG
from omni.isaac.lab.sensors.camera import Camera, CameraCfg
from omni.isaac.core.prims import RigidPrimView
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from omni.isaac.lab.sensors.contact_sensor import ContactSensor, ContactSensorCfg

import numpy as np
import matplotlib.pyplot as plt
from source.visualization.plot import plot_pointclouds, plot_cone, plot_arrow

def save_numpy_as_mp4(array, filename, fps=20, scale=1.0, add_index_rate=-1):
    """Creates a gif given a stack of images using moviepy
    Parameters
    ----------
    filename : string
        The filename of the gif to write to
    array : array_like
        A numpy array that contains a sequence of images
    fps : int
        frames per second (default: 10)
    scale : float
        how much to rescale each image by (default: 1.0)
    """
    # ensure that the file has the .gif extension
    fname, _ = os.path.splitext(filename)
    filename = fname + '.mp4'

    # copy into the color dimension if the images are black and white
    if array.ndim == 3:
        array = array[..., np.newaxis] * np.ones(3)

    # make the moviepy clip
    clip = ImageSequenceClip(list(array), fps=fps)
    print('Writing video to {}'.format(filename))
    clip.write_videofile(filename, fps=fps, bitrate="5000k")

def get_img_from_fig(fig, dpi=180, width=300, height=300):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (width, height))

    return img
# initialize warp
wp.init()

SCALE = 1
class GripperState:
    """States for the gripper."""

    OPEN = wp.constant(1.0)
    CLOSE = wp.constant(-1.0)


class PickSmState:
    """States for the pick state machine."""

    REST = wp.constant(0)
    APPROACH_ABOVE_OBJECT = wp.constant(1)
    APPROACH_OBJECT = wp.constant(2)
    GRASP_OBJECT = wp.constant(3)
    LIFT_OBJECT = wp.constant(4)
    END = wp.constant(5)

class PushSmState:
    """States for the push state machine."""

    REST = wp.constant(0)
    APPROACH_FRONT_OBJECT = wp.constant(1)
    APPROACH_OBJECT = wp.constant(2)
    PUSH_OBJECT = wp.constant(3)
    RETREAT = wp.constant(4)
    END = wp.constant(5)
    

class PickSmWaitTime:
    """Additional wait times (in s) for states for before switching."""

    REST = wp.constant(0.1)
    APPROACH_ABOVE_OBJECT = wp.constant(0.5*SCALE)
    APPROACH_OBJECT = wp.constant(0.6*SCALE)
    GRASP_OBJECT = wp.constant(0.3*SCALE)
    LIFT_OBJECT = wp.constant(1.0*SCALE)
    END = wp.constant(0.1)

class PushSmWaitTime:
    """Additional wait times (in s) for states for before switching."""

    REST = wp.constant(0.1)
    APPROACH_FRONT_OBJECT = wp.constant(0.8*SCALE)
    APPROACH_OBJECT = wp.constant(0.8*SCALE)
    PUSH_OBJECT = wp.constant(0.8*SCALE)
    RETREAT = wp.constant(0.8*SCALE)
    END = wp.constant(0.1)

@wp.kernel
def infer_state_machine(
    dt: wp.array(dtype=float),
    sm_state: wp.array(dtype=int),
    sm_wait_time: wp.array(dtype=float),
    ee_pose: wp.array(dtype=wp.transform),
    object_pose: wp.array(dtype=wp.transform),
    des_object_pose: wp.array(dtype=wp.transform),
    des_ee_pose: wp.array(dtype=wp.transform),
    gripper_state: wp.array(dtype=float),
    offset: wp.array(dtype=wp.transform),
):
    # retrieve thread id
    tid = wp.tid()
    # retrieve state machine state
    state = sm_state[tid]
    # decide next state
    if state == PickSmState.REST:
        des_ee_pose[tid] = ee_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        # wait for a while
        if sm_wait_time[tid] >= PickSmWaitTime.REST:
            # move to next state and reset wait time
            sm_state[tid] = PickSmState.APPROACH_ABOVE_OBJECT
            sm_wait_time[tid] = 0.0
    elif state == PickSmState.APPROACH_ABOVE_OBJECT:
        des_ee_pose[tid] = wp.transform_multiply(offset[tid], object_pose[tid])
        gripper_state[tid] = GripperState.OPEN
        # TODO: error between current and desired ee pose below threshold
        # wait for a while
        if sm_wait_time[tid] >= PickSmWaitTime.APPROACH_OBJECT:
            # move to next state and reset wait time
            sm_state[tid] = PickSmState.APPROACH_OBJECT
            sm_wait_time[tid] = 0.0
    elif state == PickSmState.APPROACH_OBJECT:
        des_ee_pose[tid] = object_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        # TODO: error between current and desired ee pose below threshold
        # wait for a while
        if sm_wait_time[tid] >= PickSmWaitTime.APPROACH_OBJECT:
            # move to next state and reset wait time
            sm_state[tid] = PickSmState.GRASP_OBJECT
            sm_wait_time[tid] = 0.0
    elif state == PickSmState.GRASP_OBJECT:
        des_ee_pose[tid] = object_pose[tid]
        gripper_state[tid] = GripperState.CLOSE
        # wait for a while
        if sm_wait_time[tid] >= PickSmWaitTime.GRASP_OBJECT:
            # move to next state and reset wait time
            sm_state[tid] = PickSmState.LIFT_OBJECT
            sm_wait_time[tid] = 0.0
    elif state == PickSmState.LIFT_OBJECT:
        des_ee_pose[tid] = des_object_pose[tid]
        gripper_state[tid] = GripperState.CLOSE
        # TODO: error between current and desired ee pose below threshold
        # wait for a while
        if sm_wait_time[tid] >= PickSmWaitTime.LIFT_OBJECT:
            # move to next state and reset wait time
            sm_state[tid] = PickSmState.END
            sm_wait_time[tid] = 0.0
    # increment wait time
    sm_wait_time[tid] = sm_wait_time[tid] + dt[tid]


class PickAndLiftSm:
    """A simple state machine in a robot's task space to pick and lift an object.

    The state machine is implemented as a warp kernel. It takes in the current state of
    the robot's end-effector and the object, and outputs the desired state of the robot's
    end-effector and the gripper. The state machine is implemented as a finite state
    machine with the following states:

    1. REST: The robot is at rest.
    2. APPROACH_ABOVE_OBJECT: The robot moves above the object.
    3. APPROACH_OBJECT: The robot moves to the object.
    4. GRASP_OBJECT: The robot grasps the object.
    5. LIFT_OBJECT: The robot lifts the object to the desired pose. This is the final state.
    """

    def __init__(self, dt: float, num_envs: int, device: torch.device | str = "cpu"):
        """Initialize the state machine.

        Args:
            dt: The environment time step.
            num_envs: The number of environments to simulate.
            device: The device to run the state machine on.
        """
        # save parameters
        self.dt = float(dt)
        self.num_envs = num_envs
        self.device = device
        # initialize state machine
        self.sm_dt = torch.full((self.num_envs,), self.dt, device=self.device)
        self.sm_state = torch.full((self.num_envs,), 0, dtype=torch.int32, device=self.device)
        self.sm_wait_time = torch.zeros((self.num_envs,), device=self.device)

        # desired state
        self.des_ee_pose = torch.zeros((self.num_envs, 7), device=self.device)
        self.des_gripper_state = torch.full((self.num_envs,), 0.0, device=self.device)

        # approach above object offset
        self.offset = torch.zeros((self.num_envs, 7), device=self.device)
        self.offset[:, 2] = 0.1
        self.offset[:, -1] = 1.0  # warp expects quaternion as (x, y, z, w)

        # convert to warp
        self.sm_dt_wp = wp.from_torch(self.sm_dt, wp.float32)
        self.sm_state_wp = wp.from_torch(self.sm_state, wp.int32)
        self.sm_wait_time_wp = wp.from_torch(self.sm_wait_time, wp.float32)
        self.des_ee_pose_wp = wp.from_torch(self.des_ee_pose, wp.transform)
        self.des_gripper_state_wp = wp.from_torch(self.des_gripper_state, wp.float32)
        self.offset_wp = wp.from_torch(self.offset, wp.transform)

    def reset_idx(self, env_ids: Sequence[int] = None):
        """Reset the state machine."""
        if env_ids is None:
            env_ids = slice(None)
        self.sm_state[env_ids] = 0
        self.sm_wait_time[env_ids] = 0.0

    def compute(self, ee_pose: torch.Tensor, object_pose: torch.Tensor, des_object_pose: torch.Tensor):
        """Compute the desired state of the robot's end-effector and the gripper."""
        # convert all transformations from (w, x, y, z) to (x, y, z, w)
        ee_pose = ee_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        object_pose = object_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        des_object_pose = des_object_pose[:, [0, 1, 2, 4, 5, 6, 3]]

        # convert to warp
        ee_pose_wp = wp.from_torch(ee_pose.contiguous(), wp.transform)
        object_pose_wp = wp.from_torch(object_pose.contiguous(), wp.transform)
        des_object_pose_wp = wp.from_torch(des_object_pose.contiguous(), wp.transform)

        # run state machine
        wp.launch(
            kernel=infer_state_machine,
            dim=self.num_envs,
            inputs=[
                self.sm_dt_wp,
                self.sm_state_wp,
                self.sm_wait_time_wp,
                ee_pose_wp,
                object_pose_wp,
                des_object_pose_wp,
                self.des_ee_pose_wp,
                self.des_gripper_state_wp,
                self.offset_wp,
            ],
            device=self.device,
        )

        # convert transformations back to (w, x, y, z)
        des_ee_pose = self.des_ee_pose[:, [0, 1, 2, 6, 3, 4, 5]]
        # convert to torch
        return torch.cat([des_ee_pose, self.des_gripper_state.unsqueeze(-1)], dim=-1)

@wp.kernel
def infer_push_state_machine(
    dt: wp.array(dtype=float),
    sm_state: wp.array(dtype=int),
    sm_wait_time: wp.array(dtype=float),
    ee_pose: wp.array(dtype=wp.transform),
    object_pose: wp.array(dtype=wp.transform),
    des_object_pose: wp.array(dtype=wp.transform),
    des_ee_pose: wp.array(dtype=wp.transform),
    gripper_state: wp.array(dtype=float),
    offset: wp.array(dtype=wp.transform),
):
    # retrieve thread id
    tid = wp.tid()
    # retrieve state machine state
    state = sm_state[tid]
    # decide next state
    if state == PushSmState.REST:
        des_ee_pose[tid] = ee_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        # wait for a while
        if sm_wait_time[tid] >= PushSmWaitTime.REST:
            # move to next state and reset wait time
            sm_state[tid] = PushSmState.APPROACH_FRONT_OBJECT
            sm_wait_time[tid] = 0.0
    elif state == PushSmState.APPROACH_FRONT_OBJECT:
        des_ee_pose[tid] = wp.transform_multiply(offset[tid], object_pose[tid])
        gripper_state[tid] = GripperState.CLOSE
        if sm_wait_time[tid] >= PushSmWaitTime.APPROACH_FRONT_OBJECT:
            # move to next state and reset wait time
            sm_state[tid] = PickSmState.APPROACH_OBJECT
            sm_wait_time[tid] = 0.0
    elif state == PushSmState.APPROACH_OBJECT:
        des_ee_pose[tid] = object_pose[tid]
        gripper_state[tid] = GripperState.CLOSE
        if sm_wait_time[tid] >= PushSmWaitTime.APPROACH_OBJECT:
            # move to next state and reset wait time
            sm_state[tid] = PushSmState.PUSH_OBJECT
            sm_wait_time[tid] = 0.0
    elif state == PushSmState.PUSH_OBJECT:
        des_ee_pose[tid] = des_object_pose[tid]
        # des_ee_pose[tid][0] = des_ee_pose[tid][0] + 0.2
        gripper_state[tid] = GripperState.CLOSE
        if sm_wait_time[tid] >= PushSmWaitTime.PUSH_OBJECT:
            # move to next state and reset wait time
            sm_state[tid] = PushSmState.RETREAT
            sm_wait_time[tid] = 0.0
    elif state == PushSmState.RETREAT:
        des_ee_pose[tid] = ee_pose[tid]
        # warp doesn't support subscriptiopn
        # des_ee_pose[tid][0] -= 0.2
        des_ee_pose[tid] = wp.transform_multiply(offset[tid], object_pose[tid])
        gripper_state[tid] = GripperState.CLOSE
        if sm_wait_time[tid] >= PushSmWaitTime.RETREAT:
            # move to next state and reset wait time
            sm_state[tid] = PushSmState.END
            sm_wait_time[tid] = 0.0
    sm_wait_time[tid] = sm_wait_time[tid] + dt[tid]



class PushSm:
    """Move to the front of the cube and push it outwards in the x-direction."""
    def __init__(self, dt: float, num_envs: int, device: torch.device | str = "cpu"):
        """Initialize the state machine.

        Args:
            dt: The environment time step.
            num_envs: The number of environments to simulate.
            device: The device to run the state machine on.
        """
        # save parameters
        self.dt = float(dt)
        self.num_envs = num_envs
        self.device = device
        # initialize state machine
        self.sm_dt = torch.full((self.num_envs,), self.dt, device=self.device)
        self.sm_state = torch.full((self.num_envs,), 0, dtype=torch.int32, device=self.device)
        self.sm_wait_time = torch.zeros((self.num_envs,), device=self.device)

        # desired state
        self.des_ee_pose = torch.zeros((self.num_envs, 7), device=self.device)
        self.des_gripper_state = torch.full((self.num_envs,), 0.0, device=self.device)

        # approach in front of object offset
        self.offset = torch.zeros((self.num_envs, 7), device=self.device)
        self.offset[:, 0] = -0.1
        self.offset[:, -1] = 1.0  # warp expects quaternion as (x, y, z, w)
        # convert to warp
        self.sm_dt_wp = wp.from_torch(self.sm_dt, wp.float32)
        self.sm_state_wp = wp.from_torch(self.sm_state, wp.int32)
        self.sm_wait_time_wp = wp.from_torch(self.sm_wait_time, wp.float32)
        self.des_ee_pose_wp = wp.from_torch(self.des_ee_pose, wp.transform)
        self.des_gripper_state_wp = wp.from_torch(self.des_gripper_state, wp.float32)
        self.offset_wp = wp.from_torch(self.offset, wp.transform)

    def reset_idx(self, env_ids: Sequence[int] = None):
        """Reset the state machine."""
        if env_ids is None:
            env_ids = slice(None)
        self.sm_state[env_ids] = 0
        self.sm_wait_time[env_ids] = 0.0
    
    def compute(self, ee_pose: torch.Tensor, object_pose: torch.Tensor,  des_object_pose: torch.Tensor):
        """Compute the desired state of the robot's end-effector and the gripper."""
        # convert all transformations from (w, x, y, z) to (x, y, z, w)
        ee_pose = ee_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        object_pose = object_pose[:, [0, 1, 2, 4, 5, 6, 3]]

        des_object_pose = des_object_pose[:, [0, 1, 2, 4, 5, 6, 3]]

        # convert to warp
        ee_pose_wp = wp.from_torch(ee_pose.contiguous(), wp.transform)
        object_pose_wp = wp.from_torch(object_pose.contiguous(), wp.transform)
        des_object_pose_wp = wp.from_torch(des_object_pose.contiguous(), wp.transform)
        print(self.sm_state_wp)
        # run state machine
        wp.launch(
            kernel=infer_push_state_machine,
            dim=self.num_envs,
            inputs=[
                self.sm_dt_wp,
                self.sm_state_wp,
                self.sm_wait_time_wp,
                ee_pose_wp,
                object_pose_wp,
                des_object_pose_wp,
                self.des_ee_pose_wp,
                self.des_gripper_state_wp,
                self.offset_wp,
            ],
            device=self.device,
        )

        # convert transformations back to (w, x, y, z)
        des_ee_pose = self.des_ee_pose[:, [0, 1, 2, 6, 3, 4, 5]]
        # convert to torch
        return torch.cat([des_ee_pose, self.des_gripper_state.unsqueeze(-1)], dim=-1)

def get_cube_vertices(pos, quat,scale=0.8):
    default_l = 0.03
    vs = torch.tensor([
        [-1, 1, 1],
        [-1, 1, -1],
        [-1, -1, 1],
        [1, -1, 1],
        [1, -1, -1],
        [-1, -1, -1],
        [1, 1, -1, ],
        [1, 1, 1]
    ], device=pos.device)* default_l * scale
    vs = math_utils.transform_points(vs, pos, quat)
    return vs
    

def main():
    # parse configuration
    robot_type = "Kuka"
    plot_force = False
    if robot_type == "Franka":
        task_name = "Isaac-Lift-Cube-Franka-IK-Abs-v0"
        force_target = "panda_joint7" # panda_hand panda_joint7
        view_point = ([1, 1, 1], [0.3, 0, 0])
    elif robot_type == "Kuka":
        task_name = "Isaac-Lift-Cube-Kuka-IK-Abs-v0"
        force_target = "victor_left_arm_joint_7" # panda_hand panda_joint7
        view_point = ([2, 2, 2], [0.3, 0, 0.8])
    env_cfg: LiftEnvCfg = parse_env_cfg(
        task_name,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.seed = 1234
    # create environment
    env = gym.make(task_name, cfg=env_cfg, render_mode="rgb_array")
    env.reset()
    
    for i in range(10):
        env.render()
        
    policy_mode = "push" # push or lift

    desired_orientation    = torch.zeros((env.unwrapped.num_envs, 4), device=env.unwrapped.device)
    desired_orientation[:, 1] = 1.0
    sim_dt = env_cfg.sim.dt
    # create state machine
    if policy_mode =="lift":
        pick_sm = PickAndLiftSm(env_cfg.sim.dt * env_cfg.decimation, env.unwrapped.num_envs, env.unwrapped.device)
        # -- target object frame
        desired_position = env.unwrapped.command_manager.get_command("object_pose")[..., :3]
    elif policy_mode == "push":
        pick_sm = PushSm(env_cfg.sim.dt * env_cfg.decimation, env.unwrapped.num_envs, env.unwrapped.device)
        object_data: RigidObjectData = env.unwrapped.scene["object"].data
        object_position = object_data.root_pos_w - env.unwrapped.scene.env_origins
        desired_position = object_position.clone()
        desired_position[:, 0] = desired_position[:, 0] + 0.2
        
    env.unwrapped.viewport_camera_controller.update_view_location(view_point[0], view_point[1]) 
    frames = []
    wrenches = []
    object_contact_net_forces = []
    robot = env.unwrapped.scene["robot"]
    ee_frame_sensor = env.unwrapped.scene["ee_frame"]
    tcp_rest_position = ee_frame_sensor.data.target_pos_w[..., 0, :].clone() - env.unwrapped.scene.env_origins
    tcp_rest_orientation = ee_frame_sensor.data.target_quat_w[..., 0, :].clone()
    gripper_vel = torch.zeros(tcp_rest_position.shape[0], 1, device=tcp_rest_position.device) + 1
    if robot_type== "Franka":
        actions = torch.cat([tcp_rest_position, tcp_rest_orientation, gripper_vel], dim=-1)
    elif robot_type == "Kuka":
        actions = torch.cat([tcp_rest_position, tcp_rest_orientation], dim=-1)
    target_ids, _ = robot.find_joints(force_target)
    
    contact_sensor = env.unwrapped.scene["contact_sensor"]
    
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # step environment
            dones = env.step(actions)[-2]
            # print(actions)
            # camera.update(dt=env.sim.get_physics_dt())
            frame = env.render()
            frames.append(frame)
            wrench = robot.root_physx_view.get_link_incoming_joint_force()[:, target_ids].flatten()
            wrenches.append(wrench.cpu().numpy())
            # contact_sensor.update(sim_dt)
            obj_pos, obj_quat = contact_sensor.data.pos_w, contact_sensor.data.quat_w
            cube_vs = get_cube_vertices(obj_pos.flatten(), obj_quat.flatten())
            net_contact_forces = contact_sensor.data.net_forces_w.flatten()
            friction_data = contact_sensor.contact_physx_view.get_friction_data(sim_dt)
            contact_force_mat = contact_sensor.data.force_matrix_w
            contact_data = contact_sensor.contact_physx_view.get_contact_data(sim_dt)
            
            nforce_mag, npoint, nnormal, ndist, ncount, nstarts = contact_data
            tforce, tpoint, tcount, tstarts = friction_data
            
            nforce = nnormal * nforce_mag
            
            num_env = 2
            force_scale = 0.005
            # fig = plot_pointclouds([cube_vs], labels=["Cube"], colors=["green"])
            # fig = plot_arrow(tpoint, tforce, sizeref=force_scale, fig=fig, color="blue", name="friction")
            # fig = plot_arrow(npoint, nforce, sizeref=force_scale, fig=fig, color="red", name="normal")
            # fig.show() 
            print(net_contact_forces)
            print(contact_force_mat)
            object_contact_net_forces.append(net_contact_forces.cpu().numpy())  
            # observations
            # -- end-effector frame
            ee_frame_sensor = env.unwrapped.scene["ee_frame"]
            tcp_rest_position = ee_frame_sensor.data.target_pos_w[..., 0, :].clone() - env.unwrapped.scene.env_origins
            tcp_rest_orientation = ee_frame_sensor.data.target_quat_w[..., 0, :].clone()
            # -- object frame
            object_data: RigidObjectData = env.unwrapped.scene["object"].data
            object_position = object_data.root_pos_w - env.unwrapped.scene.env_origins


            # advance state machine
            if len(frames) > 10:
                actions = pick_sm.compute(
                    torch.cat([tcp_rest_position, tcp_rest_orientation], dim=-1),
                    torch.cat([object_position, desired_orientation], dim=-1),
                    torch.cat([desired_position, desired_orientation], dim=-1),
                )
                if robot_type == "Kuka":
                    actions = actions[..., :7]

            if pick_sm.sm_state[0] == 5:
                pick_sm.reset_idx([0])
            # reset state machine
            # if dones.any() or len(frames) > 200:
            if len(frames) > 1000:
            # if False:
                
                # render wrench plot
                if plot_force:
                    wrenches = np.array(wrenches)
                    object_contact_net_forces = np.array(object_contact_net_forces)
                    indices = np.arange(len(wrenches))+1
                    aggregate = True
                    if aggregate:
                        force = np.linalg.norm(wrenches[:, :3], axis=-1)
                        torque = np.linalg.norm(wrenches[:, 3:], axis=-1)
                        plot_target = np.concatenate([force[:, None], torque[:, None], object_contact_net_forces], axis=-1)
                        labels = ["wrist_force", "wrist_torque", "object_Fx", "object_Fy", "object_Fz"]
                    else:
                        plot_target = np.concatenate([wrenches, object_contact_net_forces], axis=-1)
                        labels = ["robot_Fx", "Fy", "Fz", "Tx", "Ty", "Tz", "object_Fx", "object_Fy", "object_Fz"]
                    max_val = np.max(plot_target, 0)
                    min_val = np.min(plot_target, 0)
                    num_plots = plot_target.shape[-1]
                    
                    wrench_frames = []
                    
                    for t in tqdm.tqdm(indices):
                        fig, axs = plt.subplots(num_plots, 1, figsize=(6, 14))
                        for i in range(num_plots):
                            axs[i].plot(indices[:t], plot_target[:t, i])
                            axs[i].set_xlim([0, len(plot_target)])
                            axs[i].set_ylim([min_val[i]-5, max_val[i]+5])
                            axs[i].set_title(labels[i])
                        fig.tight_layout()
                        wrench_frame = get_img_from_fig(fig, width=frame.shape[1]//2, height=frame.shape[0])
                        wrench_frames.append(wrench_frame)
                        plt.close()
                    # combine frames 
                    frames = np.array(frames)
                    wrench_frames = np.array(wrench_frames)
                    combined_frames = np.concatenate([frames, wrench_frames], axis=2)
                else:
                    combined_frames = frames
                save_numpy_as_mp4(np.array(combined_frames), f'cube_{policy_mode}_{robot_type}.mp4')
                frames = []
                break

    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
