# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create observation terms.

The functions can be passed to the :class:`isaaclab.managers.ObservationTermCfg` object to enable
the observation introduced by the function.
"""

from __future__ import annotations

import torch
import warp as wp
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab_experimental.managers import SceneEntityCfg
from isaaclab_experimental.envs.mdp.kernels.obs_kernels import extract_z_from_pose, make_quat_unique_1D, get_body_world_pose_flattened, project_gravity_to_body, get_joint_data_by_indices, get_joint_data_rel_by_indices, get_root_pos_w
from isaaclab_experimental.managers.manager_base import ManagerTermBase
from isaaclab_experimental.managers.manager_term_cfg import ObservationTermCfg
from isaaclab_experimental.utils.warp.utils import resolve_asset_cfg
#from isaaclab.sensors import Camera, Imu, RayCaster, RayCasterCamera, TiledCamera

if TYPE_CHECKING:
    from isaaclab_experimental.envs import ManagerBasedEnvWarp, ManagerBasedRLEnvWarp

"""
Root state.
"""

#FIXME: Doc --> torch to warp (with torch example in string)
class base_pos_z(ManagerTermBase):
    """Root height in the simulation world frame.

    This is a high performance observation term that is used to extract the z-axis of the root pose.
    The torch based implementation 
    
    """
    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnvWarp):
        super().__init__(cfg, env)
        self._z_buffer = wp.zeros((env.num_envs,), dtype=wp.float32, device=env.device)
        asset_cfg: SceneEntityCfg = resolve_asset_cfg(cfg.params, env)
        self._asset = env.scene[asset_cfg.name]
    
    def update_config(self, asset_cfg: SceneEntityCfg | None = None) -> None:
        pass

    def __call__(self, env: ManagerBasedEnvWarp, **kwargs) -> wp.array:
        # extract the used quantities (to enable type-hinting)
        wp.launch(extract_z_from_pose, dim=env.num_envs, inputs=[self._asset.data.root_pose_w, self._z_buffer])
        return self._z_buffer


def base_lin_vel(env: ManagerBasedEnvWarp, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> wp.array:
    """Root linear velocity in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_b.view(wp.float32)


def base_ang_vel(env: ManagerBasedEnvWarp, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> wp.array:
    """Root angular velocity in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_ang_vel_b.view(wp.float32)


def projected_gravity(env: ManagerBasedEnvWarp, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> wp.array:
    """Gravity projection on the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.projected_gravity_b.view(wp.float32)


class root_pos_w(ManagerTermBase):
    """Asset root position in the environment frame."""
    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnvWarp):
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = resolve_asset_cfg(cfg.params, env)
        self._asset: RigidObject = env.scene[asset_cfg.name]
        self._root_pos_w_buffer = wp.zeros((env.num_envs,), dtype=wp.float32, device=env.device)

    def update_config(self, asset_cfg: SceneEntityCfg | None = None) -> None:
        pass

    def __call__(self,
        env: ManagerBasedEnvWarp,
        **kwargs,
    ) -> wp.array:
        """Asset root position in the environment frame."""
        wp.launch(
            get_root_pos_w,
            dim=env.num_envs,
            inputs=[
                self._asset.data.root_pos_w,
                env.scene.env_origins,
                self._root_pos_w_buffer,
            ],
        )
        return self._root_pos_w_buffer.view(wp.float32)


class root_quat_w(ManagerTermBase):
    """Asset root orientation (x, y, z, w) in the environment frame.

    If :attr:`make_quat_unique` is True, then returned quaternion is made unique by ensuring
    the quaternion has non-negative real component. This is because both ``q`` and ``-q`` represent
    the same orientation.
    """
    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnvWarp):
        super().__init__(cfg, env)

        asset_cfg: SceneEntityCfg = resolve_asset_cfg(cfg.params, env)
        self._asset: Articulation = env.scene[asset_cfg.name]
        self._quat_buffer = wp.zeros((env.num_envs), dtype=wp.quatf, device=env.device)
        self._make_quat_unique = False
        self.update_config(**cfg.params)

    def update_config(self, make_quat_unique: bool = False, asset_cfg: SceneEntityCfg | None = None) -> None:
        self._make_quat_unique = make_quat_unique

    def __call__(self,
        env: ManagerBasedEnvWarp,
        **kwargs,
    ) -> wp.array:
        """Asset root orientation (x, y, z, w) in the environment frame.

        If :attr:`make_quat_unique` is True, then returned quaternion is made unique by ensuring
        the quaternion has non-negative real component. This is because both ``q`` and ``-q`` represent
        the same orientation.
        """
        # extract the used quantities (to enable type-hinting)
        quat = self._asset.data.root_quat_w
        # make the quaternion real-part positive if configured
        if self._make_quat_unique:
            wp.launch(make_quat_unique_1D, dim=env.num_envs, inputs=[quat, self._quat_buffer])
            return self._quat_buffer.view(wp.float32)
        else:
            return quat.view(wp.float32)

def root_lin_vel_w(env: ManagerBasedEnvWarp, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> wp.array:
    """Asset root linear velocity in the environment frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_w.view(wp.float32)


def root_ang_vel_w(env: ManagerBasedEnvWarp, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> wp.array:
    """Asset root angular velocity in the environment frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_ang_vel_w.view(wp.float32)


"""
Body state
"""

class body_pose_w(ManagerTermBase):
    """The flattened body poses of the asset w.r.t the env.scene.origin.

    Note: Only the bodies configured in :attr:`asset_cfg.body_ids` will have their poses returned.
    """
    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnvWarp):
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = resolve_asset_cfg(cfg.params, env)
        self._asset: Articulation = env.scene[asset_cfg.name]
        self._pose_buffer = wp.zeros((env.num_envs, len(asset_cfg.body_ids)), dtype=wp.transformf, device=env.device)
        self._body_indices = wp.array(asset_cfg.body_ids, dtype=wp.int32, device=env.device)

    def update_config(self, asset_cfg: SceneEntityCfg | None = None) -> None:
        pass

    def __call__(self,
        env: ManagerBasedEnvWarp,
        **kwargs,
    ) -> wp.array:
        """The flattened body poses of the asset w.r.t the env.scene.origin.

        Note: Only the bodies configured in :attr:`asset_cfg.body_ids` will have their poses returned.

        Args:
            env: The environment.
            asset_cfg: The SceneEntity associated with this observation.

        Returns:
            The poses of bodies in articulation [num_env, 7 * num_bodies]. Pose order is [x,y,z,qx,qy,qz,qw].
            Output is stacked horizontally per body.
        """
        # extract the used quantities (to enable type-hinting)

        # Launch kernel with indexing to only return the required bodies. This is ok since the number of bodies remains
        # constant throughout the simulation.
        wp.launch(
            get_body_world_pose_flattened,
            dim=(env.num_envs, self._body_indices.shape[0]),
            inputs=[
                self._asset.data.body_pose_w,
                env.scene.env_origins,
                self._pose_buffer,
                self._body_indices
            ],
        )
        # FIXME: Does the reshape modifies the source array? Or does it returns a view with a different stride?
        return self._pose_buffer.view(wp.float32).reshape((env.num_envs, -1))

class body_projected_gravity_b(ManagerTermBase):
    """The direction of gravity projected on to bodies of an Articulation.

    Note: Only the bodies configured in :attr:`asset_cfg.body_ids` will have their poses returned.
    """
    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnvWarp):
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = resolve_asset_cfg(cfg.params, env)
        self._asset: Articulation = env.scene[asset_cfg.name]
        self._projected_gravity_buffer = wp.zeros((env.num_envs, len(asset_cfg.body_ids)), dtype=wp.vec3f, device=env.device)
        self._body_indices = wp.array(asset_cfg.body_ids, dtype=wp.int32, device=env.device)

    def update_config(self, asset_cfg: SceneEntityCfg | None = None) -> None:
        pass

    def __call__(self,
        env: ManagerBasedEnvWarp,
        **kwargs,
    ) -> wp.array:
        """The direction of gravity projected on to bodies of an Articulation.

        Note: Only the bodies configured in :attr:`asset_cfg.body_ids` will have their poses returned.

        Args:
            env: The environment.
            asset_cfg: The Articulation associated with this observation.

        Returns:
            The unit vector direction of gravity projected onto body_name's frame. Gravity projection vector order is
            [x,y,z]. Output is stacked horizontally per body.
        """
        # extract the used quantities (to enable type-hinting)

        wp.launch(
            project_gravity_to_body,
            dim=(env.num_envs, self._body_indices.shape[0]),
            inputs=[
                self._asset.data.body_pose_w,
                self._asset.data.GRAVITY_VEC_W,
                self._projected_gravity_buffer,
                self._body_indices
            ],
        )
        # FIXME: Does the reshape modifies the source array? Or does it returns a view with a different stride?
        return self._projected_gravity_buffer.view(wp.float32).reshape((env.num_envs, -1))


"""
Joint state.
"""

class joint_pos(ManagerTermBase):
    """The joint positions of the asset."""
    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnvWarp):
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = resolve_asset_cfg(cfg.params, env)
        self._asset: Articulation = env.scene[asset_cfg.name]
        self._joint_pos_buffer = wp.zeros((env.num_envs, len(asset_cfg.joint_ids)), dtype=wp.float32, device=env.device)
        self._joint_indices = wp.array(asset_cfg.joint_ids, dtype=wp.int32, device=env.device)

    def update_config(self, asset_cfg: SceneEntityCfg | None = None) -> None:
        pass

    def __call__(self,
        env: ManagerBasedEnvWarp,
        **kwargs,
    ) -> wp.array:
        """The joint positions of the asset."""
        wp.launch(
            get_joint_data_by_indices,
            dim=(env.num_envs, self._joint_indices.shape[0]),
            inputs=[
                self._asset.data.joint_pos,
                self._joint_indices,
                self._joint_pos_buffer,
            ],
        )
        return self._joint_pos_buffer

class joint_pos_rel(ManagerTermBase):
    """The joint positions of the asset w.r.t. the default joint positions."""
    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnvWarp):
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = resolve_asset_cfg(cfg.params, env)
        self._asset: Articulation = env.scene[asset_cfg.name]
        self._joint_pos_rel_buffer = wp.zeros((env.num_envs, len(asset_cfg.joint_ids)), dtype=wp.float32, device=env.device)
        self._joint_indices = wp.array(asset_cfg.joint_ids, dtype=wp.int32, device=env.device)

    def update_config(self, asset_cfg: SceneEntityCfg | None = None) -> None:
        pass

    def __call__(self,
        env: ManagerBasedEnvWarp,
        **kwargs,
    ) -> wp.array:
        """The joint positions of the asset w.r.t. the default joint positions.

        Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
        """
        wp.launch(
            get_joint_data_rel_by_indices,
            dim=(env.num_envs, self._joint_indices.shape[0]),
            inputs=[
                self._asset.data.joint_pos,
                self._asset.data.default_joint_pos,
                self._joint_indices,
                self._joint_pos_rel_buffer
            ],
        )
        return self._joint_pos_rel_buffer

def joint_pos_limit_normalized(
    env: ManagerBasedEnvWarp, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """The joint positions of the asset normalized with the asset's joint limits.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their normalized positions returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return math_utils.scale_transform(
        asset.data.joint_pos[:, asset_cfg.joint_ids],
        asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 0],
        asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 1],
    )

class joint_vel(ManagerTermBase):
    """The joint velocities of the asset."""
    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnvWarp):
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = resolve_asset_cfg(cfg.params, env)
        self._asset: Articulation = env.scene[asset_cfg.name]
        self._joint_vel_buffer = wp.zeros((env.num_envs, len(asset_cfg.joint_ids)), dtype=wp.float32, device=env.device)
        self._joint_indices = wp.array(asset_cfg.joint_ids, dtype=wp.int32, device=env.device)

    def update_config(self, asset_cfg: SceneEntityCfg | None = None) -> None:
        pass

    def __call__(self,
        env: ManagerBasedEnvWarp,
        **kwargs,
    ) -> wp.array:
        """The joint velocities of the asset.

        Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their velocities returned.
        """
        wp.launch(
            get_joint_data_by_indices,
            dim=(env.num_envs, self._joint_indices.shape[0]),
            inputs=[
                self._asset.data.joint_vel,
                self._joint_indices,
                self._joint_vel_buffer,
            ],
        )
        return self._joint_vel_buffer

class joint_vel_rel(ManagerTermBase):
    """The joint velocities of the asset w.r.t. the default joint velocities."""
    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnvWarp):
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = resolve_asset_cfg(cfg.params, env)
        self._asset: Articulation = env.scene[asset_cfg.name]
        self._joint_vel_rel_buffer = wp.zeros((env.num_envs, len(asset_cfg.joint_ids)), dtype=wp.float32, device=env.device)
        self._joint_indices = wp.array(asset_cfg.joint_ids, dtype=wp.int32, device=env.device)

    def update_config(self, asset_cfg: SceneEntityCfg | None = None) -> None:
        pass

    def __call__(self,
        env: ManagerBasedEnvWarp,
        **kwargs,
    ) -> wp.array:
        """The joint velocities of the asset w.r.t. the default joint velocities.

        Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their velocities returned.
        """
        wp.launch(
            get_joint_data_rel_by_indices,
            dim=(env.num_envs, self._joint_indices.shape[0]),
            inputs=[
                self._asset.data.joint_vel,
                self._asset.data.default_joint_vel,
                self._joint_indices,
                self._joint_vel_rel_buffer
            ],
        )
        return self._joint_vel_rel_buffer


class joint_effort(ManagerTermBase):
    """The joint applied effort of the asset."""
    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnvWarp):
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = resolve_asset_cfg(cfg.params, env)
        self._asset: Articulation = env.scene[asset_cfg.name]
        self._joint_effort_buffer = wp.zeros((env.num_envs, len(asset_cfg.joint_ids)), dtype=wp.float32, device=env.device)
        self._joint_indices = wp.array(asset_cfg.joint_ids, dtype=wp.int32, device=env.device)

    def update_config(self, asset_cfg: SceneEntityCfg | None = None) -> None:
        pass

    def __call__(self,
        env: ManagerBasedEnvWarp,
        **kwargs,
    ) -> wp.array:
        """The joint applied effort of the asset."""
        wp.launch(
            get_joint_data_by_indices,
            dim=(env.num_envs, self._joint_indices.shape[0]),
            inputs=[
                self._asset.data.applied_torque,
                self._joint_indices,
                self._joint_effort_buffer,
            ],
        )
        return self._joint_effort_buffer


"""
Sensors.
"""


#def height_scan(env: ManagerBasedEnvWarp, sensor_cfg: SceneEntityCfg, offset: float = 0.5) -> torch.Tensor:
#    """Height scan from the given sensor w.r.t. the sensor's frame.
#
#    The provided offset (Defaults to 0.5) is subtracted from the returned values.
#    """
#    # extract the used quantities (to enable type-hinting)
#    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
#    # height scan: height = sensor_height - hit_point_z - offset
#    return sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2] - offset
#
#
#def body_incoming_wrench(env: ManagerBasedEnvWarp, asset_cfg: SceneEntityCfg) -> torch.Tensor:
#    """Incoming spatial wrench on bodies of an articulation in the simulation world frame.
#
#    This is the 6-D wrench (force and torque) applied to the body link by the incoming joint force.
#    """
#    # extract the used quantities (to enable type-hinting)
#    asset: Articulation = env.scene[asset_cfg.name]
#    # obtain the link incoming forces in world frame
#    body_incoming_joint_wrench_b = asset.data.body_incoming_joint_wrench_b[:, asset_cfg.body_ids]
#    return body_incoming_joint_wrench_b.view(env.num_envs, -1)
#
#
#def imu_orientation(env: ManagerBasedEnvWarp, asset_cfg: SceneEntityCfg = SceneEntityCfg("imu")) -> torch.Tensor:
#    """Imu sensor orientation in the simulation world frame.
#
#    Args:
#        env: The environment.
#        asset_cfg: The SceneEntity associated with an IMU sensor. Defaults to SceneEntityCfg("imu").
#
#    Returns:
#        Orientation in the world frame in (w, x, y, z) quaternion form. Shape is (num_envs, 4).
#    """
#    # extract the used quantities (to enable type-hinting)
#    asset: Imu = env.scene[asset_cfg.name]
#    # return the orientation quaternion
#    return asset.data.quat_w
#
#
#def imu_projected_gravity(env: ManagerBasedEnvWarp, asset_cfg: SceneEntityCfg = SceneEntityCfg("imu")) -> torch.Tensor:
#    """Imu sensor orientation w.r.t the env.scene.origin.
#
#    Args:
#        env: The environment.
#        asset_cfg: The SceneEntity associated with an Imu sensor.
#
#    Returns:
#        Gravity projected on imu_frame, shape of torch.tensor is (num_env,3).
#    """
#
#    asset: Imu = env.scene[asset_cfg.name]
#    return asset.data.projected_gravity_b
#
#
#def imu_ang_vel(env: ManagerBasedEnvWarp, asset_cfg: SceneEntityCfg = SceneEntityCfg("imu")) -> torch.Tensor:
#    """Imu sensor angular velocity w.r.t. environment origin expressed in the sensor frame.
#
#    Args:
#        env: The environment.
#        asset_cfg: The SceneEntity associated with an IMU sensor. Defaults to SceneEntityCfg("imu").
#
#    Returns:
#        The angular velocity (rad/s) in the sensor frame. Shape is (num_envs, 3).
#    """
#    # extract the used quantities (to enable type-hinting)
#    asset: Imu = env.scene[asset_cfg.name]
#    # return the angular velocity
#    return asset.data.ang_vel_b
#
#
#def imu_lin_acc(env: ManagerBasedEnvWarp, asset_cfg: SceneEntityCfg = SceneEntityCfg("imu")) -> torch.Tensor:
#    """Imu sensor linear acceleration w.r.t. the environment origin expressed in sensor frame.
#
#    Args:
#        env: The environment.
#        asset_cfg: The SceneEntity associated with an IMU sensor. Defaults to SceneEntityCfg("imu").
#
#    Returns:
#        The linear acceleration (m/s^2) in the sensor frame. Shape is (num_envs, 3).
#    """
#    asset: Imu = env.scene[asset_cfg.name]
#    return asset.data.lin_acc_b
#
#
#def image(
#    env: ManagerBasedEnvWarp,
#    sensor_cfg: SceneEntityCfg = SceneEntityCfg("tiled_camera"),
#    data_type: str = "rgb",
#    convert_perspective_to_orthogonal: bool = False,
#    normalize: bool = True,
#) -> torch.Tensor:
#    """Images of a specific datatype from the camera sensor.
#
#    If the flag :attr:`normalize` is True, post-processing of the images are performed based on their
#    data-types:
#
#    - "rgb": Scales the image to (0, 1) and subtracts with the mean of the current image batch.
#    - "depth" or "distance_to_camera" or "distance_to_plane": Replaces infinity values with zero.
#
#    Args:
#        env: The environment the cameras are placed within.
#        sensor_cfg: The desired sensor to read from. Defaults to SceneEntityCfg("tiled_camera").
#        data_type: The data type to pull from the desired camera. Defaults to "rgb".
#        convert_perspective_to_orthogonal: Whether to orthogonalize perspective depth images.
#            This is used only when the data type is "distance_to_camera". Defaults to False.
#        normalize: Whether to normalize the images. This depends on the selected data type.
#            Defaults to True.
#
#    Returns:
#        The images produced at the last time-step
#    """
#    # extract the used quantities (to enable type-hinting)
#    sensor: TiledCamera | Camera | RayCasterCamera = env.scene.sensors[sensor_cfg.name]
#
#    # obtain the input image
#    images = sensor.data.output[data_type]
#
#    # depth image conversion
#    if (data_type == "distance_to_camera") and convert_perspective_to_orthogonal:
#        images = math_utils.orthogonalize_perspective_depth(images, sensor.data.intrinsic_matrices)
#
#    # rgb/depth/normals image normalization
#    if normalize:
#        if data_type == "rgb":
#            images = images.float() / 255.0
#            mean_tensor = torch.mean(images, dim=(1, 2), keepdim=True)
#            images -= mean_tensor
#        elif "distance_to" in data_type or "depth" in data_type:
#            images[images == float("inf")] = 0
#        elif "normals" in data_type:
#            images = (images + 1.0) * 0.5
#
#    return images.clone()
#
#
#class image_features(ManagerTermBase):
#    """Extracted image features from a pre-trained frozen encoder.
#
#    This term uses models from the model zoo in PyTorch and extracts features from the images.
#
#    It calls the :func:`image` function to get the images and then processes them using the model zoo.
#
#    A user can provide their own model zoo configuration to use different models for feature extraction.
#    The model zoo configuration should be a dictionary that maps different model names to a dictionary
#    that defines the model, preprocess and inference functions. The dictionary should have the following
#    entries:
#
#    - "model": A callable that returns the model when invoked without arguments.
#    - "reset": A callable that resets the model. This is useful when the model has a state that needs to be reset.
#    - "inference": A callable that, when given the model and the images, returns the extracted features.
#
#    If the model zoo configuration is not provided, the default model zoo configurations are used. The default
#    model zoo configurations include the models from Theia :cite:`shang2024theia` and ResNet :cite:`he2016deep`.
#    These models are loaded from `Hugging-Face transformers <https://huggingface.co/docs/transformers/index>`_ and
#    `PyTorch torchvision <https://pytorch.org/vision/stable/models.html>`_ respectively.
#
#    Args:
#        sensor_cfg: The sensor configuration to poll. Defaults to SceneEntityCfg("tiled_camera").
#        data_type: The sensor data type. Defaults to "rgb".
#        convert_perspective_to_orthogonal: Whether to orthogonalize perspective depth images.
#            This is used only when the data type is "distance_to_camera". Defaults to False.
#        model_zoo_cfg: A user-defined dictionary that maps different model names to their respective configurations.
#            Defaults to None. If None, the default model zoo configurations are used.
#        model_name: The name of the model to use for inference. Defaults to "resnet18".
#        model_device: The device to store and infer the model on. This is useful when offloading the computation
#            from the environment simulation device. Defaults to the environment device.
#        inference_kwargs: Additional keyword arguments to pass to the inference function. Defaults to None,
#            which means no additional arguments are passed.
#
#    Returns:
#        The extracted features tensor. Shape is (num_envs, feature_dim).
#
#    Raises:
#        ValueError: When the model name is not found in the provided model zoo configuration.
#        ValueError: When the model name is not found in the default model zoo configuration.
#    """
#
#    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnvWarp):
#        # initialize the base class
#        super().__init__(cfg, env)
#
#        # extract parameters from the configuration
#        self.model_zoo_cfg: dict = cfg.params.get("model_zoo_cfg")  # type: ignore
#        self.model_name: str = cfg.params.get("model_name", "resnet18")  # type: ignore
#        self.model_device: str = cfg.params.get("model_device", env.device)  # type: ignore
#
#        # List of Theia models - These are configured through `_prepare_theia_transformer_model` function
#        default_theia_models = [
#            "theia-tiny-patch16-224-cddsv",
#            "theia-tiny-patch16-224-cdiv",
#            "theia-small-patch16-224-cdiv",
#            "theia-base-patch16-224-cdiv",
#            "theia-small-patch16-224-cddsv",
#            "theia-base-patch16-224-cddsv",
#        ]
#        # List of ResNet models - These are configured through `_prepare_resnet_model` function
#        default_resnet_models = ["resnet18", "resnet34", "resnet50", "resnet101"]
#
#        # Check if model name is specified in the model zoo configuration
#        if self.model_zoo_cfg is not None and self.model_name not in self.model_zoo_cfg:
#            raise ValueError(
#                f"Model name '{self.model_name}' not found in the provided model zoo configuration."
#                " Please add the model to the model zoo configuration or use a different model name."
#                f" Available models in the provided list: {list(self.model_zoo_cfg.keys())}."
#                "\nHint: If you want to use a default model, consider using one of the following models:"
#                f" {default_theia_models + default_resnet_models}. In this case, you can remove the"
#                " 'model_zoo_cfg' parameter from the observation term configuration."
#            )
#        if self.model_zoo_cfg is None:
#            if self.model_name in default_theia_models:
#                model_config = self._prepare_theia_transformer_model(self.model_name, self.model_device)
#            elif self.model_name in default_resnet_models:
#                model_config = self._prepare_resnet_model(self.model_name, self.model_device)
#            else:
#                raise ValueError(
#                    f"Model name '{self.model_name}' not found in the default model zoo configuration."
#                    f" Available models: {default_theia_models + default_resnet_models}."
#                )
#        else:
#            model_config = self.model_zoo_cfg[self.model_name]
#
#        # Retrieve the model, preprocess and inference functions
#        self._model = model_config["model"]()
#        self._reset_fn = model_config.get("reset")
#        self._inference_fn = model_config["inference"]
#
#    def reset(self, env_ids: torch.Tensor | None = None):
#        # reset the model if a reset function is provided
#        # this might be useful when the model has a state that needs to be reset
#        # for example: video transformers
#        if self._reset_fn is not None:
#            self._reset_fn(self._model, env_ids)
#
#    def __call__(
#        self,
#        env: ManagerBasedEnvWarp,
#        sensor_cfg: SceneEntityCfg = SceneEntityCfg("tiled_camera"),
#        data_type: str = "rgb",
#        convert_perspective_to_orthogonal: bool = False,
#        model_zoo_cfg: dict | None = None,
#        model_name: str = "resnet18",
#        model_device: str | None = None,
#        inference_kwargs: dict | None = None,
#    ) -> torch.Tensor:
#        # obtain the images from the sensor
#        image_data = image(
#            env=env,
#            sensor_cfg=sensor_cfg,
#            data_type=data_type,
#            convert_perspective_to_orthogonal=convert_perspective_to_orthogonal,
#            normalize=False,  # we pre-process based on model
#        )
#        # store the device of the image
#        image_device = image_data.device
#        # forward the images through the model
#        features = self._inference_fn(self._model, image_data, **(inference_kwargs or {}))
#
#        # move the features back to the image device
#        return features.detach().to(image_device)
#
#    """
#    Helper functions.
#    """
#
#    def _prepare_theia_transformer_model(self, model_name: str, model_device: str) -> dict:
#        """Prepare the Theia transformer model for inference.
#
#        Args:
#            model_name: The name of the Theia transformer model to prepare.
#            model_device: The device to store and infer the model on.
#
#        Returns:
#            A dictionary containing the model and inference functions.
#        """
#        from transformers import AutoModel
#
#        def _load_model() -> torch.nn.Module:
#            """Load the Theia transformer model."""
#            model = AutoModel.from_pretrained(f"theaiinstitute/{model_name}", trust_remote_code=True).eval()
#            return model.to(model_device)
#
#        def _inference(model, images: torch.Tensor) -> torch.Tensor:
#            """Inference the Theia transformer model.
#
#            Args:
#                model: The Theia transformer model.
#                images: The preprocessed image tensor. Shape is (num_envs, height, width, channel).
#
#            Returns:
#                The extracted features tensor. Shape is (num_envs, feature_dim).
#            """
#            # Move the image to the model device
#            image_proc = images.to(model_device)
#            # permute the image to (num_envs, channel, height, width)
#            image_proc = image_proc.permute(0, 3, 1, 2).float() / 255.0
#            # Normalize the image
#            mean = torch.tensor([0.485, 0.456, 0.406], device=model_device).view(1, 3, 1, 1)
#            std = torch.tensor([0.229, 0.224, 0.225], device=model_device).view(1, 3, 1, 1)
#            image_proc = (image_proc - mean) / std
#
#            # Taken from Transformers; inference converted to be GPU only
#            features = model.backbone.model(pixel_values=image_proc, interpolate_pos_encoding=True)
#            return features.last_hidden_state[:, 1:]
#
#        # return the model, preprocess and inference functions
#        return {"model": _load_model, "inference": _inference}
#
#    def _prepare_resnet_model(self, model_name: str, model_device: str) -> dict:
#        """Prepare the ResNet model for inference.
#
#        Args:
#            model_name: The name of the ResNet model to prepare.
#            model_device: The device to store and infer the model on.
#
#        Returns:
#            A dictionary containing the model and inference functions.
#        """
#        from torchvision import models
#
#        def _load_model() -> torch.nn.Module:
#            """Load the ResNet model."""
#            # map the model name to the weights
#            resnet_weights = {
#                "resnet18": "ResNet18_Weights.IMAGENET1K_V1",
#                "resnet34": "ResNet34_Weights.IMAGENET1K_V1",
#                "resnet50": "ResNet50_Weights.IMAGENET1K_V1",
#                "resnet101": "ResNet101_Weights.IMAGENET1K_V1",
#            }
#
#            # load the model
#            model = getattr(models, model_name)(weights=resnet_weights[model_name]).eval()
#            return model.to(model_device)
#
#        def _inference(model, images: torch.Tensor) -> torch.Tensor:
#            """Inference the ResNet model.
#
#            Args:
#                model: The ResNet model.
#                images: The preprocessed image tensor. Shape is (num_envs, channel, height, width).
#
#            Returns:
#                The extracted features tensor. Shape is (num_envs, feature_dim).
#            """
#            # move the image to the model device
#            image_proc = images.to(model_device)
#            # permute the image to (num_envs, channel, height, width)
#            image_proc = image_proc.permute(0, 3, 1, 2).float() / 255.0
#            # normalize the image
#            mean = torch.tensor([0.485, 0.456, 0.406], device=model_device).view(1, 3, 1, 1)
#            std = torch.tensor([0.229, 0.224, 0.225], device=model_device).view(1, 3, 1, 1)
#            image_proc = (image_proc - mean) / std
#
#            # forward the image through the model
#            return model(image_proc)
#
#        # return the model, preprocess and inference functions
#        return {"model": _load_model, "inference": _inference}


"""
Actions.
"""


def last_action(env: ManagerBasedEnvWarp, action_name: str | None = None) -> wp.array:
    """The last input action to the environment.

    The name of the action term for which the action is required. If None, the
    entire action tensor is returned.
    """
    if action_name is None:
        return env.action_manager.action
    else:
        return env.action_manager.get_term(action_name).raw_actions


"""
Commands.
"""


def generated_commands(env: ManagerBasedRLEnvWarp, command_name: str | None = None) -> wp.array:
    """The generated command from command term in the command manager with the given name."""
    return env.command_manager.get_command(command_name)


"""
Time.
"""


def current_time_s(env: ManagerBasedRLEnvWarp) -> wp.array:
    """The current time in the episode (in seconds)."""
    return env.episode_length_buf.reshape(-1, 1) * env.step_dt


def remaining_time_s(env: ManagerBasedRLEnvWarp) -> wp.array:
    """The maximum time remaining in the episode (in seconds)."""
    return env.max_episode_length_s - env.episode_length_buf.reshape(-1, 1) * env.step_dt
