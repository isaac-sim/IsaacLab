# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Runtime fixtures for LEAPP; camera pixels are injected, not rendered."""

import torch
import warp as wp
from isaaclab_newton.sensors.contact_sensor.contact_sensor_data import ContactSensorData
from isaaclab_newton.sensors.joint_wrench.joint_wrench_sensor_data import JointWrenchSensorData
from isaaclab_physx.assets.deformable_object.deformable_object_data import DeformableObjectData

from isaaclab.assets.cable_object.base_cable_object_data import BaseCableObjectData
from isaaclab.managers import ActionTerm
from isaaclab.sensors import SensorBase
from isaaclab.sensors.ray_caster.multi_mesh_ray_caster_camera_data import MultiMeshRayCasterCameraData
from isaaclab.sensors.ray_caster.multi_mesh_ray_caster_data import MultiMeshRayCasterData
from isaaclab.utils.warp import ProxyArray

from isaaclab_contrib.sensors.tacsl_sensor.visuotactile_sensor_data import VisuoTactileSensorData


class InjectedCamera(SensorBase):
    """Supply changing RGB and depth buffers through the real sensor lifecycle.

    This covers camera preprocessing and export wiring, not image rendering. A spatially varying
    pattern ensures RGB mean subtraction does not cancel changes between frames.
    """

    @property
    def data(self) -> MultiMeshRayCasterCameraData:
        self._update_outdated_buffers()
        return self._data

    def _initialize_impl(self):
        super()._initialize_impl()
        self._data = MultiMeshRayCasterCameraData()
        self._data.create_buffers(self._num_envs, self.device)
        self._data._output = {
            "rgb": ProxyArray(wp.zeros((self._num_envs, 4, 4, 3), dtype=wp.uint8, device=self.device)),
            "depth": ProxyArray(wp.zeros((self._num_envs, 4, 4, 1), dtype=wp.float32, device=self.device)),
            "distance_to_camera": ProxyArray(wp.zeros((self._num_envs, 4, 4, 1), dtype=wp.float32, device=self.device)),
            "normals": ProxyArray(wp.zeros((self._num_envs, 4, 4, 3), dtype=wp.float32, device=self.device)),
            "semantic_segmentation": ProxyArray(
                wp.zeros((self._num_envs, 4, 4, 1), dtype=wp.int32, device=self.device)
            ),
            "motion_vectors": ProxyArray(wp.zeros((self._num_envs, 4, 4, 2), dtype=wp.float32, device=self.device)),
        }
        self._data.image_mesh_ids = ProxyArray(wp.zeros((self._num_envs, 4, 4, 1), dtype=wp.int32, device=self.device))
        self._pattern = torch.arange(48, device=self.device).reshape(1, 4, 4, 3)

    def _update_buffers_impl(self, env_mask: wp.array):
        phase = wp.to_torch(self._timestamp).reshape(-1, 1, 1, 1) * 120
        self._data.output["rgb"].torch.copy_(((self._pattern + 1) * (phase + 1) % 255).to(torch.uint8))
        self._data.output["depth"].torch.copy_(0.1 + self._pattern[..., :1] / 48 + phase / 100)
        self._data.output["depth"].torch[:, 0, :, 0] = torch.tensor(
            [float("nan"), float("inf"), -0.5, 3.0], device=self.device
        )
        self._data.image_mesh_ids.torch.copy_(((self._pattern[..., :1] + phase) % 4).to(torch.int32))
        self._data.pos_w.torch.copy_(phase.reshape(-1, 1) / 100 + 0.1)
        self._data.quat_w_world.torch[:, 3] = 1.0
        self._data.intrinsic_matrices.torch.copy_(torch.eye(3, device=self.device) * (phase.reshape(-1, 1, 1) + 1))
        self._data.output["distance_to_camera"].torch.copy_(0.5 + self._pattern[..., :1] / 48 + phase / 100)
        self._data.output["normals"].torch.copy_(torch.sin(self._pattern + phase))
        self._data.output["semantic_segmentation"].torch.copy_(((self._pattern[..., :1] + phase) % 4).to(torch.int32))
        self._data.output["motion_vectors"].torch.copy_(torch.cos(self._pattern[..., :2] + phase))


class InjectedContact(SensorBase):
    """Feed changing contact histories through the backend's annotated data container."""

    @property
    def data(self) -> ContactSensorData:
        self._update_outdated_buffers()
        return self._data

    def _initialize_impl(self):
        super()._initialize_impl()
        self._data = ContactSensorData()
        self._data.create_buffers(
            self._num_envs,
            1,
            1,
            3,
            True,
            True,
            True,
            self.device,
            track_contact_points=True,
            track_friction_forces=True,
        )

    def _update_buffers_impl(self, env_mask: wp.array):
        phase = wp.to_torch(self._timestamp).reshape(-1, 1, 1, 1)
        history = torch.arange(9, device=self.device).reshape(1, 3, 1, 3)
        self._data.net_normal_forces_w_history.torch.copy_(torch.sin(phase * 120 + history))
        for name in ("force_matrix_w", "friction_force_matrix_w", "contact_pos_w"):
            getattr(self._data, name).torch.copy_(torch.sin(phase * 120 + history[:, :1]))
        self._data.current_air_time.torch.copy_(phase.reshape(-1, 1))


class InjectedDeformable(SensorBase):
    """Supply nodal data without building a soft-body scene or starting Isaac Sim."""

    @property
    def data(self) -> DeformableObjectData:
        self._update_outdated_buffers()
        return self._data

    def _initialize_impl(self):
        super()._initialize_impl()

        class NodalView:
            count = self._num_envs
            max_simulation_nodes_per_body = 4
            max_simulation_elements_per_body = 1
            max_collision_elements_per_body = 1
            positions = wp.zeros((self._num_envs, 4, 3), dtype=wp.float32, device=self.device)
            velocities = wp.zeros((self._num_envs, 4, 3), dtype=wp.float32, device=self.device)

            def get_simulation_nodal_positions(self):
                return self.positions

            def get_simulation_nodal_velocities(self):
                return self.velocities

        self._view = NodalView()
        self._data = DeformableObjectData(self._view, self.device)
        self._data.nodal_kinematic_target = ProxyArray(
            wp.zeros((self._num_envs, 4), dtype=wp.vec4f, device=self.device)
        )

    def _update_buffers_impl(self, env_mask: wp.array):
        phase = wp.to_torch(self._timestamp).reshape(-1, 1, 1)
        wp.to_torch(self._view.positions).copy_(phase + torch.arange(12, device=self.device).reshape(1, 4, 3) / 100)
        wp.to_torch(self._view.velocities).copy_(
            torch.sin(phase * 120 + torch.arange(12, device=self.device).reshape(1, 4, 3))
        )
        self._data.nodal_kinematic_target.torch[..., :3].copy_(wp.to_torch(self._view.positions))
        self._data.nodal_kinematic_target.torch[..., 3] = 1.0
        self._data.update(self._sim_physics_dt)


class InjectedWrench(SensorBase):
    """Supply joint-wrench data without requiring solver-specific force reporting."""

    @property
    def data(self) -> JointWrenchSensorData:
        self._update_outdated_buffers()
        return self._data

    def _initialize_impl(self):
        super()._initialize_impl()
        self._data = JointWrenchSensorData()
        self._data.create_buffers(self._num_envs, 1, self.device)

    def _update_buffers_impl(self, env_mask: wp.array):
        phase = wp.to_torch(self._timestamp).reshape(-1, 1, 1)
        self._data.force.torch.copy_(torch.sin(phase * 120 + torch.arange(3, device=self.device)))
        self._data.torque.torch.copy_(torch.cos(phase * 120 + torch.arange(3, device=self.device)))


class InjectedCable(SensorBase):
    """Exercise the cable data contract without adding physical cable geometry."""

    class Data(BaseCableObjectData):
        def __init__(self, num_envs, device):
            super().__init__(device)
            self.pose = ProxyArray(wp.zeros((num_envs, 2), dtype=wp.transformf, device=device))
            self.velocity = ProxyArray(wp.zeros((num_envs, 2), dtype=wp.spatial_vectorf, device=device))

        @property
        def segment_pose_w(self):
            return self.pose

        @property
        def segment_velocity_w(self):
            return self.velocity

        def update(self, dt):
            self.pose.torch[..., :3] += dt
            self.pose.torch[..., 6] = 1.0
            self.velocity.torch.copy_(self.pose.torch[..., :6] * 0.2)

    @property
    def data(self) -> BaseCableObjectData:
        self._update_outdated_buffers()
        return self._data

    def _initialize_impl(self):
        super()._initialize_impl()
        self._data = self.Data(self._num_envs, self.device)

    def _update_buffers_impl(self, env_mask: wp.array):
        self._data.update(self._sim_physics_dt)


class InjectedRayMeshIds(SensorBase):
    """Supply mesh IDs through the multi-mesh ray caster's public data field."""

    @property
    def data(self) -> MultiMeshRayCasterData:
        self._update_outdated_buffers()
        return self._data

    def _initialize_impl(self):
        super()._initialize_impl()
        self._data = MultiMeshRayCasterData()
        self._data.create_buffers(self._num_envs, 4, self.device)
        self._data.ray_mesh_ids = ProxyArray(wp.zeros((self._num_envs, 4, 1), dtype=wp.int32, device=self.device))

    def _update_buffers_impl(self, env_mask: wp.array):
        phase = wp.to_torch(self._timestamp).reshape(-1, 1, 1) * 120
        self._data.ray_mesh_ids.torch.copy_(((phase + torch.arange(4, device=self.device).reshape(1, 4, 1)) % 5).int())


class InjectedTactile(SensorBase):
    """Exercise tactile tensor fields without starting the tactile renderer."""

    @property
    def data(self) -> VisuoTactileSensorData:
        self._update_outdated_buffers()
        return self._data

    def _initialize_impl(self):
        super()._initialize_impl()
        shapes = {
            "tactile_depth_image": (4, 4, 1),
            "tactile_rgb_image": (4, 4, 3),
            "tactile_points_pos_w": (4, 3),
            "tactile_points_quat_w": (4, 4),
            "penetration_depth": (4,),
            "tactile_normal_force": (4,),
            "tactile_shear_force": (4, 2),
        }
        self._data = VisuoTactileSensorData(
            **{name: torch.zeros((self._num_envs, *shape), device=self.device) for name, shape in shapes.items()}
        )

    def _update_buffers_impl(self, env_mask: wp.array):
        phase = wp.to_torch(self._timestamp) * 120
        for value in vars(self._data).values():
            pattern = torch.arange(value[0].numel(), device=self.device).reshape(1, *value.shape[1:])
            value.copy_(torch.sin(pattern + phase.reshape(-1, *([1] * (value.ndim - 1)))))


class CommandAction(ActionTerm):
    """A command for a downstream consumer, exercising processed-action fallback.

    Like a hierarchical policy command, this term has no directly intercepted joint write. Its
    processed output must survive export without inheriting the robot's low-level PD gains.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._raw_actions = torch.zeros(self.num_envs, 3, device=self.device)

    @property
    def action_dim(self) -> int:
        return 3

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._raw_actions * 0.5

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions = actions.clone()

    def apply_actions(self):
        pass

    def reset(self, env_ids=None):
        self._raw_actions[env_ids] = 0
