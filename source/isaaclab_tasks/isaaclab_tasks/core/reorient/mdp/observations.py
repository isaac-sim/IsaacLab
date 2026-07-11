# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Functions specific to the in-hand dexterous manipulation environments."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.managers import ManagerTermBase, ObservationTermCfg, SceneEntityCfg
from isaaclab.utils.noise import NoiseModelCfg

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import Camera, JointWrenchSensor

    from isaaclab_tasks.core.reorient.config.shadow_hand.feature_extractor import FeatureExtractorCfg

    from .commands import ReorientCommand


wp.init()


@wp.kernel
def _fingertip_pos_kernel(
    body_pos_w: wp.array2d(dtype=wp.vec3f),
    env_origins: wp.array(dtype=wp.vec3f),
    body_ids: wp.array(dtype=wp.int32),
    out: wp.array2d(dtype=wp.float32),
):
    i, j = wp.tid()
    p = body_pos_w[i, body_ids[j]] - env_origins[i]
    out[i, 3 * j + 0] = p[0]
    out[i, 3 * j + 1] = p[1]
    out[i, 3 * j + 2] = p[2]


@wp.kernel
def _fingertip_quat_kernel(
    body_quat_w: wp.array2d(dtype=wp.quatf),
    body_ids: wp.array(dtype=wp.int32),
    out: wp.array2d(dtype=wp.float32),
):
    i, j = wp.tid()
    q = body_quat_w[i, body_ids[j]]
    out[i, 4 * j + 0] = q[0]
    out[i, 4 * j + 1] = q[1]
    out[i, 4 * j + 2] = q[2]
    out[i, 4 * j + 3] = q[3]


@wp.kernel
def _fingertip_wrench_kernel(
    force: wp.array2d(dtype=wp.vec3f),
    torque: wp.array2d(dtype=wp.vec3f),
    body_ids: wp.array(dtype=wp.int32),
    out: wp.array2d(dtype=wp.float32),
):
    i, j = wp.tid()
    f = force[i, body_ids[j]]
    t = torque[i, body_ids[j]]
    out[i, 6 * j + 0] = f[0]
    out[i, 6 * j + 1] = f[1]
    out[i, 6 * j + 2] = f[2]
    out[i, 6 * j + 3] = t[0]
    out[i, 6 * j + 4] = t[1]
    out[i, 6 * j + 5] = t[2]


@wp.kernel
def _fingertip_vel_kernel(
    body_vel_w: wp.array2d(dtype=wp.spatial_vectorf),
    body_ids: wp.array(dtype=wp.int32),
    out: wp.array2d(dtype=wp.float32),
):
    i, j = wp.tid()
    v = body_vel_w[i, body_ids[j]]
    for k in range(6):
        out[i, 6 * j + k] = v[k]


@wp.kernel
def _goal_quat_error_kernel(
    asset_quat: wp.array(dtype=wp.quatf),
    goal_quat: wp.array(dtype=wp.quatf),
    make_unique: int,
    out: wp.array(dtype=wp.quatf),
):
    """Per-environment quaternion error ``asset * conjugate(goal)`` in (x, y, z, w) order."""
    i = wp.tid()
    q1 = asset_quat[i]
    q2 = goal_quat[i]
    # Hamilton product against the conjugate, matching isaaclab.utils.math.quat_mul/quat_conjugate;
    # quaternions are stored (x, y, z, w)
    w = q1[3] * q2[3] + q1[0] * q2[0] + q1[1] * q2[1] + q1[2] * q2[2]
    x = q1[0] * q2[3] - q1[3] * q2[0] - q1[1] * q2[2] + q1[2] * q2[1]
    y = q1[1] * q2[3] - q1[3] * q2[1] - q1[2] * q2[0] + q1[0] * q2[2]
    z = q1[2] * q2[3] - q1[3] * q2[2] - q1[0] * q2[1] + q1[1] * q2[0]
    sign = 1.0
    # make_unique keeps the real part non-negative (isaaclab.utils.math.quat_unique)
    if make_unique != 0 and w < 0.0:
        sign = -1.0
    out[i] = wp.quatf(sign * x, sign * y, sign * z, sign * w)


class goal_quat_diff(ManagerTermBase):
    """Goal orientation relative to the asset's root frame.

    The real part is always positive when ``make_quat_unique`` is set.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._out = torch.empty(env.num_envs, 4, dtype=torch.float32, device=env.device)
        # cached Warp views; the hot loop launches the kernel without conversions
        self._out_wp = wp.from_torch(self._out, dtype=wp.quatf)
        # resolved on first call: the command term does not exist yet during manager construction
        self._goal_quat_wp: wp.array | None = None

    def __call__(
        self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str, make_quat_unique: bool
    ) -> torch.Tensor:
        """Return the per-environment quaternion error, shape ``(num_envs, 4)``."""
        asset: RigidObject = env.scene[asset_cfg.name]
        if self._goal_quat_wp is None:
            command_term: ReorientCommand = env.command_manager.get_term(command_name)
            self._goal_quat_wp = wp.from_torch(command_term.quat_command_w, dtype=wp.quatf)
        wp.launch(
            _goal_quat_error_kernel,
            dim=self.num_envs,
            inputs=[asset.data.root_quat_w.warp, self._goal_quat_wp, int(make_quat_unique)],
            outputs=[self._out_wp],
            device=self._out_wp.device,
        )
        return self._out


class fingertip_pos(ManagerTermBase):
    """Flattened fingertip positions in the environment frame [m]."""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        body_ids = cfg.params["asset_cfg"].body_ids
        self._body_ids_wp = wp.array(body_ids, dtype=wp.int32, device=str(env.device))
        self._out = torch.empty(env.num_envs, len(body_ids) * 3, dtype=torch.float32, device=env.device)
        self._out_wp = wp.from_torch(self._out)
        self._env_origins_wp = wp.from_torch(env.scene.env_origins, dtype=wp.vec3f)

    def __call__(self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
        """Return the flattened per-fingertip block, shape ``(num_envs, num_fingertips * 3)``."""
        asset = env.scene[asset_cfg.name]
        wp.launch(
            _fingertip_pos_kernel,
            dim=(self.num_envs, self._body_ids_wp.shape[0]),
            inputs=[asset.data.body_pos_w.warp, self._env_origins_wp, self._body_ids_wp],
            outputs=[self._out_wp],
            device=self._out_wp.device,
        )
        return self._out


class fingertip_quat(ManagerTermBase):
    """Flattened fingertip ``(x, y, z, w)`` orientations."""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        body_ids = cfg.params["asset_cfg"].body_ids
        self._body_ids_wp = wp.array(body_ids, dtype=wp.int32, device=str(env.device))
        self._out = torch.empty(env.num_envs, len(body_ids) * 4, dtype=torch.float32, device=env.device)
        self._out_wp = wp.from_torch(self._out)

    def __call__(self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
        """Return the flattened per-fingertip block, shape ``(num_envs, num_fingertips * 4)``."""
        asset = env.scene[asset_cfg.name]
        wp.launch(
            _fingertip_quat_kernel,
            dim=(self.num_envs, self._body_ids_wp.shape[0]),
            inputs=[asset.data.body_quat_w.warp, self._body_ids_wp],
            outputs=[self._out_wp],
            device=self._out_wp.device,
        )
        return self._out


class fingertip_vel(ManagerTermBase):
    """Flattened fingertip spatial velocities [m/s, rad/s]."""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        body_ids = cfg.params["asset_cfg"].body_ids
        self._body_ids_wp = wp.array(body_ids, dtype=wp.int32, device=str(env.device))
        self._out = torch.empty(env.num_envs, len(body_ids) * 6, dtype=torch.float32, device=env.device)
        self._out_wp = wp.from_torch(self._out)

    def __call__(self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
        """Return the flattened per-fingertip block, shape ``(num_envs, num_fingertips * 6)``."""
        asset = env.scene[asset_cfg.name]
        wp.launch(
            _fingertip_vel_kernel,
            dim=(self.num_envs, self._body_ids_wp.shape[0]),
            inputs=[asset.data.body_vel_w.warp, self._body_ids_wp],
            outputs=[self._out_wp],
            device=self._out_wp.device,
        )
        return self._out


class fingertip_wrench(ManagerTermBase):
    """Fingertip reaction wrenches [N, N·m] with Direct-compatible zero fallback."""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        body_ids = cfg.params["sensor_cfg"].body_ids
        self._body_ids_wp = wp.array(body_ids, dtype=wp.int32, device=str(env.device))
        self._out = torch.zeros(env.num_envs, len(body_ids) * 6, dtype=torch.float32, device=env.device)
        self._out_wp = wp.from_torch(self._out)

    def __call__(self, env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
        """Return the flattened wrench block, shape ``(num_envs, num_fingertips * 6)``."""
        sensor: JointWrenchSensor = env.scene.sensors[sensor_cfg.name]
        force_data = sensor.data.force
        torque_data = sensor.data.torque
        if force_data is None or torque_data is None:
            # Direct-compatible fallback: report zero wrenches until the sensor produces data
            return self._out
        wp.launch(
            _fingertip_wrench_kernel,
            dim=(self.num_envs, self._body_ids_wp.shape[0]),
            inputs=[force_data.warp, torque_data.warp, self._body_ids_wp],
            outputs=[self._out_wp],
            device=self._out_wp.device,
        )
        return self._out


def reorient_last_action(env: ManagerBasedRLEnv, action_name: str) -> torch.Tensor:
    """Return the Direct-compatible last action across same-step autoreset.

    Args:
        env: Environment containing the action term and reset buffers.
        action_name: Action term whose raw action is observed.

    Returns:
        Raw actions, retaining each terminal action in its same-step reset observation.
    """
    raw_action = env.action_manager.get_term(action_name).raw_actions
    reset_action = getattr(env, "_reorient_reset_action", None)
    reset_step = getattr(env, "_reorient_reset_step", None)
    common_step_counter = getattr(env, "common_step_counter", None)
    if reset_action is None or reset_step is None or common_step_counter is None:
        return raw_action
    return torch.where((reset_step == common_step_counter).unsqueeze(-1), reset_action, raw_action)


class OpenAIPolicyObservation(ManagerTermBase):
    """Apply one stateful noise model to the concatenated OpenAI actor observation."""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        noise_model: NoiseModelCfg = cfg.params["noise_model"]
        self._noise_model = noise_model.class_type(noise_model, num_envs=self.num_envs, device=self.device)
        self._quat_error = torch.empty(env.num_envs, 4, dtype=torch.float32, device=env.device)
        robot_body_ids = cfg.params["robot_cfg"].body_ids
        self._robot_body_ids_wp = wp.array(robot_body_ids, dtype=wp.int32, device=str(env.device))
        self._fingertip_buf = torch.empty(env.num_envs, len(robot_body_ids) * 3, dtype=torch.float32, device=env.device)
        self._fingertip_buf_wp = wp.from_torch(self._fingertip_buf)
        self._env_origins_wp = wp.from_torch(env.scene.env_origins, dtype=wp.vec3f)
        # cached Warp views; the hot loop launches the kernel without conversions
        self._quat_error_wp = wp.from_torch(self._quat_error, dtype=wp.quatf)
        # resolved on first call: the command term does not exist yet during manager construction
        self._goal_quat_wp: wp.array | None = None
        # ObservationManager probes callable terms once for their shape and then
        # calls reset. Keep that probe side-effect free so initialization matches
        # DirectRLEnv's first noise-model reset and application.
        self._shape_probe_pending = True

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset the actor observation bias for selected environments.

        Args:
            env_ids: Environment indices to reset, or ``None`` for every environment.
        """
        if self._shape_probe_pending:
            self._shape_probe_pending = False
            return
        self._noise_model.reset(env_ids)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        action_name: str,
        noise_model: NoiseModelCfg,
        robot_cfg: SceneEntityCfg,
        object_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """Return the corrupted 42-dimensional actor observation."""
        del noise_model
        object_asset: RigidObject = env.scene[object_cfg.name]
        object_pos = object_asset.data.root_pos_w.torch - env.scene.env_origins
        if self._goal_quat_wp is None:
            command_term: ReorientCommand = env.command_manager.get_term(command_name)
            self._goal_quat_wp = wp.from_torch(command_term.quat_command_w, dtype=wp.quatf)
        wp.launch(
            _goal_quat_error_kernel,
            dim=self.num_envs,
            inputs=[object_asset.data.root_quat_w.warp, self._goal_quat_wp, 0],
            outputs=[self._quat_error_wp],
            device=self._quat_error_wp.device,
        )
        robot = env.scene[robot_cfg.name]
        wp.launch(
            _fingertip_pos_kernel,
            dim=(self.num_envs, self._robot_body_ids_wp.shape[0]),
            inputs=[robot.data.body_pos_w.warp, self._env_origins_wp, self._robot_body_ids_wp],
            outputs=[self._fingertip_buf_wp],
            device=self._fingertip_buf_wp.device,
        )
        # Direct actor-observation order: fingertips, object position, goal quat error, last action
        observation = torch.cat(
            (self._fingertip_buf, object_pos, self._quat_error, reorient_last_action(env, action_name)),
            dim=-1,
        )
        if self._shape_probe_pending:
            return observation
        return self._noise_model(observation)


# ---------------------------------------------------------------------------
# Shadow Hand camera observation terms.
#
# These terms wrap the CNN feature pipeline defined in the shadow-hand config
# package. The config layer imports the mdp layer, so the FeatureExtractor
# machinery is imported lazily at term construction/call time.
# ---------------------------------------------------------------------------


class ShadowHandCameraFeatures(ManagerTermBase):
    """Run the Direct camera feature pipeline as one Manager observation term."""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        sensor_cfg: SceneEntityCfg = cfg.params["sensor_cfg"]
        camera: Camera = env.scene.sensors[sensor_cfg.name]
        # Runtime-only import: the mdp layer must not import the task-config layer
        # at module load (config modules import mdp; see the layering note above).
        from isaaclab_tasks.core.reorient.config.shadow_hand.feature_extractor import FeatureExtractor

        feature_extractor_cfg: FeatureExtractorCfg = env.cfg.feature_extractor
        self._feature_extractor = FeatureExtractor(
            feature_extractor_cfg,
            env.device,
            camera.cfg.data_types,
            env.cfg.log_dir,
            height=camera.cfg.height,
            width=camera.cfg.width,
        )
        # ObservationManager calls terms once to infer their shape. Do not train
        # or save a CNN checkpoint during that initialization probe.
        self._shape_probe_pending = True
        self._keypoints_buf = torch.empty(env.num_envs, 8, 3, dtype=torch.float32, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Finish the shape-probe phase on the first Manager reset.

        Args:
            env_ids: Environment indices being reset. The feature extractor
                has no per-environment state, so the indices are unused.
        """
        del env_ids
        if self._shape_probe_pending:
            self._shape_probe_pending = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        feature_extractor_cfg: FeatureExtractorCfg,
        sensor_cfg: SceneEntityCfg,
        object_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """Return the detached 27-dimensional cube-pose embedding.

        Args:
            env: Environment containing the object and tiled camera.
            feature_extractor_cfg: Feature-extractor configuration captured by
                the observation term. The initialized extractor owns its copy.
            sensor_cfg: Tiled-camera scene entity.
            object_cfg: Reoriented-object scene entity.

        Returns:
            Predicted object position and cube keypoints [m], shape
            ``(num_envs, 27)``.
        """
        del feature_extractor_cfg
        if self._shape_probe_pending:
            embeddings = torch.zeros(env.num_envs, 27, dtype=torch.float32, device=env.device)
            env._shadow_hand_camera_embeddings = embeddings
            return embeddings

        from isaaclab_tasks.core.reorient.config.shadow_hand.feature_extractor import compute_cube_keypoints

        camera: Camera = env.scene.sensors[sensor_cfg.name]
        object_asset: RigidObject = env.scene[object_cfg.name]
        object_pos = object_asset.data.root_pos_w.torch - env.scene.env_origins
        object_pose = torch.cat((object_pos, object_asset.data.root_quat_w.torch), dim=-1)
        keypoints = compute_cube_keypoints(object_pose, out=self._keypoints_buf)
        target = torch.cat((object_pos, keypoints.flatten(start_dim=1)), dim=-1)
        camera_output = {
            data_type: value if isinstance(value, torch.Tensor) else value.torch
            for data_type, value in camera.data.output.items()
        }
        pose_loss, embeddings = self._feature_extractor.step(camera_output, target)
        embeddings = embeddings.clone().detach()
        env._shadow_hand_camera_embeddings = embeddings
        if pose_loss is not None:
            env.extras.setdefault("log", {})["pose_loss"] = pose_loss
        return embeddings


def shadow_hand_camera_cached_features(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return camera features computed by the preceding policy observation group.

    Args:
        env: Environment whose policy group cached the current camera embedding.

    Returns:
        Detached camera embeddings, shape ``(num_envs, 27)``.
    """
    embeddings = getattr(env, "_shadow_hand_camera_embeddings", None)
    if embeddings is None:
        raise RuntimeError("Shadow Hand camera policy features must be computed before critic observations.")
    return embeddings


class shadow_hand_goal_keypoints(ManagerTermBase):
    """Flattened zero-origin cube keypoints [m] for the current goal orientation."""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._out = torch.empty(env.num_envs, 24, dtype=torch.float32, device=env.device)
        self._out_wp = wp.from_torch(self._out)
        # resolved on first call: the command term does not exist yet during manager construction
        self._goal_quat_wp: wp.array | None = None

    def __call__(self, env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
        """Return flattened zero-origin cube keypoints [m], shape ``(num_envs, 24)``."""
        # imported lazily: the config layer imports this mdp layer (see the module comment)
        from isaaclab_tasks.core.reorient.config.shadow_hand.feature_extractor import (
            CUBE_HALF_SIZE,
            cube_keypoints_from_quat_kernel,
        )

        if self._goal_quat_wp is None:
            command_term = env.command_manager.get_term(command_name)
            self._goal_quat_wp = wp.from_torch(command_term.quat_command_w, dtype=wp.quatf)
        wp.launch(
            cube_keypoints_from_quat_kernel,
            dim=(self.num_envs, 8),
            inputs=[self._goal_quat_wp, CUBE_HALF_SIZE],
            outputs=[self._out_wp],
            device=self._out_wp.device,
        )
        return self._out
