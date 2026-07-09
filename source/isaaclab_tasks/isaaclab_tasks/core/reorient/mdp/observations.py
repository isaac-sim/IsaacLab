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
def _goal_quat_error_kernel(
    asset_quat: wp.array(dtype=wp.vec4),
    goal_quat: wp.array(dtype=wp.vec4),
    make_unique: int,
    out: wp.array(dtype=wp.vec4),
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
    out[i] = wp.vec4(sign * x, sign * y, sign * z, sign * w)


def compute_goal_quat_error(
    asset_quat: torch.Tensor, goal_quat: torch.Tensor, make_quat_unique: bool, out: torch.Tensor
) -> torch.Tensor:
    """Compute the quaternion error between asset and goal orientations.

    Args:
        asset_quat: Asset ``(x, y, z, w)`` orientations, shape ``(num_envs, 4)``.
        goal_quat: Goal ``(x, y, z, w)`` orientations, shape ``(num_envs, 4)``.
        make_quat_unique: Flip the sign so the real part is always non-negative.
        out: Caller-owned output buffer, shape ``(num_envs, 4)``, float32.

    Returns:
        ``out`` filled with per-environment quaternion errors.
    """
    wp.launch(
        _goal_quat_error_kernel,
        dim=out.shape[0],
        # .contiguous(): the goal quaternion arrives as a non-contiguous command slice
        inputs=[
            wp.from_torch(asset_quat.contiguous(), dtype=wp.vec4),
            wp.from_torch(goal_quat.contiguous(), dtype=wp.vec4),
            int(make_quat_unique),
        ],
        outputs=[wp.from_torch(out, dtype=wp.vec4)],
        device=wp.device_from_torch(out.device),
    )
    return out


class goal_quat_diff(ManagerTermBase):
    """Goal orientation relative to the asset's root frame.

    The real part is always positive when ``make_quat_unique`` is set.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._out = torch.empty(env.num_envs, 4, dtype=torch.float32, device=env.device)

    def __call__(
        self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str, make_quat_unique: bool
    ) -> torch.Tensor:
        """Return the per-environment quaternion error, shape ``(num_envs, 4)``."""
        asset: RigidObject = env.scene[asset_cfg.name]
        command_term: ReorientCommand = env.command_manager.get_term(command_name)
        return compute_goal_quat_error(
            asset.data.root_quat_w.torch, command_term.command[:, 3:7], make_quat_unique, self._out
        )


def fingertip_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return flattened fingertip positions in the environment frame [m].

    Args:
        env: Environment containing the hand.
        asset_cfg: Hand entity with resolved fingertip body indices.

    Returns:
        Fingertip positions [m], shape ``(num_envs, num_fingertips * 3)``.
    """
    asset = env.scene[asset_cfg.name]
    positions = asset.data.body_pos_w.torch[:, asset_cfg.body_ids]
    positions = positions - env.scene.env_origins[:, None, :]
    return positions.flatten(start_dim=1)


def fingertip_quat(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return flattened fingertip ``(x, y, z, w)`` orientations.

    Args:
        env: Environment containing the hand.
        asset_cfg: Hand entity with resolved fingertip body indices.

    Returns:
        Unit quaternions, shape ``(num_envs, num_fingertips * 4)``.
    """
    asset = env.scene[asset_cfg.name]
    return asset.data.body_quat_w.torch[:, asset_cfg.body_ids].flatten(start_dim=1)


def fingertip_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return flattened fingertip spatial velocities in the world frame.

    Args:
        env: Environment containing the hand.
        asset_cfg: Hand entity with resolved fingertip body indices.

    Returns:
        Spatial velocities [m/s, rad/s], shape ``(num_envs, num_fingertips * 6)``.
    """
    asset = env.scene[asset_cfg.name]
    return asset.data.body_vel_w.torch[:, asset_cfg.body_ids].flatten(start_dim=1)


def fingertip_wrench(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return fingertip reaction wrenches with Direct-compatible zero fallback.

    Args:
        env: Environment containing the joint-wrench sensor.
        sensor_cfg: Joint-wrench sensor entity with resolved fingertip body indices.

    Returns:
        Fingertip reaction wrenches [N, N·m], shape ``(num_envs, num_fingertips * 6)``.
    """
    sensor: JointWrenchSensor = env.scene.sensors[sensor_cfg.name]
    force_data = sensor.data.force
    torque_data = sensor.data.torque
    if force_data is None or torque_data is None:
        body_count = len(sensor_cfg.body_ids)
        return torch.zeros(env.num_envs, body_count * 6, device=env.device)
    force = force_data.torch[:, sensor_cfg.body_ids]
    torque = torque_data.torch[:, sensor_cfg.body_ids]
    return torch.cat((force, torque), dim=-1).flatten(start_dim=1)


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
        command_term: ReorientCommand = env.command_manager.get_term(command_name)
        compute_goal_quat_error(
            object_asset.data.root_quat_w.torch, command_term.command[:, 3:7], False, self._quat_error
        )
        # Direct actor-observation order: fingertips, object position, goal quat error, last action
        observation = torch.cat(
            (fingertip_pos(env, robot_cfg), object_pos, self._quat_error, reorient_last_action(env, action_name)),
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


def shadow_hand_goal_keypoints(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Return zero-origin cube keypoints for the current goal orientation.

    Args:
        env: Environment containing the goal command.
        command_name: Goal command term name.

    Returns:
        Flattened zero-origin cube keypoints [m], shape ``(num_envs, 24)``.
    """
    from isaaclab_tasks.core.reorient.config.shadow_hand.feature_extractor import compute_cube_keypoints

    command = env.command_manager.get_command(command_name)
    goal_pose = torch.cat((torch.zeros_like(command[:, :3]), command[:, 3:7]), dim=-1)
    return compute_cube_keypoints(goal_pose).flatten(start_dim=1)
