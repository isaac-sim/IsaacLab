# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import glob
import os
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torchvision

from isaaclab.managers import ManagerTermBase, ObservationTermCfg, SceneEntityCfg
from isaaclab.sensors import save_images_to_file
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import quat_apply

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import Camera

# Number of output channels for each supported camera data type.
_DATA_TYPE_CHANNELS: dict[str, int] = {
    "rgb": 3,
    "depth": 1,
    "semantic_segmentation": 3,
    "albedo": 3,
    "simple_shading_constant_diffuse": 3,
    "simple_shading_diffuse_mdl": 3,
    "simple_shading_full_mdl": 3,
}

# Data types whose channels should receive ImageNet normalization in the CNN forward pass.
_IMAGENET_NORM_TYPES: frozenset[str] = frozenset(
    {
        "rgb",
        "semantic_segmentation",
        "albedo",
        "simple_shading_constant_diffuse",
        "simple_shading_diffuse_mdl",
        "simple_shading_full_mdl",
    }
)


def _conv_out(size: int, kernel: int, stride: int, padding: int = 0) -> int:
    """Compute the spatial output size of a single convolutional layer."""
    return (size + 2 * padding - kernel) // stride + 1


class FeatureExtractorNetwork(nn.Module):
    """CNN architecture used to regress keypoint positions of the in-hand cube from image data."""

    def __init__(
        self,
        num_channel: int = 7,
        data_types: list[str] | None = None,
        height: int = 120,
        width: int = 120,
    ):
        """Initialize the CNN.

        Args:
            num_channel: Total number of input channels across all data types.
            data_types: Ordered list of camera data types that form the channel stack.
                Used to determine which channel ranges receive ImageNet normalization.
                Defaults to ``["rgb", "depth", "semantic_segmentation"]``.
            height: Input image height [px]. Used to compute :class:`~torch.nn.LayerNorm`
                spatial dimensions. Default is ``120``.
            width: Input image width [px]. Used to compute :class:`~torch.nn.LayerNorm`
                spatial dimensions. Default is ``120``.
        """
        super().__init__()
        if data_types is None:
            data_types = ["rgb", "depth", "semantic_segmentation"]

        # Compute spatial sizes after each conv to build resolution-adaptive LayerNorms.
        h1, w1 = _conv_out(height, 6, 2), _conv_out(width, 6, 2)
        h2, w2 = _conv_out(h1, 4, 2), _conv_out(w1, 4, 2)
        h3, w3 = _conv_out(h2, 4, 2), _conv_out(w2, 4, 2)
        h4, w4 = _conv_out(h3, 3, 2), _conv_out(w3, 3, 2)

        self.cnn = nn.Sequential(
            nn.Conv2d(num_channel, 16, kernel_size=6, stride=2, padding=0),
            nn.ReLU(),
            nn.LayerNorm([16, h1, w1]),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.LayerNorm([32, h2, w2]),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.LayerNorm([64, h3, w3]),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.LayerNorm([128, h4, w4]),
            nn.AdaptiveAvgPool2d(1),
        )

        self.linear = nn.Sequential(
            nn.Linear(128, 27),
        )

        self.data_transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        # Pre-compute channel ranges that require ImageNet normalization.
        self._imagenet_norm_ranges: list[tuple[int, int]] = []
        channel_idx = 0
        for dt in data_types:
            n_ch = _DATA_TYPE_CHANNELS.get(dt, 3)
            if dt in _IMAGENET_NORM_TYPES:
                self._imagenet_norm_ranges.append((channel_idx, channel_idx + n_ch))
            channel_idx += n_ch

    def forward(self, x):
        x = x.permute(0, 3, 1, 2).clone()
        for start, end in self._imagenet_norm_ranges:
            x[:, start:end, :, :] = self.data_transforms(x[:, start:end, :, :])
        cnn_x = self.cnn(x)
        out = self.linear(cnn_x.view(-1, 128))
        return out


@configclass
class FeatureExtractorCfg:
    """Configuration for the feature extractor model."""

    train: bool = True
    """If True, the feature extractor model is trained during the rollout process. Default is True."""

    load_checkpoint: bool = False
    """If True, the feature extractor model is loaded from a checkpoint. Default is False."""

    write_image_to_file: bool = False
    """If True, the images from the camera sensor are written to file. Default is False."""

    enabled: bool = True
    """If True, the CNN forward pass is executed each step.

    Set to False to bypass the network entirely and return zero embeddings. This is useful
    for benchmarking rendering throughput without CNN inference overhead. Default is True.
    """


@torch.jit.script
def compute_cube_keypoints(
    pose: torch.Tensor,
    num_keypoints: int = 8,
    size: tuple[float, float, float] = (2 * 0.03, 2 * 0.03, 2 * 0.03),
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute cube-corner positions for batched poses.

    Args:
        pose: Cube center poses ``(x, y, z, qx, qy, qz, qw)`` [m, unit quaternion].
        num_keypoints: Number of binary-sign corners to compute.
        size: Cube side lengths along each axis [m].
        out: Optional output buffer [m], shape ``(num_envs, num_keypoints, 3)``.

    Returns:
        Cube-corner positions [m], shape ``(num_envs, num_keypoints, 3)``.
    """
    num_envs = pose.shape[0]
    if out is None:
        out = torch.ones(num_envs, num_keypoints, 3, dtype=torch.float32, device=pose.device)
    else:
        out[:] = 1.0
    for i in range(num_keypoints):
        positive_axes = [((i >> axis) & 1) == 0 for axis in range(3)]
        corner_values = ([(1 if positive_axes[axis] else -1) * side / 2 for axis, side in enumerate(size)],)
        corner = torch.tensor(corner_values, dtype=torch.float32, device=pose.device) * out[:, i, :]
        out[:, i, :] = pose[:, :3] + quat_apply(pose[:, 3:7], corner)
    return out


class FeatureExtractor:
    """Class for extracting features from image data.

    It uses a CNN to regress keypoint positions from normalized images.
    If :attr:`FeatureExtractorCfg.train` is ``True``, the CNN is trained during rollouts.
    If :attr:`FeatureExtractorCfg.enabled` is ``False``, the network is bypassed and zero
    embeddings are returned (useful for benchmarking rendering throughput).

    The input data types (and therefore the CNN's input channel count) are determined by
    the camera's ``data_types`` at construction time, passed via the ``data_types`` argument.
    This means changing the camera preset (e.g. ``presets=rgb``) automatically reconfigures
    the CNN without requiring a separate environment config class.
    """

    def __init__(
        self,
        cfg: FeatureExtractorCfg,
        device: str,
        data_types: list[str],
        log_dir: str | None = None,
        height: int = 120,
        width: int = 120,
    ):
        """Initialize the feature extractor model.

        Args:
            cfg: Configuration for the feature extractor model.
            device: Device to run the model on.
            data_types: Ordered list of camera data types that form the CNN input channel
                stack. Should match the resolved :attr:`~isaaclab.sensors.CameraCfg.data_types`
                of the camera. Total input channels are derived from
                :data:`_DATA_TYPE_CHANNELS`.
            log_dir: Directory to save checkpoints. Default is None, which uses the local
                "logs" folder resolved relative to this file.
            height: Camera image height [px]. Must match the camera
                :attr:`~isaaclab.sensors.CameraCfg.height`. Default is ``120``.
            width: Camera image width [px]. Must match the camera
                :attr:`~isaaclab.sensors.CameraCfg.width`. Default is ``120``.
        """
        self.cfg = cfg
        self.device = device
        self.data_types = data_types

        # Compute total input channels from the camera data types.
        num_channel = sum(_DATA_TYPE_CHANNELS.get(dt, 3) for dt in data_types)

        # Feature extractor model.
        self.feature_extractor = FeatureExtractorNetwork(
            num_channel=num_channel, data_types=data_types, height=height, width=width
        )
        self.feature_extractor.to(self.device)

        self.step_count = 0
        if log_dir is not None:
            self.log_dir = log_dir
        else:
            self.log_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "logs")
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        if self.cfg.load_checkpoint:
            list_of_files = glob.glob(self.log_dir + "/*.pth")
            latest_file = max(list_of_files, key=os.path.getctime)
            checkpoint = os.path.join(self.log_dir, latest_file)
            print(f"[INFO]: Loading feature extractor checkpoint from {checkpoint}")
            self.feature_extractor.load_state_dict(torch.load(checkpoint, weights_only=True))

        if self.cfg.train:
            self.optimizer = torch.optim.Adam(self.feature_extractor.parameters(), lr=1e-4)
            self.l2_loss = nn.MSELoss()
            self.feature_extractor.train()
        else:
            self.feature_extractor.eval()

    def _preprocess_images(self, camera_output: dict[str, torch.Tensor]) -> torch.Tensor:
        """Preprocesses and concatenates camera images into a single tensor.

        Each data type in :attr:`FeatureExtractorCfg.data_types` is extracted from
        ``camera_output``, normalized, and concatenated along the channel dimension.

        Args:
            camera_output: Dictionary mapping data type names to image tensors.

        Returns:
            Concatenated preprocessed image tensor of shape (N, H, W, C).
        """
        tensors = []
        for dt in self.data_types:
            img = camera_output[dt].float()
            if dt == "rgb":
                img = img / 255.0
            elif dt == "depth":
                img[img == float("inf")] = 0
                img /= 5.0
                max_val = img.max()
                if max_val > 0:
                    img /= max_val
            elif dt == "semantic_segmentation":
                img = img[..., :3] / 255.0
                mean_tensor = torch.mean(img, dim=(1, 2), keepdim=True)
                img = img - mean_tensor
            else:
                # albedo and simple_shading_* are RGB-like 3-channel outputs.
                img = img[..., :3] / 255.0
            tensors.append(img)
        return torch.cat(tensors, dim=-1)

    def _save_images(self, camera_output: dict[str, torch.Tensor]):
        """Writes configured camera data buffers to file as normalized float images.

        Raw camera tensors are converted to float ``[0, 1]`` before saving so that
        :func:`~isaaclab.sensors.save_images_to_file` (which delegates to
        ``torchvision.utils.save_image``) receives the expected float input.

        Args:
            camera_output: Dictionary mapping data type names to image tensors.
        """
        for dt in self.data_types:
            if dt not in camera_output:
                continue
            img = camera_output[dt].float()
            if dt == "depth":
                img = img.clone()
                img[img == float("inf")] = 0
                max_val = img.max()
                if max_val > 0:
                    img = img / max_val
            else:
                # rgb, semantic_segmentation, albedo, and simple_shading_* are uint8 [0, 255]
                img = img[..., :3] / 255.0
            save_images_to_file(img, f"shadow_hand_{dt}.png")

    def step(
        self, camera_output: dict[str, torch.Tensor], gt_pose: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """Extracts features and optionally trains the CNN.

        Image saving (when :attr:`FeatureExtractorCfg.write_image_to_file` is ``True``) always
        runs first, regardless of whether the network is enabled.  When
        :attr:`FeatureExtractorCfg.enabled` is ``False``, the network is then bypassed and
        zero embeddings are returned without any further image preprocessing.

        Args:
            camera_output: Dictionary mapping data type names to image tensors from the
                tiled camera sensor.
            gt_pose: Ground truth pose tensor (position and keypoint corners). Shape: (N, 27).

        Returns:
            tuple[torch.Tensor | None, torch.Tensor]: Pose loss (``None`` when not training
                or when the network is disabled) and the predicted pose embedding of shape
                (N, 27).
        """
        if self.cfg.write_image_to_file:
            self._save_images(camera_output)

        if not self.cfg.enabled:
            batch_size = next(iter(camera_output.values())).shape[0]
            return None, torch.zeros(batch_size, 27, dtype=torch.float32, device=self.device)

        img_input = self._preprocess_images(camera_output)

        if self.cfg.train:
            with torch.enable_grad():
                with torch.inference_mode(False):
                    self.optimizer.zero_grad()

                    predicted_pose = self.feature_extractor(img_input)
                    pose_loss = self.l2_loss(predicted_pose, gt_pose.clone()) * 100

                    pose_loss.backward()
                    self.optimizer.step()

                    if self.step_count % 50000 == 0:
                        torch.save(
                            self.feature_extractor.state_dict(),
                            os.path.join(self.log_dir, f"cnn_{self.step_count}_{pose_loss.detach().cpu().numpy()}.pth"),
                        )

                    self.step_count += 1

                    return pose_loss, predicted_pose
        else:
            predicted_pose = self.feature_extractor(img_input)
            return None, predicted_pose


class ShadowHandCameraFeatures(ManagerTermBase):
    """Run the Direct camera feature pipeline as one Manager observation term."""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        sensor_cfg: SceneEntityCfg = cfg.params["sensor_cfg"]
        camera: Camera = env.scene.sensors[sensor_cfg.name]
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

        camera: Camera = env.scene.sensors[sensor_cfg.name]
        object_asset: RigidObject = env.scene[object_cfg.name]
        object_pos = object_asset.data.root_pos_w.torch - env.scene.env_origins
        object_pose = torch.cat((object_pos, object_asset.data.root_quat_w.torch), dim=-1)
        keypoints = compute_cube_keypoints(object_pose)
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
    command = env.command_manager.get_command(command_name)
    goal_pose = torch.cat((torch.zeros_like(command[:, :3]), command[:, 3:7]), dim=-1)
    return compute_cube_keypoints(goal_pose).flatten(start_dim=1)
