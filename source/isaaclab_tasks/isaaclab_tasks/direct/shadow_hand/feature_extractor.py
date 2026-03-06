# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import glob
import os

import torch
import torch.nn as nn
import torchvision

from isaaclab.sensors import save_images_to_file
from isaaclab.utils import configclass

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


class FeatureExtractorNetwork(nn.Module):
    """CNN architecture used to regress keypoint positions of the in-hand cube from image data."""

    def __init__(self, num_channel: int = 7, data_types: list[str] | None = None):
        """Initialize the CNN.

        Args:
            num_channel: Total number of input channels across all data types.
            data_types: Ordered list of camera data types that form the channel stack.
                Used to determine which channel ranges receive ImageNet normalization.
                Defaults to ``["rgb", "depth", "semantic_segmentation"]``.
        """
        super().__init__()
        if data_types is None:
            data_types = ["rgb", "depth", "semantic_segmentation"]

        self.cnn = nn.Sequential(
            nn.Conv2d(num_channel, 16, kernel_size=6, stride=2, padding=0),
            nn.ReLU(),
            nn.LayerNorm([16, 58, 58]),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.LayerNorm([32, 28, 28]),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.LayerNorm([64, 13, 13]),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.LayerNorm([128, 6, 6]),
            nn.AvgPool2d(6),
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
        x = x.permute(0, 3, 1, 2)
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
    ):
        """Initialize the feature extractor model.

        Args:
            cfg: Configuration for the feature extractor model.
            device: Device to run the model on.
            data_types: Ordered list of camera data types that form the CNN input channel
                stack. Should match the resolved :attr:`~isaaclab.sensors.TiledCameraCfg.data_types`
                of the tiled camera. Total input channels are derived from
                :data:`_DATA_TYPE_CHANNELS`.
            log_dir: Directory to save checkpoints. Default is None, which uses the local
                "logs" folder resolved relative to this file.
        """
        self.cfg = cfg
        self.device = device
        self.data_types = data_types

        # Compute total input channels from the camera data types.
        num_channel = sum(_DATA_TYPE_CHANNELS.get(dt, 3) for dt in data_types)

        # Feature extractor model.
        self.feature_extractor = FeatureExtractorNetwork(num_channel=num_channel, data_types=data_types)
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
