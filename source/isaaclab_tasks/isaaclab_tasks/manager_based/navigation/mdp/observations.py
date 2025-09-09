# Copyright (c) 2025, The Nav-Suite Project Developers (https://github.com/leggedrobotics/nav-suite/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import torch.types
from typing import TYPE_CHECKING

import torchvision.transforms as transforms
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import Camera, RayCasterCamera

from .depth_embedding import DepthEmbedderSingleton

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

    from .camera_observations_cfg import DINOEmbeddedRGBImageCfg, EmbeddedDepthImageCfg


def camera_image(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    data_type: str = "distance_to_image_plane",
    flatten: bool = False,
    nan_fill_value: float | None = None,
) -> torch.Tensor:
    """Camera image Observations.

    The camera image observation from the given sensor w.r.t. the asset's root frame.
    Also removes nan/inf values and sets them to the maximum distance of the sensor

    Args:
        env: The environment object.
        sensor_cfg: The name of the sensor.
        data_type: The type of data to extract from the sensor. Default is "distance_to_image_plane".
        flatten: If True, the image will be flattened to 1D. Default is False.
        nan_fill_value: The value to fill nan/inf values with. If None, the maximum distance of the sensor will be used.

    Returns:
        The image data."""
    # extract the used quantities (to enable type-hinting)
    sensor: Camera | RayCasterCamera = env.scene.sensors[sensor_cfg.name]

    img = sensor.data.output[data_type].clone()

    if data_type == "distance_to_image_plane":
        if nan_fill_value is None:
            nan_fill_value = sensor.cfg.max_distance
        img = torch.nan_to_num(img, nan=nan_fill_value, posinf=nan_fill_value, neginf=0.0)

    # if type torch.uint8, convert to float and scale between 0 and 1
    if img.dtype == torch.uint8:
        img = img.to(torch.float32) / 255.0

    if flatten:
        return img.flatten(start_dim=1)
    else:
        # reorder the image to [BS, C, H, W] if it is not already in that shape
        if img.shape[-1] == 1 or img.shape[-1] == 3:
            img = img.permute(0, 3, 1, 2)

        return img


class EmbeddedDepthImageTerm(ManagerTermBase):
    """An observation term that embeds the depth image using the pre-trained PerceptNet."""

    cfg: EmbeddedDepthImageCfg

    def __init__(self, cfg: EmbeddedDepthImageCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        # Use the singleton to get the shared DepthEmbedder instance
        self.embedder = DepthEmbedderSingleton.get_embedder(env.device)

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        """Embedded depth image observations.

        Uses the PerceptNet to embed the depth image.

        Args:
            env: The environment object.

        Returns:
            The embedded depth image data.
        """
        depth_img = camera_image(env, self.cfg.sensor_cfg)
        # We only use the first channel of the depth image, there is only one channel but the shape is (H, W, 1)
        depth_img = self.embedder.process_image(depth_img[:, :, :, 0])
        return depth_img


class DINOEmbeddedRGBImageTerm(ManagerTermBase):
    """An observation term that embeds the RGB image using the pre-trained DINO."""

    cfg: DINOEmbeddedRGBImageCfg

    def __init__(self, cfg: DINOEmbeddedRGBImageCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        # get the backbone name
        backbone_archs = {
            "small": "vits14",
            "base": "vitb14",
            "large": "vitl14",
            "giant": "vitg14",
        }
        backbone_name = f"dinov2_{backbone_archs[self.cfg.backbone_size]}"
        if self.cfg.with_registers:
            backbone_name = f"{backbone_name}_reg"

        # setup dinov2 model
        self.model = torch.hub.load("facebookresearch/dinov2", backbone_name)
        self.model.eval()
        self.model.to(self._env.device)

        # transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # default with bilinear interpolation
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        """Embedded RGB image observations.

        Uses the pre-trained DINO model to embed the RGB image.

        Args:
            env: The environment object.

        Returns:
            The embedded RGB image data.
        """

        # get the rgb image
        rgb_img = camera_image(env, self.cfg.sensor_cfg, data_type="rgb", flatten=False)

        # apply the transforms
        rgb_img = self.transform(rgb_img)

        # embed the rgb image
        embedded_rgb_img = self.model(rgb_img)

        return embedded_rgb_img
