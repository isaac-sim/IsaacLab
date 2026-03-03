# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractmethod
from dataclasses import dataclass

from gr00t.data.dataset import ModalityConfig
from gr00t.data.transform.base import ComposedModalityTransform, ModalityTransform
from gr00t.data.transform.concat import ConcatTransform
from gr00t.data.transform.state_action import (
    StateActionToTensor,
    StateActionTransform,
)
from gr00t.data.transform.video import (
    VideoColorJitter,
    VideoCrop,
    VideoResize,
    VideoToNumpy,
    VideoToTensor,
)
from gr00t.model.transforms import GR00TTransform


@dataclass
class BaseDataConfig(ABC):
    """Base class for GR00T data configurations defining modalities and transforms."""

    def modality_config(self) -> dict[str, ModalityConfig]:
        """Build the modality config mapping (video, state, action, language) for the dataset.

        Returns:
            Dict mapping modality names to ModalityConfig instances.
        """
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        return {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }

    @abstractmethod
    def transform(self) -> ModalityTransform:
        """Return the composed modality transform (video, state, action, concat, model-specific).

        Returns:
            ModalityTransform to apply to dataset samples.
        """
        pass


#####################################################################################
# helper functions
#####################################################################################


def import_external_data_config(data_config_str: str) -> BaseDataConfig | None:
    """Import and instantiate an external data configuration class.

    Format: "module_path:ClassName" (e.g., "my_configs:RobotConfig").
    Supports nested modules like "package.submodule:ClassName".
    The current working directory is prepended to sys.path for resolution.

    Args:
        data_config_str: String in "module_path:ClassName" format. If no ":" is present,
            returns None.

    Returns:
        Instance of the requested config class, or None if data_config_str has no ":".
    """
    if ":" not in data_config_str:
        return None

    import importlib
    import os
    import sys
    from pathlib import Path

    # Add current working directory to Python path
    current_dir = str(Path(os.getcwd()).absolute())
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)

    try:
        module_path, class_name = data_config_str.split(":", 1)
        if not module_path or not class_name:
            raise ValueError(f"Invalid format: '{data_config_str}'. Use 'module:ClassName'")

        print(f"Loading external config: {module_path}.{class_name}")

        module = importlib.import_module(module_path)
        if not hasattr(module, class_name):
            available = [n for n in dir(module) if not n.startswith("_") and isinstance(getattr(module, n), type)]
            raise AttributeError(f"Class '{class_name}' not found in '{module_path}'. Available: {available}")

        # assert if the class has 'transform' and 'modality_config' methods
        if not hasattr(getattr(module, class_name), "transform"):
            raise AttributeError(f"Class '{class_name}' does not have a 'transform' method")
        if not hasattr(getattr(module, class_name), "modality_config"):
            raise AttributeError(f"Class '{class_name}' does not have a 'modality_config' method")

        return getattr(module, class_name)()

    except (ModuleNotFoundError, AttributeError, ValueError) as e:
        print(f"Config loading failed: {e}")
        print("Example: my_configs:MyConfig, package.submodule:ClassName")
        raise


def load_data_config(data_config_str: str) -> BaseDataConfig:
    """Resolve a data config from a short name or external module:class string.

    First looks up the string in DATA_CONFIG_MAP; if not found, tries to import
    and instantiate via import_external_data_config (e.g. "my_package.configs:MyConfig").

    Args:
        data_config_str: Short name (e.g. "g1_locomanipulation_sdg") or "module_path:ClassName".

    Returns:
        The corresponding BaseDataConfig instance.
    """
    if data_config_str in DATA_CONFIG_MAP:
        return DATA_CONFIG_MAP[data_config_str]
    data_config_cls = import_external_data_config(data_config_str)
    if data_config_cls is not None:
        return data_config_cls
    # Yellow warning color
    yellow = "\033[93m"
    reset = "\033[0m"
    raise ValueError(
        f"{yellow}Invalid data_config '{data_config_str}'. "
        f"Available options: {list(DATA_CONFIG_MAP.keys())}, "
        f"or use 'module:ClassName' for external configs{reset}"
    )


###########################################################################################


class G1LocomanipulationSDGDataConfig(BaseDataConfig):
    """Data config for G1 locomanipulation SDG (video, state, action; no language)."""

    video_keys = ["video.ego_view"]
    state_keys = [
        "state.left_hand_pose",
        "state.right_hand_pose",
        "state.left_hand_joint_positions",
        "state.right_hand_joint_positions",
        "state.object_pose",
        "state.goal_pose",
        "state.end_fixture_pose",
    ]
    action_keys = [
        "action.left_hand_pose",
        "action.right_hand_pose",
        "action.left_hand_joint_positions",
        "action.right_hand_joint_positions",
        "action.base_velocity",
        "action.base_height",
    ]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self) -> dict[str, ModalityConfig]:
        """Build modality config for video, state, and action (no language).

        Returns:
            Dict with "video", "state", and "action" ModalityConfig entries.
        """
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )

        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )

        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )

        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
        }

        return modality_configs

    def transform(self) -> ModalityTransform:
        """Composed transform: video (crop, resize, color jitter), state/action (tensor, normalize), concat, GR00T.

        Returns:
            ComposedModalityTransform for locomanipulation SDG samples.
        """
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "min_max" for key in self.state_keys},
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            # model-specific transform
            GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)


###########################################################################################

DATA_CONFIG_MAP = {
    "g1_locomanipulation_sdg": G1LocomanipulationSDGDataConfig(),
}
