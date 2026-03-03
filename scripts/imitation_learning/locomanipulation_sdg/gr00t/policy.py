# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# GROOT IMPORTS

import torch
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import Gr00tPolicy


class Policy:
    """Wrapper around GR00T policy for G1 locomanipulation SDG."""

    def __init__(self, model_path: str, embodiment_tag: str):
        """Load the GR00T policy and locomanipulation SDG data config.

        Args:
            model_path: Path to the model checkpoint.
            embodiment_tag: Embodiment tag used by the model (e.g. "new_embodiment").
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.data_config = DATA_CONFIG_MAP["g1_locomanipulation_sdg"]
        self.modality_config = self.data_config.modality_config()
        self.modality_transform = self.data_config.transform()
        self.policy = Gr00tPolicy(
            model_path=model_path,
            embodiment_tag=embodiment_tag,
            modality_config=self.modality_config,
            modality_transform=self.modality_transform,
            device=self.device,
        )
