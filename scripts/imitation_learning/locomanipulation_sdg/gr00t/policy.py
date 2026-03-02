# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# GROOT IMPORTS

import torch
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import Gr00tPolicy


class Policy:
    def __init__(self, model_path, embodiment_tag):
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
