# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Lift environments.

Task definitions adapted from:

Reorient:
@article{petrenko2023dexpbt,
  title={Dexpbt: Scaling up dexterous manipulation for hand-arm systems with population based training},
  author={Petrenko, Aleksei and Allshire, Arthur and State, Gavriel and Handa, Ankur and Makoviychuk, Viktor},
  journal={arXiv preprint arXiv:2305.12127},
  year={2023}
}

Lift:
@article{singh2024dextrah,
  title={Dextrah-rgb: Visuomotor policies to grasp anything with dexterous hands},
  author={Singh, Ritvik and Allshire, Arthur and Handa, Ankur and Ratliff, Nathan and Van Wyk, Karl},
  journal={arXiv preprint arXiv:2412.01791},
  year={2024}
}

The MDP implementation is heavily inspired by OmniReset:

@inproceedings{yin2026emergent,
  title={Emergent Dexterity via Diverse Resets and Large-Scale Reinforcement Learning},
  author={Patrick Yin and Tyler Westenbroek and Zhengyu Zhang and Joshua Tran and Ignacio Dagnino and Eeshani Shilamkar and Numfor Mbiziwo-Tiapo and Simran Bagaria and Xinlei Liu and Galen Mullins and Andrey Kolobov and Abhishek Gupta},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://arxiv.org/abs/2603.15789},
}


"""
