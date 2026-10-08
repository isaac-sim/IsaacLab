# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Keyboard teleoperation in the viewer window."""

import numpy as np
import torch

# Keys that move the hand along x, y and z and turn it about x, y and z: (positive, negative).
_AXES = ("WS", "AD", "QE", "ZX", "TG", "CV")


def keyboard_action(viewer) -> torch.Tensor:
    """Return the environment action from the keys held in ``viewer``'s window.

    W/S, A/D and Q/E move the hand 1.5 mm per step along x, y and z; Z/X, T/G and C/V turn it 0.02 rad per step; K
    closes and J opens the gripper at 12 mm/s, and releasing them holds its opening.
    """
    action = torch.zeros((1, 7))
    for axis, (positive, negative) in enumerate(_AXES):
        held = int(viewer.is_key_down(positive)) - int(viewer.is_key_down(negative))
        action[0, axis] = (0.0015 if axis < 3 else 0.02) * held
    opening = int(viewer.is_key_down("J")) - int(viewer.is_key_down("K"))
    viewer.keyboard_aperture = float(np.clip(viewer.keyboard_aperture + 0.012 / 30 * opening, 0.0, 0.08))
    action[0, 6] = viewer.keyboard_aperture / 0.04 - 1
    return action
