# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task declarations for the teleoperator's view, separate from learning observations."""

from __future__ import annotations

import math

from isaaclab_ov.renderers import OVRTXRendererCfg

from isaaclab.managers import ObservationPresentationCfg
from isaaclab.utils import configclass

from .ovxr_cfg import OvLibrariesXrCfg


@configclass
class TeleopObservationCfg(ObservationPresentationCfg):
    """An operator-only XR view declared alongside a task's observation groups.

    ``isaaclab play`` activates this declaration when it activates the task's
    :class:`~isaaclab_teleop.TeleopInputCfg`. Ordinary environment construction
    and training ignore it. The observation manager never computes this view,
    adds it to the observation space, or exposes its images to recorders.

    Isaac Capture owns the independent XR rendering loop. Learning cameras must
    use a separate renderer; cameras selected as XR panels are presentation-only.
    """

    presentation: OvLibrariesXrCfg = OvLibrariesXrCfg()
    """Stereo view, optional camera panels, and desktop mirror settings."""

    renderer: OVRTXRendererCfg = OVRTXRendererCfg()
    """Renderer leased exclusively to Capture's XR worker."""

    startup_timeout_s: float = 30.0
    """Maximum wait [s] for the XR render worker to start."""

    def validate_config(self) -> None:
        """Reject invalid startup limits before native resources are created."""
        if not math.isfinite(self.startup_timeout_s) or self.startup_timeout_s <= 0:
            raise ValueError("XR startup_timeout_s must be finite and positive.")
