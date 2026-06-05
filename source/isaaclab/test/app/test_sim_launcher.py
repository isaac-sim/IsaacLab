# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse

import pytest

from isaaclab.app.sim_launcher import _ensure_livestream_kit_visualizer


def test_livestream_injects_kit_visualizer_when_missing():
    args = argparse.Namespace(livestream=2, visualizer=None, visualizer_explicit=False)

    _ensure_livestream_kit_visualizer(args)

    assert args.visualizer == ["kit"]


def test_livestream_rejects_disabled_visualizers():
    args = argparse.Namespace(livestream=2, visualizer=None, visualizer_explicit=True)

    with pytest.raises(ValueError, match="Livestreaming requires the Kit visualizer"):
        _ensure_livestream_kit_visualizer(args)
