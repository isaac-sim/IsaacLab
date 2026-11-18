# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The H2 joint mapping on an analytic robot."""

import json

import numpy as np
import pytest

from isaaclab_tasks.contrib.medical_pad_unfold.h2_action_codec import VALID, H2JointMapping, H2JointMappingCfg

SCALE = np.arcsinh(1.0)


@pytest.fixture
def mapping(tmp_path):
    """Two wrists that rotate about z with the first arm joint of each side, and fingertips on fixed offsets."""
    joints = []
    for side in ("left", "right"):
        wrist = f"{side}_hand_C_MC"
        joints.append(
            f'<joint name="{side}_shoulder_pitch_joint" type="revolute">'
            f'<parent link="torso_link"/><child link="{wrist}"/><axis xyz="0 0 1"/></joint>'
        )
        for index, finger in enumerate(("thumb", "index", "middle", "ring", "pinky")):
            joints.append(
                f'<joint name="{side}_{finger}_fixed" type="fixed">'
                f'<parent link="{wrist}"/><child link="{side}_{finger}_fingertip"/>'
                f'<origin xyz="{(index + 1) / 100} 0 0"/></joint>'
            )
    urdf = tmp_path / "robot.urdf"
    urdf.write_text('<robot name="analytic_h2">' + "".join(joints) + "</robot>")
    stats = tmp_path / "stats.json"
    stats.write_text(json.dumps({"global": {"q01": [-1] * 59, "q99": [1] * 59}}))
    return H2JointMapping(H2JointMappingCfg(urdf_path=str(urdf), global_stats_path=str(stats)))


def raw_controls(mapping, targets, start):
    """Undo the normalization, which is arcsinh with unit offset and scale for these statistics."""
    return np.sinh(mapping.encode(targets, start) * SCALE)


@pytest.mark.parametrize("anchor_angle", [0, np.pi / 4])
def test_controls_are_relative_to_the_start_wrist_with_fingertips_in_the_wrist_frame(mapping, anchor_angle):
    targets = np.zeros((16, 58))
    targets[:, 7] = np.pi / 2
    start = np.zeros(58)
    start[7] = anchor_angle
    raw = raw_controls(mapping, targets, start)
    angle = np.pi / 2 - anchor_angle
    rotation = [np.cos(angle), np.sin(angle), 0, -np.sin(angle), np.cos(angle), 0]
    np.testing.assert_allclose(raw[:, 12:18], np.tile(rotation, (16, 1)), atol=1e-6)
    np.testing.assert_allclose(raw[:, 37:43], np.tile([1, 0, 0, 0, 1, 0], (16, 1)), atol=1e-6)
    tips = np.zeros((5, 3))
    tips[:, 1] = np.arange(1, 6) / 100
    np.testing.assert_allclose(raw[:, 18:33], np.tile(tips.ravel(), (16, 1)), atol=1e-6)
    np.testing.assert_allclose(raw[:, 43:58], np.tile(-tips.ravel(), (16, 1)), atol=1e-6)


def test_controls_are_normalized_and_inactive_slots_are_zero(mapping):
    controls = mapping.encode(np.zeros((16, 58)), np.zeros(58))
    assert controls.shape == (16, 59) and controls.dtype == np.float32
    np.testing.assert_allclose(controls[:, 19], np.arcsinh(0.01) / SCALE, rtol=1e-5)
    assert not controls[:, ~VALID].any()
