# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The H2 robot of the medical-pad task: the 58 recorded joints, in the order of the reset records."""

from __future__ import annotations

import contextlib
import hashlib
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

from .h2_action_codec import H2_URDF, JOINT_NAMES, urdf_joint_name

with contextlib.suppress(ImportError):
    # Without Kit, the URDF importer registers the USD schemas it applies (``PhysicsJointStateAPI``) when it is first
    # imported. USD fixes its schema registry on first use, so importing the importer after the simulation context has
    # created a stage leaves those schemas unknown and the URDF import fails. This module is imported with the task
    # configuration, before the stage exists. Without an importer installed, spawning the URDF reports it.
    import isaacsim.asset.importer.utils.impl.physx_types  # noqa: F401

H2_JOINT_NAMES = [urdf_joint_name(name) for name in JOINT_NAMES]
"""URDF names of the 58 recorded joints (arms, then hands), in the order of the reset records and the policy."""


def recorded_joints_urdf() -> Path:
    """Return the H2 URDF with every joint outside the 58 recorded ones fixed.

    The robot also has waist, head and leg joints that the records neither measure nor command. Fixing them makes the
    articulation's joints exactly the recorded ones, which is what the world model's joint state and the policy's
    actions index.
    """
    text = H2_URDF.read_text()
    target = Path(tempfile.gettempdir()) / f"isaaclab_cosmos3_h2_{hashlib.sha256(text.encode()).hexdigest()[:12]}.urdf"
    if not target.is_file():
        tree = ET.ElementTree(ET.fromstring(text))
        recorded = set(H2_JOINT_NAMES)
        for joint in tree.getroot().findall("joint"):
            if joint.get("type") != "fixed" and joint.get("name") not in recorded:
                joint.set("type", "fixed")
        tree.write(target)
    return target
