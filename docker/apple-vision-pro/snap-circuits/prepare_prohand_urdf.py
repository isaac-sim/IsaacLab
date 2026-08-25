#!/usr/bin/env python3

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Add the ProHand MJCF fingertip frames to retargeting-only URDF copies."""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

_TIP_SPECS = {
    "thumb": ("thumb_dist", "0.0045 0.035 0"),
    "index": ("gen1d01_dist", "-0.006 -0.0275 0"),
    "middle": ("gen1d02_dist", "-0.006 -0.0275 0"),
    "ring": ("dist", "-0.006 -0.0275 0"),
    "pinky": ("pinky_dist", "-0.006 -0.0275 0"),
}


def _prepare(repository: Path, side: str, output: Path) -> None:
    tag = "L" if side == "left" else "R"
    word = side.upper()
    source = (
        repository / "assets" / "urdf" / f"gen_1_D_{tag}" / f"ASY__{word}_HAND_GEN1D_ON_FOREARM_GEN1D_optimized.urdf"
    )
    tree = ET.parse(source)
    robot = tree.getroot()
    existing_links = {node.attrib["name"] for node in robot.findall("link")}

    for finger, (parent_suffix, xyz) in _TIP_SPECS.items():
        parent = f"{tag}_{side}_forearm_{parent_suffix}"
        if parent not in existing_links:
            raise ValueError(f"Missing expected ProHand link {parent} in {source}")
        tip = f"{tag}_{finger}_tip"
        ET.SubElement(robot, "link", {"name": tip})
        joint = ET.SubElement(robot, "joint", {"name": f"{tip}_fixed", "type": "fixed"})
        ET.SubElement(joint, "parent", {"link": parent})
        ET.SubElement(joint, "child", {"link": tip})
        ET.SubElement(joint, "origin", {"xyz": xyz, "rpy": "0 0 0"})

    output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(output, encoding="utf-8", xml_declaration=True)
    print(f"Prepared ProHand retargeting URDF: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    args = parser.parse_args()
    # The published URDF mesh paths are relative to ``assets/meshes``. Keep
    # the generated retargeting copies at that root so yourdfpy resolves them
    # without location-dependent absolute paths.
    output_dir = args.repository / "assets" / "meshes"
    _prepare(args.repository, "left", output_dir / "prohand_left_with_tips.urdf")
    _prepare(args.repository, "right", output_dir / "prohand_right_with_tips.urdf")


if __name__ == "__main__":
    main()
