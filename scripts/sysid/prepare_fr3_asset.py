# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Prepare the FR3 asset for the sysid fitting env.

The source ``fr3.urdf`` references meshes via ``package://franka_emika_panda/...``
which is not resolvable in this workspace. Meshes are irrelevant for free-air
joint-space sysid (fixed base, no contacts): inertials and joint dynamics are
inline in the URDF. This script strips all <visual>/<collision> elements into
``fr3_nomesh.urdf`` and prints the one-time USD conversion command (URDF→USD
needs Kit, while the Newton backend itself runs kitless — hence offline).
"""

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

ASSETS_DIR = (
    Path(__file__).resolve().parents[2] / "source/isaaclab_tasks/isaaclab_tasks/contrib/sysid/config/franka_fr3/assets"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=str, default=str(ASSETS_DIR / "fr3.urdf"))
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument(
        "--variant",
        type=str,
        default="nomesh",
        choices=["nomesh", "visual"],
        help="nomesh: strip all geometry (fitting). visual: keep visual meshes, resolve package:// (rendering only).",
    )
    parser.add_argument(
        "--mesh-root",
        type=str,
        default=None,
        help=(
            "Directory holding visual/*.dae (e.g. franka_description/meshes/robots/fr3), used to"
            " resolve package:// for the visual variant."
        ),
    )
    args = parser.parse_args()
    if args.variant == "visual" and args.mesh_root is None:
        parser.error("--mesh-root is required for --variant visual")
    out = args.out or str(ASSETS_DIR / f"fr3_{args.variant}.urdf")

    tree = ET.parse(args.urdf)
    removed = rewritten = 0
    for link in tree.getroot().iter("link"):
        # collisions are never needed: free-air sysid, no contacts
        for element in link.findall("collision"):
            link.remove(element)
            removed += 1
        for element in link.findall("visual"):
            if args.variant == "nomesh":
                link.remove(element)
                removed += 1
                continue
            for mesh in element.iter("mesh"):
                name = Path(mesh.get("filename", "")).name
                mesh.set("filename", str(Path(args.mesh_root) / "visual" / name))
                rewritten += 1
    tree.write(out, xml_declaration=True, encoding="unicode")
    print(f"stripped {removed} elements, rewrote {rewritten} mesh paths -> {out}")

    usd_out = ASSETS_DIR / "fr3.usd" / f"fr3_{args.variant}" / f"fr3_{args.variant}.usda"
    print("\nNow convert once (requires Kit / Isaac Sim; keep multi-physics conversion")
    print("enabled so the USD carries the MuJoCo payload the Newton backend consumes):\n")
    print(f"  ./isaaclab.sh -p scripts/tools/convert_urdf.py {out} {usd_out} --fix-base --headless")


if __name__ == "__main__":
    main()
