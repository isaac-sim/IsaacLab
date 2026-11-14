# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Prepare pinned G1 tracking inputs; convert the resulting MJCF with scripts/tools/convert_mjcf.py.

Downloads the retargeted LAFAN1 motions and matching Unitree robot description, retaining the source
revision in a manifest. The datasets retain their original licenses; this script does not redistribute them.
"""

import argparse
import hashlib
import json
import shutil
import tempfile
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco

REVISION = "ce1572906efe6157840e8474d5a0d7aa87481e74"
REPOSITORY = "lvhaidong/LAFAN1_Retargeting_Dataset"


def main() -> None:
    """Download the inputs or reuse a local copy, then compile a free-floating 29-DoF G1 MJCF."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source_dir", type=Path, help="Existing prepared directory with g1_description and lafan1_g1."
    )
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("Output must be empty to keep preparation reproducible.")
    output.mkdir(parents=True, exist_ok=True)
    if args.source_dir:
        source = {"local_directory": str(args.source_dir.resolve())}
        for name in ("g1_description", "lafan1_g1"):
            shutil.copytree(args.source_dir / name, output / name)
    else:
        source = {"repository": REPOSITORY, "revision": REVISION}
        paths = ["robot_description/g1/g1_29dof_rev_1_0.urdf"]
        for directory in ("g1", "robot_description/g1/meshes"):
            with urllib.request.urlopen(
                f"https://huggingface.co/api/datasets/{REPOSITORY}/tree/{REVISION}/{directory}?limit=1000"
            ) as response:
                paths.extend(entry["path"] for entry in json.load(response) if entry["type"] == "file")
        for name in paths:
            relative = Path(name)
            if relative.parts[0] == "g1":
                target = output / "lafan1_g1" / relative.name
            else:
                target = output / "g1_description" / relative.relative_to("robot_description/g1")
            target.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(
                f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}/{name}", target
            )
    description = output / "g1_description"
    robot = ET.parse(description / "g1_29dof_rev_1_0.urdf")
    root = robot.getroot()
    for element in root.findall("mujoco"):
        root.remove(element)
    compiler = ET.SubElement(ET.SubElement(root, "mujoco"), "compiler")
    compiler.attrib.update(
        meshdir=".", balanceinertia="true", discardvisual="false", fusestatic="false", strippath="false"
    )
    with tempfile.NamedTemporaryFile(dir=description, suffix=".urdf") as temporary:
        robot.write(temporary.name)
        model = mujoco.MjModel.from_xml_path(temporary.name)
    mjcf = description / "g1_29dof.xml"
    mujoco.mj_saveLastXML(str(mjcf), model)
    compiled = ET.parse(mjcf)
    pelvis = compiled.find(".//body[@name='pelvis']")
    if pelvis is None:
        raise ValueError("The pinned robot must retain its pelvis body.")
    pelvis.insert(0, ET.Element("freejoint", name="floating_base_joint"))
    compiled.write(mjcf)
    model = mujoco.MjModel.from_xml_path(str(mjcf))
    if model.nq != 36 or model.nv != 35:
        raise ValueError("Expected a free-floating 29-DoF robot.")
    source["sha256"] = {}
    for path in sorted(output.rglob("*")):
        if path.is_file():
            with path.open("rb") as stream:
                source["sha256"][path.relative_to(output).as_posix()] = hashlib.file_digest(
                    stream, "sha256"
                ).hexdigest()
    (output / "source.json").write_text(json.dumps(source, indent=2) + "\n")
    print(f"Prepared {mjcf}; convert it with scripts/tools/convert_mjcf.py before training.")


if __name__ == "__main__":
    main()
