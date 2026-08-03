# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Utility to convert a URDF into USD format.

Unified Robot Description Format (URDF) is an XML file format used in ROS to describe all elements of
a robot. For more information, see: http://wiki.ros.org/urdf

This script uses the URDF importer API (``isaacsim.asset.importer.urdf``) from Isaac Sim or its standalone
wheel to convert a URDF asset into USD format. It is designed as a convenience script for command-line use.
For more information on the URDF importer, see the documentation for the extension:
https://docs.isaacsim.omniverse.nvidia.com/latest/robot_setup/ext_isaacsim_asset_importer_urdf.html


positional arguments:
  input               The path to the input URDF file.
  output              The path to store the USD file.

optional arguments:
  -h, --help                Show this help message and exit
  --merge_joints            Consolidate links that are connected by fixed joints. (default: False)
  --fix_base                Fix the base to where it is imported. (default: False)
  --joint_stiffness         The stiffness of the joint drive. (default: 100.0)
  --joint_damping           The damping of the joint drive. (default: 1.0)
  --joint_target_type       The type of control to use for the joint drive. (default: "position")

The standard launcher arguments are also accepted. In particular, ``--viz`` previews the converted
asset: ``--viz kit`` opens it in the Isaac Sim viewport, while ``--viz newton`` (or ``rerun`` /
``viser``) opens it kitlessly. Run with ``--help`` for the full list.

"""

"""Parse CLI first so we can decide whether to launch Isaac Sim Kit."""

import argparse
from importlib import metadata

from isaaclab.app import AppLauncher, add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description="Utility to convert a URDF into USD format.")
parser.add_argument("input", type=str, help="The path to the input URDF file.")
parser.add_argument("output", type=str, help="The path to store the USD file.")
parser.add_argument(
    "--merge_joints",
    "--merge-joints",
    action="store_true",
    default=False,
    help="Consolidate links that are connected by fixed joints.",
)
parser.add_argument(
    "--fix_base", "--fix-base", action="store_true", default=False, help="Fix the base to where it is imported."
)
parser.add_argument(
    "--joint_stiffness",
    "--joint-stiffness",
    type=float,
    default=100.0,
    help="The stiffness of the joint drive.",
)
parser.add_argument(
    "--joint_damping",
    "--joint-damping",
    type=float,
    default=1.0,
    help="The damping of the joint drive.",
)
parser.add_argument(
    "--joint_target_type",
    "--joint-target-type",
    type=str,
    default="position",
    choices=["position", "velocity", "none"],
    help="The type of control to use for the joint drive.",
)
add_launcher_args(parser)
args_cli = parser.parse_args()

# The URDF importer ships as a Kit extension unless the standalone importer wheel is installed, so
# Kit is only required when the wheel is absent. With the wheel present the conversion runs kitlessly
# and the kitless visualizers can host the preview.
try:
    metadata.distribution("isaacsim-asset-isolated")
    args_cli.require_kit = False
except metadata.PackageNotFoundError:
    args_cli.require_kit = True

# Report the missing importer before converting anything. Without this the launcher reports only
# that Isaac Sim is absent, which does not mention the wheel that would make this run kitlessly.
if args_cli.require_kit and not AppLauncher.is_available():
    raise ImportError(
        "URDF conversion requires either the full Isaac Sim runtime or the standalone"
        " 'isaacsim-asset-isolated' importer wheel, but neither is installed."
    )

import os  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.physics import PhysicsCfg  # noqa: E402
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg  # noqa: E402
from isaaclab.utils.assets import check_file_path  # noqa: E402
from isaaclab.utils.dict import print_dict  # noqa: E402


def preview(usd_path: str, physics_cfg: PhysicsCfg) -> None:
    """Open the converted asset in the visualizer selected on the command line.

    Args:
        usd_path: Path of the generated USD file to display.
        physics_cfg: Physics config resolved by :func:`~isaaclab.app.launch_simulation`.
    """
    visualizers = args_cli.visualizer or []
    if not visualizers:
        return

    if "kit" in visualizers:
        # a Kit app that resolved without a GUI has no viewport to display the asset in
        if AppLauncher.has_gui():
            sim_utils.show_stage_in_viewport(usd_path)
        return

    # Kitless preview: the physics backend ingests the USD stage and every visualizer renders the
    # shared scene data, so no backend-specific code is needed here. Physics is not stepped -- the
    # asset is shown in its imported pose until the visualizer window is closed.
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device=args_cli.device, physics=physics_cfg))
    light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)
    asset_cfg = sim_utils.UsdFileCfg(usd_path=usd_path)
    asset_cfg.func("/World/ConvertedAsset", asset_cfg)
    sim.reset()

    # Checked per visualizer rather than through ``SimulationContext.is_headless_or_exist_active_visualizer``:
    # that predicate also reports True for an empty visualizer list (headless stepping), and ``render``
    # drops visualizers once they close, so the preview would never exit.
    while any(viz.is_running() and not viz.is_closed for viz in sim.visualizers):
        sim.render()


def main():
    # check valid file path
    urdf_path = args_cli.input
    if not os.path.isabs(urdf_path):
        urdf_path = os.path.abspath(urdf_path)
    if not check_file_path(urdf_path):
        raise ValueError(f"Invalid file path: {urdf_path}")
    # create destination path
    dest_path = args_cli.output
    if not os.path.isabs(dest_path):
        dest_path = os.path.abspath(dest_path)

    # Create Urdf converter config
    # Note: usd_file_name is determined by the URDF importer 3.0 based on the robot name
    # and cannot be overridden. The output is placed under dest_path as usd_dir.
    urdf_converter_cfg = UrdfConverterCfg(
        asset_path=urdf_path,
        usd_dir=dest_path,
        fix_base=args_cli.fix_base,
        merge_fixed_joints=args_cli.merge_joints,
        force_usd_conversion=True,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=args_cli.joint_stiffness,
                damping=args_cli.joint_damping,
            ),
            target_type=args_cli.joint_target_type,
        ),
    )

    # Print info
    print("-" * 80)
    print("-" * 80)
    print(f"Input URDF file: {urdf_path}")
    print("URDF importer config:")
    print_dict(urdf_converter_cfg.to_dict(), nesting=0)
    print("-" * 80)
    print("-" * 80)

    with launch_simulation(cfg=PhysicsCfg(), launcher_args=args_cli) as physics_cfg:
        # Create Urdf converter and import the file
        urdf_converter = UrdfConverter(urdf_converter_cfg)
        # print output
        print("URDF importer output:")
        print(f"Generated USD file: {urdf_converter.usd_path}")
        print("-" * 80)
        print("-" * 80)

        preview(urdf_converter.usd_path, physics_cfg)


if __name__ == "__main__":
    main()
