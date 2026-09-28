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

import argparse

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.utils import instantiate, to_dict
from isaaclab.utils.version import standalone_importers_available

parser = argparse.ArgumentParser(description="Utility to convert a URDF into USD format.")
parser.add_argument("input", type=str, help="The path to the input URDF file.")
parser.add_argument("output", type=str, help="The path to store the USD file.")
parser.add_argument(
    "--ros_package_path",
    nargs=2,
    action="append",
    default=[],
    metavar=("NAME", "PATH"),
    help="Map a ROS package name to its directory for package:// mesh paths. May be repeated.",
)
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

# Prefer kit-less: it skips Kit startup and the kitless visualizers can host the preview.
args_cli.require_kit = not standalone_importers_available()

# Select the concrete physics backend before composing the preview scene.
args_cli.physics = "isaacsim_physx" if args_cli.require_kit else "newton_mjwarp"

import os  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import AssetBaseCfg  # noqa: E402
from isaaclab.scene import InteractiveSceneCfg  # noqa: E402
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg  # noqa: E402
from isaaclab.utils.assets import check_file_path  # noqa: E402
from isaaclab.utils.dict import print_dict  # noqa: E402


def preview(usd_path: str, sim_cfg: sim_utils.SimulationCfg) -> None:
    """Open the converted asset in the visualizer selected on the command line.

    Args:
        usd_path: Path of the generated USD file to display.
        sim_cfg: Simulation config resolved by :func:`~isaaclab.app.launch_simulation`.
    """
    visualizers = sim_cfg.visualizer_cfgs
    visualizers = visualizers if isinstance(visualizers, list) else [visualizers] if visualizers else []
    if not visualizers:
        return

    # The physics backend ingests the USD stage and every visualizer renders the shared scene data,
    # so no backend-specific code is needed here. Physics is not stepped -- the
    # asset is shown in its imported pose until the visualizer window is closed.
    sim = sim_utils.SimulationContext(sim_cfg)
    scene_cfg = InteractiveSceneCfg(num_envs=1, env_spacing=0.0)
    scene_cfg.light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )
    scene_cfg.asset = AssetBaseCfg(prim_path="/World/ConvertedAsset", spawn=sim_utils.UsdFileCfg(usd_path=usd_path))
    _scene = instantiate(scene_cfg)
    sim.reset()

    # Checked per visualizer rather than through ``SimulationContext.is_running``:
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
        ros_package_paths=[{"name": name, "path": os.path.abspath(path)} for name, path in args_cli.ros_package_path],
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
    print_dict(to_dict(urdf_converter_cfg), nesting=0)
    print("-" * 80)
    print("-" * 80)

    sim_cfg = sim_utils.SimulationCfg()
    with launch_simulation(cfg=sim_cfg, launcher_args=args_cli):
        # Create Urdf converter and import the file
        urdf_converter = UrdfConverter(urdf_converter_cfg)
        # print output
        print("URDF importer output:")
        print(f"Generated USD file: {urdf_converter.usd_path}")
        print("-" * 80)
        print("-" * 80)

        preview(urdf_converter.usd_path, sim_cfg)


if __name__ == "__main__":
    main()
