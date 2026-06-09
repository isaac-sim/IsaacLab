# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Helper functions for Isaac Lab controllers.

This module provides utility functions to help with controller implementations.
"""

import contextlib
import glob
import logging
import os
import re
import sys

# import logger
logger = logging.getLogger(__name__)

# NOTE: As of Isaac Sim 6.0, ``isaacsim.robot_motion.lula`` (and ``isaacsim.robot_motion.motion_generation``)
# are deprecated -- ``lula`` has been subsumed by cuMotion (``isaacsim.robot_motion.cumotion``), which is
# built on the new experimental motion-generation API and is the long-term replacement for RMPFlow here.
# ``lula`` is still shipped (under ``extsDeprecated`` in pip installs) so this controller keeps working,

_LULA_EXT_NAME = "isaacsim.robot_motion.lula"
_RMPFLOW_EXT_PREFIX = "rmpflow_ext:"
_RMPFLOW_EXT_NAME = "isaacsim.robot_motion.motion_generation"


def convert_usd_to_urdf(usd_path: str, output_path: str, force_conversion: bool = True) -> tuple[str, str]:
    """Convert a USD file to URDF format.

    Args:
        usd_path: Path to the USD file to convert.
        output_path: Directory to save the converted URDF and mesh files.
        force_conversion: Whether to force the conversion even if the URDF and mesh files already exist.
    Returns:
        A tuple containing the paths to the URDF file and the mesh directory.
    """
    from isaacsim.core.experimental.utils.app import enable_extension

    enable_extension("isaacsim.asset.exporter.urdf")

    from nvidia.srl.from_usd.to_urdf import UsdToUrdf

    usd_to_urdf_kwargs = {
        "node_names_to_remove": None,
        "edge_names_to_remove": None,
        "root": None,
        "parent_link_is_body_1": None,
        "log_level": logging.ERROR,
    }

    urdf_output_dir = os.path.join(output_path, "urdf")
    urdf_file_name = os.path.basename(usd_path).split(".")[0] + ".urdf"
    urdf_output_path = urdf_output_dir + "/" + urdf_file_name
    urdf_meshes_output_dir = os.path.join(output_path, "meshes")

    if not os.path.exists(urdf_output_path) or not os.path.exists(urdf_meshes_output_dir) or force_conversion:
        usd_to_urdf = UsdToUrdf.init_from_file(usd_path, **usd_to_urdf_kwargs)
        os.makedirs(urdf_output_dir, exist_ok=True)
        os.makedirs(urdf_meshes_output_dir, exist_ok=True)

        output_path = usd_to_urdf.save_to_file(
            urdf_output_path=urdf_output_path,
            visualize_collision_meshes=False,
            mesh_dir=urdf_meshes_output_dir,
            mesh_path_prefix="",
        )

        # The current version of the usd to urdf converter creates "inf" effort,
        # This has to be replaced with a max value for the urdf to be valid
        # Open the file for reading and writing
        with open(urdf_output_path) as file:
            # Read the content of the file
            content = file.read()

        # Replace all occurrences of 'inf' with '0'
        content = content.replace("inf", "0.")

        # Open the file again to write the modified content
        with open(urdf_output_path, "w") as file:
            # Write the modified content back to the file
            file.write(content)
    return urdf_output_path, urdf_meshes_output_dir


def change_revolute_to_fixed(urdf_path: str, fixed_joints: list[str], verbose: bool = False):
    """Change revolute joints to fixed joints in a URDF file.

    This function modifies a URDF file by changing specified revolute joints to fixed joints.
    This is useful when you want to disable certain joints in a robot model.

    Args:
        urdf_path: Path to the URDF file to modify.
        fixed_joints: List of joint names to convert from revolute to fixed.
        verbose: Whether to print information about the changes being made.
    """
    with open(urdf_path) as file:
        content = file.read()

    for joint in fixed_joints:
        old_str = f'<joint name="{joint}" type="revolute">'
        new_str = f'<joint name="{joint}" type="fixed">'
        if verbose:
            logger.warning(f"Replacing {joint} with fixed joint")
            logger.warning(old_str)
            logger.warning(new_str)
            if old_str not in content:
                logger.warning(f"Error: Could not find revolute joint named '{joint}' in URDF file")
        content = content.replace(old_str, new_str)

    with open(urdf_path, "w") as file:
        file.write(content)


def change_revolute_to_fixed_regex(urdf_path: str, fixed_joints: list[str], verbose: bool = False):
    """Change revolute joints to fixed joints in a URDF file.

    This function modifies a URDF file by changing specified revolute joints to fixed joints.
    This is useful when you want to disable certain joints in a robot model.

    Args:
        urdf_path: Path to the URDF file to modify.
        fixed_joints: List of regular expressions matching joint names to convert from revolute to fixed.
        verbose: Whether to print information about the changes being made.
    """

    with open(urdf_path) as file:
        content = file.read()

    # Find all revolute joints in the URDF
    revolute_joints = re.findall(r'<joint name="([^"]+)" type="revolute">', content)

    for joint in revolute_joints:
        # Check if this joint matches any of the fixed joint patterns
        should_fix = any(re.match(pattern, joint) for pattern in fixed_joints)

        if should_fix:
            old_str = f'<joint name="{joint}" type="revolute">'
            new_str = f'<joint name="{joint}" type="fixed">'
            if verbose:
                logger.warning(f"Replacing {joint} with fixed joint")
                logger.warning(old_str)
                logger.warning(new_str)
            content = content.replace(old_str, new_str)

    with open(urdf_path, "w") as file:
        file.write(content)


def resolve_rmpflow_path(path: str) -> str:
    """Resolve a sentinel ``rmpflow_ext:`` path to an absolute filesystem path.

    Paths stored in :class:`~isaaclab.controllers.rmp_flow_cfg.RmpFlowControllerCfg`
    that begin with ``"rmpflow_ext:"`` are relative to the
    ``isaacsim.robot_motion.motion_generation`` extension directory.  This avoids
    importing ``isaacsim`` in the cfg file (which is loaded without Kit).
    """
    if path.startswith(_RMPFLOW_EXT_PREFIX):
        rel = path[len(_RMPFLOW_EXT_PREFIX) :]
        # imported lazily so the module loads without Kit (e.g. the kitless Newton visualizer)
        from isaacsim.core.experimental.utils.app import get_extension_path

        ext_dir = get_extension_path(_RMPFLOW_EXT_NAME)
        return os.path.join(ext_dir, rel)
    return path


def find_lula_prebundle_dir() -> str | None:
    """Locate the ``pip_prebundle`` directory shipping the ``lula`` module, or ``None`` if not found.

    ``lula`` is prebundled inside the ``isaacsim.robot_motion.lula`` extension, under the Isaac Sim
    install directory exposed via the ``ISAAC_PATH`` environment variable (importing ``isaacsim`` sets
    it as a side effect). The extension lives under different sub-trees depending on the install:
    ``exts/<name>`` for binary installs, ``extscache/<name>-<version>`` for pip caches, and
    ``extsDeprecated/<name>`` for the Isaac Sim 6.0 pip packages (where the Kit resolver reports it as
    unavailable even though the prebundled module is present). All layouts are searched.
    """
    isaac_path = os.environ.get("ISAAC_PATH")
    if not isaac_path:
        with contextlib.suppress(ImportError):
            import isaacsim  # noqa: F401  (sets ``os.environ["ISAAC_PATH"]`` as a side effect)
        isaac_path = os.environ.get("ISAAC_PATH")
    if not isaac_path:
        return None
    candidates = [os.path.join(isaac_path, "exts", _LULA_EXT_NAME, "pip_prebundle")]
    for parent in ("extscache", "extsDeprecated", "exts"):
        candidates.extend(sorted(glob.glob(os.path.join(isaac_path, parent, f"{_LULA_EXT_NAME}*", "pip_prebundle"))))
    for prebundle in candidates:
        if os.path.isdir(prebundle):
            return prebundle
    return None


def import_lula():
    """Import and return the ``lula`` library, making it importable across backends.

    ``lula`` ships as a prebundled module of the ``isaacsim.robot_motion.lula`` Isaac Sim extension.
    Resolution proceeds in order: import directly when the ``pip_prebundle`` paths are already on
    :data:`sys.path` (e.g. when launched via ``isaaclab.sh``); otherwise locate the prebundle directory
    and add it to :data:`sys.path` (works under both Kit and the kitless Newton visualizer); and only as
    a last resort ask a running Kit app to enable the owning extension. The prebundle is tried before
    :func:`enable_extension` on purpose -- in Isaac Sim 6.0 the extension is deprecated and unresolvable
    by Kit, so enabling it first would log a spurious "failed to resolve extension dependencies" error
    even though ``lula`` itself loads fine from the prebundle.
    """
    try:
        import lula

        return lula
    except ModuleNotFoundError:
        pass

    # Locate the prebundle and add it ourselves -- works under both Kit and kitless, and avoids asking
    # Kit to enable the (in 6.0, deprecated and unresolvable) extension.
    prebundle = find_lula_prebundle_dir()
    if prebundle is not None and prebundle not in sys.path:
        sys.path.insert(0, prebundle)
        try:
            import lula

            return lula
        except ModuleNotFoundError:
            pass

    # Last resort: under a running Kit app, enabling the owning extension registers its prebundle.
    try:
        from isaacsim.core.experimental.utils.app import enable_extension
    except (ImportError, ModuleNotFoundError):
        pass
    else:
        enable_extension(_LULA_EXT_NAME)
        try:
            import lula

            return lula
        except ModuleNotFoundError:
            pass

    raise ModuleNotFoundError(
        "Could not import 'lula', which is required by the RMPFlow controller. It ships with the"
        f" '{_LULA_EXT_NAME}' Isaac Sim extension, which was not found in this Isaac Sim install."
        " The Isaac Sim 6.0 pip packages (early developer release) do not yet include this"
        " extension; use a binary Isaac Sim install, or select a task that does not rely on"
        " RMPFlow."
    )
