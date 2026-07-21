# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script sets up the VS Code / Cursor settings for a generated Isaac Lab project.

It generates a ``pyrightconfig.json`` at the project root that adds the Isaac Sim
kit-extension search paths (for ``omni.*``, ``pxr.*``, ``isaacsim.*`` type information) and
the project's own ``source/<project>`` packages to the language server's search paths.

These paths are written to ``pyrightconfig.json`` instead of ``python.analysis.extraPaths``
in ``settings.json`` on purpose: ``pyrightconfig.json`` is read by Pylance (VS Code) and
basedpyright (Cursor) alike, and it does not conflict with a ``[tool.pyright]`` table in
``pyproject.toml`` (which makes VS Code reject ``python.analysis.extraPaths``).

Running this script is optional. For most task authoring it is enough to select the Python
interpreter that has Isaac Lab installed and to run ``pip install -e source/<project>`` -
that already resolves ``isaaclab.*`` and the project's own package. Run this script when you
also want static type information for the Isaac Sim kit extensions.
"""

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys


ISAACLAB_DIR = pathlib.Path(__file__).parents[2]
"""Path to the generated project's root directory."""


def resolve_isaacsim_dir(isaac_path: str | None) -> str:
    """Resolve the Isaac Sim installation directory.

    Resolution order: the ``--isaac_path`` argument, then an ``import isaacsim`` probe using
    the current interpreter, then the ``_isaac_sim`` symlink in the project root.

    Args:
        isaac_path: Explicit Isaac Sim path passed on the command line, or None.

    Returns:
        The resolved Isaac Sim directory, or an empty string if none was found.
    """
    if isaac_path and os.path.exists(isaac_path):
        return isaac_path
    # try to import isaacsim with the current interpreter to discover its install path
    probe = subprocess.run(
        [sys.executable, "-c", "import isaacsim; import os; print(os.environ.get('ISAAC_PATH', ''))"],
        capture_output=True,
        text=True,
        check=False,
        # avoid EULA prompt
        stdin=subprocess.DEVNULL,
    )
    if probe.returncode == 0 and probe.stdout.strip():
        return probe.stdout.strip()
    # fall back to the ``_isaac_sim`` symlink used by binaries installations
    fallback = os.path.join(ISAACLAB_DIR, "_isaac_sim")
    return fallback if os.path.exists(fallback) else ""


def build_extra_paths(isaacsim_dir: str) -> list[str]:
    """Build the Pyright ``extraPaths`` for the project.

    The list combines the Isaac Sim kit-extension paths (parsed from Isaac Sim's own
    ``.vscode/settings.json``) with the project's ``source/<project>`` packages. All paths are
    returned relative to the project root, using forward slashes.

    Args:
        isaacsim_dir: The Isaac Sim installation directory, or an empty string.

    Returns:
        The list of extra search paths.
    """
    path_names: list[str] = []

    # kit-extension paths, parsed from Isaac Sim's own vscode settings
    isaacsim_vscode_filename = os.path.join(isaacsim_dir, ".vscode", "settings.json")
    if isaacsim_dir and os.path.exists(isaacsim_vscode_filename):
        with open(isaacsim_vscode_filename) as f:
            vscode_settings = f.read()
        # extract the contents of the python.analysis.extraPaths section
        match = re.search(
            r"\"python.analysis.extraPaths\": \[.*?\]", vscode_settings, flags=re.MULTILINE | re.DOTALL
        )
        if match:
            body = match.group(0).split('"python.analysis.extraPaths": [')[-1].split("]")[0]
            kit_paths = [p.strip().strip('"') for p in body.split(",")]
            kit_paths = [p for p in kit_paths if p]
            # make the paths relative to the project root
            rel_path = os.path.relpath(isaacsim_dir, ISAACLAB_DIR)
            path_names.extend(os.path.join(rel_path, p) for p in kit_paths)
    else:
        print(
            "[WARN] Could not find Isaac Sim's .vscode/settings.json."
            "\n\tKit-extension paths (omni.*, pxr.*, isaacsim.*) will not be added."
            "\n\tPass --isaac_path <ISAAC_SIM_DIR> to enable them."
        )

    # the project's own source packages
    source_dir = os.path.join(ISAACLAB_DIR, "source")
    if os.path.exists(source_dir):
        path_names.extend(os.path.join("source", ext) for ext in os.listdir(source_dir))

    # normalize to forward slashes so the config is valid on Windows too
    return [p.replace("\\", "/") for p in path_names]


def write_pyright_config(extra_paths: list[str]) -> None:
    """Write ``pyrightconfig.json`` at the project root.

    The config is read by Pylance (VS Code) and basedpyright (Cursor). It takes precedence
    over any ``[tool.pyright]`` table in ``pyproject.toml``, so the reporting defaults below
    keep the dynamically loaded kit extensions from producing missing-import noise.

    Args:
        extra_paths: The extra search paths to add, relative to the project root.
    """
    config = {
        "extraPaths": extra_paths,
        "typeCheckingMode": "basic",
        "reportMissingImports": "none",
        "reportMissingModuleSource": "none",
    }
    pyright_config_filename = os.path.join(ISAACLAB_DIR, "pyrightconfig.json")
    with open(pyright_config_filename, "w") as f:
        f.write(json.dumps(config, indent=4) + "\n")


def overwrite_default_python_interpreter(isaaclab_settings: str) -> str:
    """Overwrite the default python interpreter in the Isaac Lab settings file.

    The default python interpreter is replaced with the path to the python interpreter used by the
    isaac-sim project. This is necessary because the default python interpreter is the one shipped with
    isaac-sim.

    Args:
        isaaclab_settings: The settings string to use as template.

    Returns:
        The settings string with overwritten default python interpreter.
    """
    # read executable name
    python_exe = sys.executable.replace("\\", "/")

    # We make an exception for replacing the default interpreter if the
    # path (/kit/python/bin/python3) indicates that we are using a local/container
    # installation of IsaacSim. We will preserve the calling script as the default, python.sh.
    # We want to use python.sh because it modifies LD_LIBRARY_PATH and PYTHONPATH
    # (among other envars) that we need for all of our dependencies to be accessible.
    if "kit/python/bin/python3" in python_exe:
        return isaaclab_settings
    # replace the default python interpreter in the Isaac Lab settings file with the path to the
    # python interpreter in the Isaac Lab directory
    isaaclab_settings = re.sub(
        r"\"python.defaultInterpreterPath\": \".*?\"",
        f'"python.defaultInterpreterPath": "{python_exe}"',
        isaaclab_settings,
        flags=re.DOTALL,
    )
    # return the Isaac Lab settings file
    return isaaclab_settings


def main():
    parser = argparse.ArgumentParser(description="Set up VS Code / Cursor settings for the project.")
    parser.add_argument("--isaac_path", default=None, help="Absolute path to the Isaac Sim installation.")
    args, _ = parser.parse_known_args()

    # resolve the Isaac Sim directory and write the editor-agnostic pyright config
    isaacsim_dir = resolve_isaacsim_dir(args.isaac_path)
    write_pyright_config(build_extra_paths(isaacsim_dir))

    # Isaac Lab template settings
    isaaclab_vscode_template_filename = os.path.join(ISAACLAB_DIR, ".vscode", "tools", "settings.template.json")
    # make sure the Isaac Lab template settings file exists
    if not os.path.exists(isaaclab_vscode_template_filename):
        raise FileNotFoundError(
            f"Could not find the Isaac Lab template settings file: {isaaclab_vscode_template_filename}"
        )
    # read the Isaac Lab template settings file
    with open(isaaclab_vscode_template_filename) as f:
        isaaclab_template_settings = f.read()

    # overwrite the default python interpreter in the Isaac Lab settings file with the path to the
    # python interpreter used to call this script
    isaaclab_settings = overwrite_default_python_interpreter(isaaclab_template_settings)

    # add template notice to the top of the file
    header_message = (
        "// This file is a template and is automatically generated by the setup_vscode.py script.\n"
        "// Do not edit this file directly.\n"
        "// \n"
        f"// Generated from: {isaaclab_vscode_template_filename}\n"
    )
    isaaclab_settings = header_message + isaaclab_settings

    # write the Isaac Lab settings file
    isaaclab_vscode_filename = os.path.join(ISAACLAB_DIR, ".vscode", "settings.json")
    with open(isaaclab_vscode_filename, "w") as f:
        f.write(isaaclab_settings)

    # copy the launch.json file if it doesn't exist
    isaaclab_vscode_launch_filename = os.path.join(ISAACLAB_DIR, ".vscode", "launch.json")
    isaaclab_vscode_template_launch_filename = os.path.join(ISAACLAB_DIR, ".vscode", "tools", "launch.template.json")
    if not os.path.exists(isaaclab_vscode_launch_filename):
        # read template launch settings
        with open(isaaclab_vscode_template_launch_filename) as f:
            isaaclab_template_launch_settings = f.read()
        # add header
        header_message = header_message.replace(
            isaaclab_vscode_template_filename, isaaclab_vscode_template_launch_filename
        )
        isaaclab_launch_settings = header_message + isaaclab_template_launch_settings
        # write the Isaac Lab launch settings file
        with open(isaaclab_vscode_launch_filename, "w") as f:
            f.write(isaaclab_launch_settings)


if __name__ == "__main__":
    main()
