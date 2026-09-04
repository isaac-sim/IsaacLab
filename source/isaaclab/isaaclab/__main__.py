# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import pathlib
import sys

from isaaclab.utils.vscode import build_extra_paths, resolve_isaacsim_dir, write_pyright_config

VSCODE_SETTINGS_TEMPLATE = """
{
    "editor.rulers": [120],

    // Enables python language server (seems to work slightly better than jedi)
    "python.languageServer": "Pylance",
    "python.jediEnabled": false,

    // This path is automatically filled by isaaclab
    "python.defaultInterpreterPath": "PYTHON.DEFAULTINTERPRETERPATH",

    // Use "black" as a formatter
    "python.formatting.provider": "black",
    "python.formatting.blackArgs": ["--line-length", "120"],

    // Use "flake8" for linting
    "python.linting.pylintEnabled": false,
    "python.linting.flake8Enabled": true
}
"""


def generate_vscode_settings(isaac_path: str | None = None, verbose: bool = False):
    """Generate editor settings and a Pyright configuration in the current workspace.

    Args:
        isaac_path: Explicit Isaac Sim installation path, or None to discover it.
        verbose: Whether to print every generated Pyright search path.
    """
    project_dir = pathlib.Path.cwd()
    vscode_settings_path = project_dir / ".vscode" / "settings.json"
    if vscode_settings_path.exists():
        print(f"VS Code settings already exists: {vscode_settings_path}")
        if input("Overwrite? (y/N): ").lower() not in ["y", "yes"]:
            print("Cancelled: VS Code settings not overwritten")
            return

    isaacsim_dir = resolve_isaacsim_dir(project_dir, isaac_path)
    extra_paths = build_extra_paths(project_dir, isaacsim_dir)
    write_pyright_config(project_dir, extra_paths)
    if verbose:
        for path in extra_paths:
            print(f"Registered Pyright search path: {path}")

    settings = VSCODE_SETTINGS_TEMPLATE.replace("PYTHON.DEFAULTINTERPRETERPATH", pathlib.Path(sys.executable).as_posix())
    vscode_settings_path.parent.mkdir(parents=True, exist_ok=True)
    vscode_settings_path.write_text(settings, encoding="utf-8")
    print(f"VS Code settings generated at {vscode_settings_path}")
    print(f"Pyright configuration generated at {project_dir / 'pyrightconfig.json'}")


def main():
    """Run the installed Isaac Lab CLI while preserving the legacy VS Code option."""
    if len(sys.argv) > 1 and sys.argv[1] == "--generate-vscode-settings":
        parser = argparse.ArgumentParser()
        parser.add_argument("--generate-vscode-settings", action="store_true", help="Generate VS Code settings.")
        parser.add_argument("--isaac_path", help="Absolute path to the Isaac Sim installation.")
        parser.add_argument("--verbose", action="store_true", help="Print discovered extension paths.")
        args = parser.parse_args()
        try:
            if args.isaac_path or args.verbose:
                generate_vscode_settings(isaac_path=args.isaac_path, verbose=args.verbose)
            else:
                generate_vscode_settings()
        except ValueError as error:
            parser.error(str(error))
        return

    from isaaclab.cli import cli

    cli()


if __name__ == "__main__":
    main()
