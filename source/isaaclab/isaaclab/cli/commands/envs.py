# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import re
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

from ..utils import (
    ISAACLAB_ROOT,
    determine_python_version,
    is_isaac_sim_source_build,
    is_windows,
    print_debug,
    print_error,
    print_info,
    print_warning,
    run_command,
)


def _reject_downloaded_isaac_sim(environment_type: str) -> None:
    """Reject virtual environments combined with a downloaded Isaac Sim package.

    Args:
        environment_type: User-facing environment type requested by the command.

    Raises:
        SystemExit: If ``_isaac_sim`` is not a marked live source build.
    """
    isaac_sim_path = ISAACLAB_ROOT / "_isaac_sim"
    if isaac_sim_path.exists() and not is_isaac_sim_source_build(isaac_sim_path):
        print_error(f"Downloaded Isaac Sim packages cannot be combined with a {environment_type} environment.")
        print_error(
            "Use the bundled Python through isaaclab.sh/isaaclab.bat, or remove '_isaac_sim' and install "
            "Isaac Sim from pip in the virtual environment."
        )
        raise SystemExit(1)


def _sanitized_conda_env() -> dict[str, str]:
    """
    Return an environment safe for invoking conda after Isaac Sim has added a bunch of env vars.
    Otherwise if there were different python version in the system vs IS python,
    it causes conda to fail with 'SRE mismatch error' due to incompatible python
    stdlib/runtime mix.
    """
    env = dict(os.environ)

    # Prevent mixed Python stdlib/runtime when the CLI is launched from Isaac Sim's bundled Python.
    for key in ("PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP", "PYTHONEXECUTABLE"):
        env.pop(key, None)

    # Isaac Sim injects Kit shared libraries that can interfere with conda's Py runtime.
    env.pop("LD_LIBRARY_PATH", None)

    return env


def _patch_environment_yml(yml_path: str | Path, python_version: str = "3.12") -> str:
    """Return the contents of an environment file with its Python version pin replaced.

    Args:
        yml_path: Path to the source environment file.
        python_version: Python version to inject.

    Returns:
        Patched YML file content.
    """
    content = Path(yml_path).read_text(encoding="utf-8")
    return re.sub(r"python=3\.\d+(?:\.\d+)?", f"python={python_version}", content)


def _get_conda_prefix(env_name: str) -> Path | None:
    """Get the prefix of the conda environment.

    Args:
        env_name: Name of the conda environment.

    Returns:
        Environment path, or ``None`` if it cannot be determined.
    """
    # Use conda run to get sys.prefix
    env = _sanitized_conda_env()
    cmd = ["conda", "run", "-n", env_name, "python", "-c", "import sys; print(sys.prefix)"]
    result = run_command(cmd, capture_output=True, text=True, check=False, env=env)
    if result.returncode == 0:
        return Path(result.stdout.strip())
    return None


def _conda_hook_paths(conda_prefix: Path, activate_name: str, deactivate_name: str) -> tuple[Path, Path]:
    """Return the activation and deactivation hook paths of a conda environment, creating their directories."""
    hooks_dir = conda_prefix / "etc" / "conda"
    activate_hook = hooks_dir / "activate.d" / activate_name
    deactivate_hook = hooks_dir / "deactivate.d" / deactivate_name
    activate_hook.parent.mkdir(parents=True, exist_ok=True)
    deactivate_hook.parent.mkdir(parents=True, exist_ok=True)
    return activate_hook, deactivate_hook


def _write_hooks(activate_hook: Path, activate_content: str, deactivate_hook: Path, deactivate_content: str) -> None:
    """Write an activation/deactivation hook pair."""
    activate_hook.write_text(activate_content, encoding="utf-8")
    deactivate_hook.write_text(deactivate_content, encoding="utf-8")
    print_debug(f"Created activation hook: {activate_hook}")
    print_debug(f"Created deactivation hook: {deactivate_hook}")


def _create_conda_envhooks_shell(conda_prefix: Path) -> None:
    """Write Linux/Mac conda activation/deactivation hooks for Isaac Lab environment variables.

    Args:
        conda_prefix: Prefix path of the target conda environment.
    """
    activate_hook, deactivate_hook = _conda_hook_paths(conda_prefix, "setenv.sh", "unsetenv.sh")
    isaacsim_setup_conda_env_script = ISAACLAB_ROOT / "_isaac_sim" / "setup_conda_env.sh"

    activate_content = textwrap.dedent(
        f"""\
        #!/usr/bin/env bash

        # for Isaac Lab
        : "${{_IL_PREV_PYTHONPATH:=${{PYTHONPATH-}}}}"
        : "${{_IL_PREV_LD_LIBRARY_PATH:=${{LD_LIBRARY_PATH-}}}}"
        export ISAACLAB_PATH="{ISAACLAB_ROOT}"
        alias isaaclab="{ISAACLAB_ROOT / "isaaclab.sh"}"

        # show icon if not running headless
        export RESOURCE_NAME="IsaacSim"

        # for Isaac Sim
        if [ -f "{isaacsim_setup_conda_env_script}" ]; then
            source "{isaacsim_setup_conda_env_script}"
        fi
        """
    )

    deactivate_content = textwrap.dedent(
        f"""\
        #!/usr/bin/env bash

        # for Isaac Lab
        unalias isaaclab &>/dev/null
        unset ISAACLAB_PATH

        # restore paths
        if [ -v _IL_PREV_PYTHONPATH ]; then
            export PYTHONPATH="$_IL_PREV_PYTHONPATH"
            unset _IL_PREV_PYTHONPATH
        fi
        if [ -v _IL_PREV_LD_LIBRARY_PATH ]; then
            export LD_LIBRARY_PATH="$_IL_PREV_LD_LIBRARY_PATH"
            unset _IL_PREV_LD_LIBRARY_PATH
        fi

        # for Isaac Sim
        unset RESOURCE_NAME
        if [ -f "{isaacsim_setup_conda_env_script}" ]; then
            unset CARB_APP_PATH
            unset EXP_PATH
            unset ISAAC_PATH
        fi
        """
    )

    _write_hooks(activate_hook, activate_content, deactivate_hook, deactivate_content)


def _write_torch_gomp_hooks_linux(conda_prefix: Path) -> None:
    """Write Linux-only conda hooks for torch libgomp/libstdc++ LD_PRELOAD handling.

    Args:
        conda_prefix: Path of the target conda environment.
    """
    if not sys.platform.startswith("linux"):
        return

    activate_hook, deactivate_hook = _conda_hook_paths(conda_prefix, "torch_gomp.sh", "torch_gomp_unset.sh")

    activate_content = textwrap.dedent(
        """\
                # Resolve Torch-bundled libgomp and prepend to LD_PRELOAD (quiet + idempotent)
                : "${_IL_PREV_LD_PRELOAD:=${LD_PRELOAD-}}"

                __gomp="$($CONDA_PREFIX/bin/python - <<'PY' 2>/dev/null || true
                import pathlib
                try:
                        import torch
                        p = pathlib.Path(torch.__file__).parent / 'lib' / 'libgomp.so.1'
                        print(p if p.exists() else "", end="")
                except Exception:
                        pass
                PY
                )"

                if [ -n "$__gomp" ] && [ -r "$__gomp" ]; then
                    case ":${LD_PRELOAD:-}:" in
                        *":$__gomp:"*) : ;;
                        *) export LD_PRELOAD="$__gomp${LD_PRELOAD:+:$LD_PRELOAD}";;
                    esac
                fi
                unset __gomp

                # WAR for Ubuntu 22.04: preload conda's libstdc++ to provide CXXABI_1.3.15
                __libstdcxx="$CONDA_PREFIX/lib/libstdc++.so.6"
                if [ -r "$__libstdcxx" ]; then
                    __sys_libstdcxx=$(readlink -f /lib/x86_64-linux-gnu/libstdc++.so.6 \
                        2>/dev/null || echo "/lib/x86_64-linux-gnu/libstdc++.so.6")
                    if [ -r "$__sys_libstdcxx" ] \
                        && ! strings "$__sys_libstdcxx" 2>/dev/null | grep -qE 'CXXABI_1\\.3\\.15'; then
                        case ":${LD_PRELOAD:-}:" in
                            *":$__libstdcxx:"*) : ;;
                            *) export LD_PRELOAD="$__libstdcxx${LD_PRELOAD:+:$LD_PRELOAD}";;
                        esac
                    fi
                    unset __sys_libstdcxx
                fi
                unset __libstdcxx
                """
    )

    deactivate_content = textwrap.dedent(
        """\
                # restore LD_PRELOAD to pre-activation value
                if [ -v _IL_PREV_LD_PRELOAD ]; then
                    export LD_PRELOAD="$_IL_PREV_LD_PRELOAD"
                    unset _IL_PREV_LD_PRELOAD
                fi
                """
    )

    _write_hooks(activate_hook, activate_content, deactivate_hook, deactivate_content)


def _create_conda_envhooks_cmdexe(conda_prefix: Path) -> None:
    """Write Windows cmd.exe conda activation/deactivation hooks.

    Args:
        conda_prefix: Path of the target conda environment.
    """
    activate_hook, deactivate_hook = _conda_hook_paths(conda_prefix, "setenv.bat", "unsetenv.bat")
    isaacsim_setup_conda_env_script = ISAACLAB_ROOT / "_isaac_sim" / "setup_conda_env.bat"

    activate_content = textwrap.dedent(
        f"""\
        @echo off

        REM for Isaac Lab
        if not defined _IL_PREV_PYTHONPATH set "_IL_PREV_PYTHONPATH=%PYTHONPATH%"
        if not defined _IL_PREV_PATH set "_IL_PREV_PATH=%PATH%"
        set "ISAACLAB_PATH={ISAACLAB_ROOT}"
        doskey isaaclab={ISAACLAB_ROOT / "isaaclab.bat"} $*

        REM show icon if not running headless
        set "RESOURCE_NAME=IsaacSim"

        REM for Isaac Sim
        if exist "{isaacsim_setup_conda_env_script}" call "{isaacsim_setup_conda_env_script}"
        """
    )

    deactivate_content = textwrap.dedent(
        """\
        @echo off

        REM for Isaac Lab
        set "ISAACLAB_PATH="
        doskey isaaclab=

        REM restore paths
        if defined _IL_PREV_PYTHONPATH (
            set "PYTHONPATH=%_IL_PREV_PYTHONPATH%"
            set "_IL_PREV_PYTHONPATH="
        )
        if defined _IL_PREV_PATH (
            set "PATH=%_IL_PREV_PATH%"
            set "_IL_PREV_PATH="
        )

        REM for Isaac Sim
        set "RESOURCE_NAME="
        set "CARB_APP_PATH="
        set "EXP_PATH="
        set "ISAAC_PATH="
        """
    )

    _write_hooks(activate_hook, activate_content, deactivate_hook, deactivate_content)


def _create_conda_envhooks_powershell(conda_prefix: Path) -> None:
    """Write Windows PowerShell conda activation/deactivation hooks.

    Args:
        conda_prefix: Path of the target conda environment.
    """
    activate_hook, deactivate_hook = _conda_hook_paths(conda_prefix, "setenv.ps1", "unsetenv.ps1")
    isaacsim_setup_conda_env_script = ISAACLAB_ROOT / "_isaac_sim" / "setup_conda_env.ps1"

    activate_content = textwrap.dedent(
        f"""\
        # for Isaac Lab
        if (-not (Test-Path Env:_IL_PREV_PYTHONPATH)) {{
            $Env:_IL_PREV_PYTHONPATH = $Env:PYTHONPATH
        }}
        if (-not (Test-Path Env:_IL_PREV_PATH)) {{
            $Env:_IL_PREV_PATH = $Env:PATH
        }}
        $Env:ISAACLAB_PATH = "{ISAACLAB_ROOT}"
        Set-Alias -Scope Global isaaclab "{ISAACLAB_ROOT / "isaaclab.bat"}" -Force

        # show icon if not running headless
        $Env:RESOURCE_NAME = "IsaacSim"

        # for Isaac Sim
        if (Test-Path "{isaacsim_setup_conda_env_script}") {{
            . "{isaacsim_setup_conda_env_script}"
        }}
        """
    )

    deactivate_content = textwrap.dedent(
        """\
        # for Isaac Lab
        Remove-Item Alias:isaaclab -ErrorAction SilentlyContinue
        Remove-Item Env:ISAACLAB_PATH -ErrorAction SilentlyContinue

        # restore paths
        if (Test-Path Env:_IL_PREV_PYTHONPATH) {
            $Env:PYTHONPATH = $Env:_IL_PREV_PYTHONPATH
            Remove-Item Env:_IL_PREV_PYTHONPATH -ErrorAction SilentlyContinue
        }
        if (Test-Path Env:_IL_PREV_PATH) {
            $Env:PATH = $Env:_IL_PREV_PATH
            Remove-Item Env:_IL_PREV_PATH -ErrorAction SilentlyContinue
        }

        # for Isaac Sim
        Remove-Item Env:RESOURCE_NAME -ErrorAction SilentlyContinue
        Remove-Item Env:CARB_APP_PATH -ErrorAction SilentlyContinue
        Remove-Item Env:EXP_PATH -ErrorAction SilentlyContinue
        Remove-Item Env:ISAAC_PATH -ErrorAction SilentlyContinue
        """
    )

    _write_hooks(activate_hook, activate_content, deactivate_hook, deactivate_content)


def _write_conda_env_hooks(conda_prefix: Path) -> None:
    """Write conda activation/deactivation hooks for current platform shell(s).

    Args:
        conda_prefix: Path of the target conda environment.
    """
    if is_windows():
        _create_conda_envhooks_cmdexe(conda_prefix)
        _create_conda_envhooks_powershell(conda_prefix)
    else:
        _create_conda_envhooks_shell(conda_prefix)
        _write_torch_gomp_hooks_linux(conda_prefix)


def _append_hook_if_missing(script_path: Path, marker: str, hook_content: str) -> None:
    """Append hook content to a script once, guarded by a marker.

    Args:
        script_path: Activation script file to update.
        marker: Unique marker used to detect an existing hook.
        hook_content: Hook text to append when missing.
    """
    if not script_path.exists():
        print_warning(f"Activation script not found, skipping hook injection: {script_path}")
        return

    content = script_path.read_text(encoding="utf-8")
    if marker in content:
        print_debug(f"Hook already present in: {script_path}")
        return

    if content and not content.endswith("\n"):
        content += "\n"
    content += hook_content
    script_path.write_text(content, encoding="utf-8")
    print_debug(f"Injected hook into: {script_path}")


def _create_uv_envhooks_shell(env_path: Path) -> None:
    """Inject Bash activation hook for uv environments.

    Args:
        env_path: Root path of the uv environment.
    """
    activate_script = env_path / "bin" / "activate"
    isaacsim_setup_conda_env_script = ISAACLAB_ROOT / "_isaac_sim" / "setup_conda_env.sh"

    hook_content = textwrap.dedent(
        f"""\
        # >>> Isaac Lab hook >>>
        export ISAACLAB_PATH="{ISAACLAB_ROOT}"
        alias isaaclab="{ISAACLAB_ROOT / "isaaclab.sh"}"
        export RESOURCE_NAME="IsaacSim"

        if [ -f "{isaacsim_setup_conda_env_script}" ]; then
            . "{isaacsim_setup_conda_env_script}"
        fi
        # <<< Isaac Lab hook <<<
        """
    )

    _append_hook_if_missing(activate_script, "# >>> Isaac Lab hook >>>", hook_content)


def _create_uv_envhooks_cmdexe(env_path: Path) -> None:
    """Inject cmd.exe activation hook for uv environments.

    Args:
        env_path: Root path of the uv environment.
    """
    activate_script = env_path / "Scripts" / "activate.bat"
    isaacsim_setup_conda_env_script = ISAACLAB_ROOT / "_isaac_sim" / "setup_conda_env.bat"

    hook_content = textwrap.dedent(
        f"""\
        REM >>> Isaac Lab hook >>>
        set "ISAACLAB_PATH={ISAACLAB_ROOT}"
        doskey isaaclab={ISAACLAB_ROOT / "isaaclab.bat"} $*
        set "RESOURCE_NAME=IsaacSim"

        if exist "{isaacsim_setup_conda_env_script}" call "{isaacsim_setup_conda_env_script}"
        REM <<< Isaac Lab hook <<<
        """
    )

    _append_hook_if_missing(activate_script, "REM >>> Isaac Lab hook >>>", hook_content)


def _create_uv_envhooks_powershell(env_path: Path) -> None:
    """Inject PowerShell activation hook for uv environments.

    Args:
        env_path: Root path of the uv environment.
    """
    activate_script = env_path / "Scripts" / "Activate.ps1"
    isaacsim_setup_conda_env_script = ISAACLAB_ROOT / "_isaac_sim" / "setup_conda_env.ps1"

    hook_content = textwrap.dedent(
        f"""\
        # >>> Isaac Lab hook >>>
        $Env:ISAACLAB_PATH = "{ISAACLAB_ROOT}"
        Set-Alias -Scope Global isaaclab "{ISAACLAB_ROOT / "isaaclab.bat"}" -Force
        $Env:RESOURCE_NAME = "IsaacSim"

        if (Test-Path "{isaacsim_setup_conda_env_script}") {{
            . "{isaacsim_setup_conda_env_script}"
        }}
        # <<< Isaac Lab hook <<<
        """
    )

    _append_hook_if_missing(activate_script, "# >>> Isaac Lab hook >>>", hook_content)


def _write_uv_env_hooks(env_path: Path) -> None:
    """Write uv activation hooks for current platform shell(s).

    Args:
        env_path: Root path of the uv environment.
    """
    if is_windows():
        _create_uv_envhooks_cmdexe(env_path)
        _create_uv_envhooks_powershell(env_path)
    else:
        _create_uv_envhooks_shell(env_path)


def _isaacsim_pip_package_installed() -> bool:
    """Check whether the current interpreter can import a pip-installed Isaac Sim."""
    result = run_command(
        [sys.executable, "-c", "import isaacsim"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,  # avoid EULA prompt
    )
    return result.returncode == 0


def _warn_isaac_sim_missing(environment_type: str) -> None:
    """Warn that neither a local ``_isaac_sim`` nor a pip-installed Isaac Sim was found."""
    print_warning(f"_isaac_sim symlink not found at {ISAACLAB_ROOT}/_isaac_sim")
    print("\tThis warning can be ignored if you plan to install Isaac Sim via pip.")
    print(f"\tDownloaded Isaac Sim packages use their bundled Python and cannot be combined with {environment_type}.")


def _print_environment_instructions(activate: str, deactivate: str) -> None:
    """Print the next steps after an environment was created."""
    launcher = "isaaclab.bat" if is_windows() else "./isaaclab.sh"
    print(f"\t\t1. To activate the environment, run:                {activate}")
    print(f"\t\t2. To install Isaac Lab extensions, run:            {launcher} -i")
    print(f"\t\t3. To perform formatting, run:                      {launcher} -f")
    print(f"\t\t4. To deactivate the environment, run:              {deactivate}")


def command_setup_conda(env_name: str) -> None:
    """Setup conda environment for Isaac Lab

    Args:
        env_name: Name for the conda environment to create or reuse.
    """
    _reject_downloaded_isaac_sim("conda")

    if not shutil.which("conda"):
        print_error("Conda could not be found. Please install conda and try again.")
        sys.exit(1)

    if not (ISAACLAB_ROOT / "_isaac_sim").exists() and not _isaacsim_pip_package_installed():
        _warn_isaac_sim_missing("conda")

    conda_env = _sanitized_conda_env()
    result = run_command(["conda", "env", "list", "--json"], capture_output=True, text=True, check=False, env=conda_env)
    if f'"{env_name}"' in result.stdout:
        print_info(f"Conda environment named '{env_name}' already exists.")
    else:
        print_info(f"Creating conda environment named '{env_name}'...")
        print_info(f"Installing dependencies from {ISAACLAB_ROOT}/environment.yml")
        # Create from a copy of environment.yml pinned to the Python version Isaac Sim requires.
        temp_yml = ISAACLAB_ROOT / "environment_temp.yml"
        temp_yml.write_text(_patch_environment_yml(ISAACLAB_ROOT / "environment.yml", determine_python_version()))
        try:
            run_command(["conda", "env", "create", "-y", "--file", str(temp_yml), "-n", env_name], env=conda_env)
        finally:
            temp_yml.unlink(missing_ok=True)

    conda_prefix = _get_conda_prefix(env_name)
    if not conda_prefix:
        print_error(f"Could not determine prefix for env {env_name}")
        return

    _write_conda_env_hooks(conda_prefix)

    if not is_windows():
        print_info("Added 'isaaclab' alias and environment hooks to conda activation scripts.")
    print_info(f"Created conda environment named '{env_name}'.\n")
    _print_environment_instructions(f"conda activate {env_name}", "conda deactivate")
    print("\n")


def _check_venv_python_version(env_path: Path, required_ver: str) -> None:
    """Validate that a virtual environment's Python matches the required version.

    Args:
        env_path: Root path of the virtual environment.
        required_ver: Required Python minor version string (e.g. "3.12").

    Raises:
        SystemExit: If the Python version does not match.
    """
    python_exe = env_path / ("Scripts/python.exe" if is_windows() else "bin/python")
    if not python_exe.exists():
        return

    result = run_command(
        [str(python_exe), "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        print_warning(
            f"Could not determine Python version in virtual environment at {env_path}. "
            "The environment may be corrupted."
        )
        return
    actual_ver = result.stdout.strip()
    if actual_ver != required_ver:
        print_error(f"Virtual environment Python is {actual_ver}, but Isaac Sim requires {required_ver}.")
        print_error(
            "Please recreate the environment with the correct Python version, e.g.:\n"
            f"\tuv venv --python {required_ver} {env_path}"
        )
        sys.exit(1)


def command_setup_uv(env_name: str) -> None:
    """setup uv environment for Isaac Lab

    Args:
        env_name: Name for the uv environment directory to create or reuse.
            If a virtual environment is already active (VIRTUAL_ENV is set),
            the active environment is configured for Isaac Lab instead of
            creating a new one.
    """
    _reject_downloaded_isaac_sim("uv")

    if not shutil.which("uv"):
        print_error("uv could not be found. Please install uv and try again.")
        print_error("uv can be installed here:")
        print_error("https://docs.astral.sh/uv/getting-started/installation/")
        sys.exit(1)

    if not (ISAACLAB_ROOT / "_isaac_sim").exists() and not _isaacsim_pip_package_installed():
        _warn_isaac_sim_missing("uv")

    py_ver = determine_python_version()

    # Configure an already active virtual environment instead of creating a new one.
    active_venv = os.environ.get("VIRTUAL_ENV")
    if active_venv:
        env_path = Path(active_venv)
        print_info(f"Detected active virtual environment: {env_path}")
        _check_venv_python_version(env_path, py_ver)
        _write_uv_env_hooks(env_path)
        print_info("Added Isaac Lab environment hooks to the active virtual environment.")
        print_info("Deactivate and reactivate the environment for hooks to take effect:\n")
        if is_windows():
            print("\t\tdeactivate && " + str(env_path / "Scripts" / "activate"))
        else:
            print("\t\tdeactivate && source " + str(env_path / "bin" / "activate"))
        print("\n")
        return

    env_path = ISAACLAB_ROOT / env_name
    if not env_path.exists():
        print_info(f"Creating uv environment named '{env_name}'...")
        run_command(["uv", "venv", "--clear", "--seed", "--python", py_ver, str(env_path)])
    else:
        print_info(f"uv environment '{env_name}' already exists.")
        _check_venv_python_version(env_path, py_ver)

    _write_uv_env_hooks(env_path)
    print_info("Added environment hooks to uv activation scripts.")
    print_info(f"Created uv environment named '{env_name}'.")
    print()
    activate = f"{env_name}\\Scripts\\activate" if is_windows() else f"source {env_name}/bin/activate"
    _print_environment_instructions(activate, "deactivate")
