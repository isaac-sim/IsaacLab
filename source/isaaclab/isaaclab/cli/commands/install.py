# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import shutil
import sys
from pathlib import Path

from ..utils import (
    ISAACLAB_ROOT,
    extract_isaacsim_path,
    extract_python_exe,
    get_pip_command,
    is_arm,
    is_windows,
    print_debug,
    print_info,
    print_warning,
    run_command,
)
from .misc import command_vscode_settings


def _install_system_deps() -> None:
    """install system dependencies"""
    if is_windows():
        return

    # Check if cmake is already installed.
    if shutil.which("cmake"):
        print_info("cmake is already installed.")
    else:
        print_info("Installing system dependencies...")

        # apt-get update
        cmd = ["apt-get", "update"]
        run_command(["sudo"] + cmd if os.geteuid() != 0 else cmd)

        # apt-get install -y --no-install-recommends cmake build-essential
        cmd = [
            "apt-get",
            "install",
            "-y",
            "--no-install-recommends",
            "cmake",
            "build-essential",
        ]
        run_command(["sudo"] + cmd if os.geteuid() != 0 else cmd)

    # On ARM Linux (e.g. DGX Spark), Python dev headers (Python.h) are needed
    # to build C extensions such as quadprog. They are typically pre-installed
    # in x86 Docker images but missing on bare-metal ARM systems.
    if is_arm():
        python_dev_pkg = f"python{sys.version_info.major}.{sys.version_info.minor}-dev"
        try:
            import sysconfig

            if sysconfig.get_path("include") and os.path.isfile(
                os.path.join(sysconfig.get_path("include"), "Python.h")
            ):
                print_info("Python dev headers are already installed.")
            else:
                raise FileNotFoundError
        except (FileNotFoundError, AttributeError):
            print_info(f"Installing {python_dev_pkg} (required for building C extensions on ARM)...")
            cmd = ["apt-get", "update"]
            run_command(["sudo"] + cmd if os.geteuid() != 0 else cmd)
            cmd = [
                "apt-get",
                "install",
                "-y",
                "--no-install-recommends",
                python_dev_pkg,
            ]
            run_command(["sudo"] + cmd if os.geteuid() != 0 else cmd)

        # nlopt has no aarch64 manylinux wheel for the version pinned by
        # isaacteleop[retargeters], so pip falls back to a CMake source build
        # that needs SWIG. Mirrors the apt step in docker/Dockerfile.base.
        if not shutil.which("swig"):
            print_info("Installing swig (required for building nlopt on ARM)...")
            cmd = ["apt-get", "update"]
            run_command(["sudo"] + cmd if os.geteuid() != 0 else cmd)
            cmd = ["apt-get", "install", "-y", "--no-install-recommends", "swig"]
            run_command(["sudo"] + cmd if os.geteuid() != 0 else cmd)


def _torch_first_on_sys_path_is_prebundle(python_exe: str, *, env: dict[str, str]) -> bool:
    """Return True when the first ``torch`` on ``sys.path`` comes from a prebundle directory.

    Checks whether the first directory on ``sys.path`` that contains a
    ``torch`` package lives under a ``pip_prebundle`` path (e.g.
    ``omni.isaac.ml_archive/pip_prebundle``).  This catches the prebundle
    regardless of whether the extension lives under ``exts/``,
    ``extsDeprecated/``, or any other search path.

    Does not import ``torch`` (that can fail on missing ``libcudnn`` while the
    prebundle still appears earlier on ``sys.path`` than ``site-packages``).
    """
    probe = """import os, sys
for p in sys.path:
    if not p:
        continue
    if os.path.isfile(os.path.join(p, "torch", "__init__.py")):
        norm = os.path.normpath(p)
        sys.exit(1 if "pip_prebundle" in norm else 0)
sys.exit(0)
"""
    result = run_command(
        [python_exe, "-c", probe],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 1


def _maybe_preinstall_arm_nlopt(pip_cmd: list[str]) -> None:
    """Pre-install ``nlopt==2.6.2`` on ARM Linux to skip the source-build fallback.

    There is no aarch64 manylinux wheel for the ``nlopt 2.6.2`` version pinned
    by ``isaacteleop[retargeters]``, so pip falls back to a CMake source build
    that hides the host-Python ``numpy`` from its isolated build env. Mirror
    the docker/Dockerfile.base arm64 step: install ``setuptools wheel numpy``
    in the host Python first, then ``--no-build-isolation`` install nlopt so
    later submodule installs see it as already satisfied.
    """
    if is_windows() or not is_arm():
        return
    print_info("Pre-installing nlopt==2.6.2 on ARM (no-build-isolation)...")
    print_info("  step 1/2: ensure setuptools/wheel/numpy are importable for the no-build-isolation backend")
    run_command(pip_cmd + ["install", "setuptools", "wheel", "numpy"])
    print_info("  step 2/2: install nlopt==2.6.2 with --no-build-isolation")
    run_command(pip_cmd + ["install", "--no-build-isolation", "nlopt==2.6.2"])


def _maybe_uninstall_prebundled_torch(
    python_exe: str,
    pip_cmd: list[str],
    using_uv: bool,
    *,
    probe_env: dict[str, str],
) -> None:
    """Uninstall pip torch stack when ``sys.path`` would load ``torch`` from a prebundle first."""
    if not _torch_first_on_sys_path_is_prebundle(python_exe, env=probe_env):
        return
    print_info(
        "The first ``torch`` on ``sys.path`` is under a prebundle directory (e.g. "
        "``omni.isaac.ml_archive/pip_prebundle``). Uninstalling pip "
        "``torch``/``torchvision``/``torchaudio`` before continuing."
    )
    uninstall_flags = ["-y"] if not using_uv else []
    run_command(
        pip_cmd + ["uninstall"] + uninstall_flags + ["torch", "torchvision", "torchaudio"],
        check=False,
    )


# Pinocchio stack required by isaaclab.controllers.pink_ik. Installed via the cmeel
# ``pin`` wheel, which provides the ``pinocchio`` Python module under
# ``cmeel.prefix/lib/python3.12/site-packages/`` and registers it on sys.path via a
# ``cmeel.pth`` hook.
_PINOCCHIO_STACK = ("pin", "pin-pink==3.1.0", "daqp==0.7.2")


def _ensure_pinocchio_installed(python_exe: str, pip_cmd: list[str], *, probe_env: dict[str, str]) -> None:
    """Ensure ``pinocchio`` is importable, force-installing the cmeel pin stack if not.

    Recent Isaac Sim base images preinstall ``pin-pink`` into the kit's bundled
    ``site-packages`` without its ``pin`` (cmeel pinocchio) dependency.  Pip then
    treats the ``pin-pink`` requirement as satisfied and never resolves the
    transitive ``pin`` dep, leaving ``import pinocchio`` broken.  This probes
    for ``pinocchio`` at runtime and force-installs the cmeel stack when needed
    so the pink IK controller and its tests work out of the box.

    Only runs on Linux x86_64 / aarch64 — the same platforms that have
    pinocchio listed in :mod:`isaaclab`'s ``setup.py`` install requirements.
    Skipped on Windows and macOS (no cmeel wheels) and on unsupported
    architectures so the rest of ``--install`` behaves unchanged there.

    A force-reinstall failure (e.g. transient PyPI / NVIDIA Artifactory issue)
    is logged as a warning rather than aborting ``--install``: pinocchio is only
    needed by the optional pink IK controller, so the rest of Isaac Lab should
    still install cleanly.
    """
    import platform

    if platform.system() != "Linux":
        return
    if platform.machine() not in {"x86_64", "AMD64", "aarch64", "arm64"}:
        return

    probe_result = run_command(
        [python_exe, "-c", "import pinocchio"],
        env=probe_env,
        check=False,
        capture_output=True,
        text=True,
    )
    if probe_result.returncode == 0:
        return

    print_info(
        "``import pinocchio`` failed — the kit-bundled ``pin-pink`` likely shipped without its"
        " ``pin`` dep. Force-installing the cmeel pinocchio stack."
    )
    install_result = run_command(
        pip_cmd + ["install", "--upgrade", "--force-reinstall", *_PINOCCHIO_STACK],
        check=False,
    )
    if install_result.returncode != 0:
        print_warning(
            "Force-installing the cmeel pinocchio stack failed (returncode "
            f"{install_result.returncode}). The pink IK controller and its tests will not be"
            " usable until ``pin pin-pink==3.1.0 daqp==0.7.2`` is installed manually."
        )


def _ensure_cuda_torch() -> None:
    """Ensure correct PyTorch and CUDA versions are installed."""
    python_exe = extract_python_exe()
    pip_cmd = get_pip_command(python_exe)
    using_uv = pip_cmd[0] == "uv"

    # Base index for torch.
    base_index = "https://download.pytorch.org/whl"

    # Choose pins per arch.
    torch_ver = "2.10.0"
    tv_ver = "0.25.0"

    if is_arm():
        cuda_ver = "130"
    else:
        cuda_ver = "128"

    cuda_tag = f"cu{cuda_ver}"
    index_url = f"{base_index}/{cuda_tag}"

    want_torch = f"{torch_ver}+{cuda_tag}"

    # Check current torch version using pip show (includes build tags).
    current_ver = ""
    try:
        result = run_command(
            pip_cmd + ["show", "torch"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            for line in result.stdout.split("\n"):
                if line.startswith("Version: "):
                    current_ver = line.split("Version: ", 1)[1].strip()
                    break
    except Exception:
        pass

    # Skip install if version already matches (including CUDA build tag).
    if current_ver == want_torch:
        print_info(f"PyTorch {want_torch} already installed.")
        return

    # Clean install torch.
    print_info(f"Installing torch=={torch_ver} and torchvision=={tv_ver} ({cuda_tag}) from {index_url}...")

    # uv pip uninstall does not accept -y
    uninstall_flags = ["-y"] if not using_uv else []
    run_command(
        pip_cmd + ["uninstall"] + uninstall_flags + ["torch", "torchvision", "torchaudio"],
        check=False,
    )

    run_command(pip_cmd + ["install", "--index-url", index_url, f"torch=={torch_ver}", f"torchvision=={tv_ver}"])


# Isaac Sim install settings.
ISAACSIM_VERSION_SPEC = ">=6.0.0"
ISAACSIM_EXTRAS = "all"
NVIDIA_INDEX_URL = "https://pypi.nvidia.com"


def _install_isaacsim() -> None:
    """Install Isaac Sim pip package if not already present."""
    python_exe = extract_python_exe()
    pip_cmd = get_pip_command(python_exe)

    # Check if already installed.
    result = run_command(
        [python_exe, "-c", "from importlib.metadata import version; print(version('isaacsim'))"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        installed_ver = result.stdout.strip()
        print_info(f"Isaac Sim {installed_ver} already installed.")
        return

    print_info("Installing Isaac Sim...")
    using_uv = pip_cmd[0] == "uv"
    extra_flags = []
    if using_uv:
        # uv needs unsafe-best-match to resolve packages across multiple indexes
        # (isaacsim is on pypi.nvidia.com, its deps are on pypi.org).
        extra_flags = ["--index-strategy", "unsafe-best-match"]

    run_command(
        pip_cmd
        + [
            "install",
            f"isaacsim[{ISAACSIM_EXTRAS}]{ISAACSIM_VERSION_SPEC}",
            "--extra-index-url",
            NVIDIA_INDEX_URL,
        ]
        + extra_flags
    )


# Valid Isaac Lab submodule names that can be passed to --install.
# Each Isaac Lab submodule maps to a source directory named "isaaclab_<name>" under source/.
VALID_ISAACLAB_SUBMODULES: set[str] = {
    "assets",
    "contrib",
    "mimic",
    "newton",
    "ov",
    "physx",
    "rl",
    "tasks",
    "teleop",
    "visualizers",
}

# RL framework names accepted.
# Passing one of these installs all extensions + that framework.
VALID_RL_FRAMEWORKS: set[str] = {"rl_games", "rsl_rl", "sb3", "skrl", "robomimic"}


def _split_install_items(install_type: str) -> list[str]:
    """Split comma-separated install items, ignoring commas inside brackets."""
    parts: list[str] = []
    buf: list[str] = []
    bracket_depth = 0
    for ch in install_type:
        if ch == "[":
            bracket_depth += 1
        elif ch == "]":
            bracket_depth = max(0, bracket_depth - 1)
        if ch == "," and bracket_depth == 0:
            token = "".join(buf).strip()
            if token:
                parts.append(token)
            buf = []
        else:
            buf.append(ch)
    token = "".join(buf).strip()
    if token:
        parts.append(token)
    return parts


def _install_isaaclab_submodules(
    isaaclab_submodules: list[str] | None = None,
    submodule_extras: dict[str, str] | None = None,
    exclude: set[str] | None = None,
) -> None:
    """Install Isaac Lab submodules from the source directory.

    Scans ``source/`` for sub-directories that contain a ``setup.py`` and
    installs each one as an editable pip package.

    Args:
        isaaclab_submodules: Optional, list of source directory names to install.
            If ``None`` is provided, every submodule found under ``source/``
            is installed (subject to *exclude*).
        submodule_extras: Optional mapping from submodule source directory
            name to pip editable selector (e.g.
            ``{"isaaclab_visualizers": "[rerun]"}``).
        exclude: Optional set of source directory names to skip even when
            *isaaclab_submodules* is ``None``.
    """
    python_exe = extract_python_exe()
    source_dir = ISAACLAB_ROOT / "source"

    if not source_dir.exists():
        print_warning(f"Source directory not found: {source_dir}")
        return

    # Collect installable submodules from source/.
    install_items = []
    for item in source_dir.iterdir():
        if not (item.is_dir() and (item / "setup.py").exists()):
            continue
        if isaaclab_submodules is not None and item.name not in isaaclab_submodules:
            continue
        if exclude and item.name in exclude:
            continue
        install_items.append(item)

    # Install order matters for local editable deps:
    # packages like isaaclab_visualizers depend on the local isaaclab package.
    install_items.sort(key=lambda item: (item.name != "isaaclab", item.name))

    pip_cmd = get_pip_command(python_exe)
    for item in install_items:
        print_info(f"Installing submodule: {item.name}")
        editable = (submodule_extras or {}).get(item.name, "")
        install_target = f"{item}{editable}"
        run_command(pip_cmd + ["install", "--editable", install_target])


def _install_extra_frameworks(framework_name: str = "all") -> None:
    """install the python packages for supported reinforcement learning frameworks

    Args:
        framework_name: Framework extra to install (for example ``all`` or ``none``).
    """
    python_exe = extract_python_exe()
    pip_cmd = get_pip_command(python_exe)

    extras = ""
    if framework_name != "none":
        extras = f"[{framework_name}]"

    # Check if specified which rl-framework to install.
    if framework_name == "none":
        print_info("No rl-framework will be installed.")
        return

    print_info(f"Installing rl-framework: {framework_name}")

    # Install the learning frameworks specified.
    run_command(pip_cmd + ["install", "-e", f"{ISAACLAB_ROOT}/source/isaaclab_rl{extras}"])
    run_command(pip_cmd + ["install", "-e", f"{ISAACLAB_ROOT}/source/isaaclab_mimic{extras}"])


_PREBUNDLE_REPOINT_PACKAGES: list[str] = [
    "torch",
    "torchvision",
    "torchaudio",
    "nvidia",
    "newton",
    "newton_actuators",
    "warp",
    "mujoco_warp",
    "websockets",
    "viser",
    "imgui_bundle",
]
"""Package directory names in Isaac Sim prebundle directories to repoint.

When a local ``_isaac_sim`` symlink exists, its ``setup_conda_env.sh`` injects
``pip_prebundle`` paths into ``PYTHONPATH``.  These prebundled copies can shadow
the versions installed in the active conda/uv environment (e.g. ``torch+cu128``
overriding the ``torch+cu130`` the user installed).

After installation we replace each prebundled copy with a symlink that points
back to the environment's ``site-packages``, so the *same* version is loaded
regardless of import path order.
"""


def _repoint_prebundle_packages() -> None:
    """Replace prebundled packages in Isaac Sim with symlinks to the active environment.

    Scans every ``pip_prebundle`` directory under the Isaac Sim installation
    for package directories listed in :data:`_PREBUNDLE_REPOINT_PACKAGES`.
    When the same package exists in the active environment's ``site-packages``,
    the prebundled copy is moved to ``<name>.bak`` and replaced with a symlink.

    This is idempotent — existing symlinks that already point to the correct
    target are left untouched.
    """
    use_symlinks = not is_windows()

    isaacsim_path = extract_isaacsim_path(required=False)
    if isaacsim_path is None or not isaacsim_path.exists():
        print_debug("No Isaac Sim installation found — skipping prebundle repoint.")
        return

    python_exe = extract_python_exe()
    result = run_command(
        [python_exe, "-c", "import site; print(site.getsitepackages()[0])"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        print_warning("Could not determine site-packages path — skipping prebundle repoint.")
        return
    site_packages = Path(result.stdout.strip())
    if not site_packages.is_dir():
        print_warning(f"site-packages directory not found: {site_packages} — skipping prebundle repoint.")
        return

    # Discover pip_prebundle directories from both the Isaac Sim tree and
    # Omniverse cache roots. Some Isaac Sim directories are symlinked into
    # ~/.local/share/ov and may be missed by a plain rglob() on _isaac_sim.
    candidate_roots: set[Path] = set()
    for root in (
        isaacsim_path,
        isaacsim_path.resolve(),
        isaacsim_path / "extscache",
        Path.home() / ".local" / "share" / "ov" / "data" / "exts",
        Path.home() / ".local" / "share" / "ov" / "data" / "exts" / "v2",
    ):
        if root.exists():
            candidate_roots.add(root)
            candidate_roots.add(root.resolve())

    prebundle_dirs: set[Path] = set()
    for root in candidate_roots:
        prebundle_dirs.update(root.rglob("pip_prebundle"))

    if not prebundle_dirs:
        print_debug("No pip_prebundle directories found under Isaac Sim.")
        return

    repointed = 0
    for prebundle_dir in prebundle_dirs:
        for pkg_name in _PREBUNDLE_REPOINT_PACKAGES:
            prebundled = prebundle_dir / pkg_name
            venv_pkg = site_packages / pkg_name

            if not venv_pkg.exists():
                continue
            if not prebundled.exists() and not prebundled.is_symlink():
                continue

            # The 'nvidia' directory is a Python namespace package shared across many
            # distributions (nvidia-cudnn-cu12, nvidia-cublas-cu12, nvidia-srl, …).
            # When using Isaac Sim's built-in Python, site-packages/nvidia only contains
            # 'srl'; replacing the whole prebundle nvidia/ with that symlink strips away
            # the CUDA shared libraries (libcudnn.so.9, etc.) that torch needs.
            # Only repoint the nvidia namespace when the target actually provides the
            # CUDA subpackages (cudnn is the minimal required indicator).
            if pkg_name == "nvidia" and not (venv_pkg / "cudnn").exists():
                print_debug(f"Skipping repoint of {prebundled}: {venv_pkg} lacks CUDA subpackages (cudnn missing).")
                continue

            try:
                if prebundled.is_symlink():
                    if prebundled.resolve() == venv_pkg.resolve():
                        continue
                    prebundled.unlink()
                else:
                    backup = prebundle_dir / f"{pkg_name}.bak"
                    if backup.exists() or backup.is_symlink():
                        shutil.rmtree(backup) if backup.is_dir() else backup.unlink()
                    prebundled.rename(backup)

                if use_symlinks:
                    prebundled.symlink_to(venv_pkg)
                else:
                    shutil.copytree(venv_pkg, prebundled)
                repointed += 1
                print_debug(f"Repointed {prebundled} -> {venv_pkg}")
            except OSError as exc:
                print_warning(f"Could not repoint {prebundled}: {exc} — skipping.")

    if repointed:
        print_info(
            f"Repointed {repointed} prebundled package(s) in Isaac Sim to the active environment's site-packages."
        )
    else:
        print_debug("All prebundled packages already up-to-date — nothing to repoint.")


def command_install(install_type: str = "all") -> None:
    """Install Isaac Lab extensions and optional submodules.

    Args:
        install_type: Comma-separated list of extras to install, or one of the
            special values ``"all"`` / ``"none"``. Extra names match the keys
            in ``source/isaaclab/setup.py``'s ``extras_require``:
            * ``"all"`` (default) — install every extension found under
              ``source/``, plus all RL frameworks.
            * ``"none"`` — install only the "core" ``isaaclab`` package and skip
              RL frameworks.
            * Comma-separated extras, e.g. ``"mimic,assets"`` — install
              only the "core" ``isaaclab`` package plus the listed submodules.
    """

    # Install system dependencies first.
    _install_system_deps()

    # Install the python packages in IsaacLab/source directory.
    print_info("Installing extensions inside the Isaac Lab repository...")
    python_exe = extract_python_exe()

    # Show which environment is being used.
    if os.environ.get("VIRTUAL_ENV"):
        print_info(f"Using uv/venv environment: {os.environ['VIRTUAL_ENV']}")
    elif os.environ.get("CONDA_PREFIX"):
        print_info(f"Using conda environment: {os.environ['CONDA_PREFIX']}")

    print_info(f"Python executable: {python_exe}")

    # Decide which source directories (source/isaaclab/*) to install.
    # "all"        : install everything + all RL frameworks
    # "none"       : core isaaclab only, no RL frameworks
    # RL framework : install everything + only that RL framework (e.g. "skrl")
    # "a,b"        : core + selected submodule directories, no RL frameworks
    install_isaacsim = False

    if install_type == "all":
        isaaclab_submodules = None
        exclude = None
        submodule_extras = {"isaaclab_visualizers": "[all]"}
        framework_type = "all"
    elif install_type == "none":
        isaaclab_submodules = ["isaaclab"]
        exclude = None
        submodule_extras = {}
        framework_type = "none"
    elif install_type in VALID_RL_FRAMEWORKS:
        isaaclab_submodules = None
        exclude = None
        submodule_extras = {"isaaclab_visualizers": "[all]"}
        framework_type = install_type
    else:
        # Parse comma-separated submodule names and RL framework names.
        isaaclab_submodules = ["isaaclab"]  # core is always required
        exclude = None  # explicit selection — no exclusions
        submodule_extras = {}
        framework_type = "none"
        for token in _split_install_items(install_type):
            # Parse optional editable selector: "name[extra1,extra2]"
            if "[" in token:
                bracket_pos = token.index("[")
                name = token[:bracket_pos].strip()
                editable = token[bracket_pos:].strip()
            else:
                name = token.strip()
                editable = ""
            if name == "isaacsim":
                install_isaacsim = True
                continue
            if name in VALID_RL_FRAMEWORKS:
                framework_type = name
                # Ensure isaaclab_rl is installed so the framework extra works.
                if "isaaclab_rl" not in isaaclab_submodules:
                    isaaclab_submodules.append("isaaclab_rl")
                continue
            if name in VALID_ISAACLAB_SUBMODULES:
                pkg_dir = f"isaaclab_{name}"
                if pkg_dir not in isaaclab_submodules:
                    isaaclab_submodules.append(pkg_dir)
                if editable:
                    submodule_extras[pkg_dir] = editable
                # Auto-include the matching visualizer when installing a physics backend.
                if name == "newton" and "isaaclab_visualizers" not in isaaclab_submodules:
                    isaaclab_submodules.append("isaaclab_visualizers")
                    submodule_extras["isaaclab_visualizers"] = "[newton]"
            else:
                valid = sorted(VALID_ISAACLAB_SUBMODULES) + sorted(VALID_RL_FRAMEWORKS) + ["isaacsim"]
                print_warning(f"Unknown Isaac Lab submodule '{name}'. Valid values: {', '.join(valid)}. Skipping.")

    # Configure extra package indexes for NVIDIA and MuJoCo wheels.
    os.environ.setdefault("UV_EXTRA_INDEX_URL", "https://pypi.nvidia.com")
    os.environ.setdefault("PIP_EXTRA_INDEX_URL", "https://pypi.nvidia.com")
    os.environ.setdefault("PIP_FIND_LINKS", "https://py.mujoco.org/")

    # if on ARM arch, temporarily clear LD_PRELOAD
    # LD_PRELOAD is restored below, after installation
    saved_ld_preload = None
    if is_arm() and "LD_PRELOAD" in os.environ:
        print_info("ARM install sandbox: temporarily unsetting LD_PRELOAD for installation.")
        saved_ld_preload = os.environ.pop("LD_PRELOAD")

    # Temporarily filter Isaac Sim pre-bundled package paths from PYTHONPATH during all pip operations.
    # This prevents pip from scanning and managing packages in Isaac Sim's pip_prebundle directories,
    # which can cause those packages to be deleted or modified. This is especially important
    # in conda environments where Isaac Sim setup scripts add these paths to PYTHONPATH.
    saved_pythonpath = None
    filtered_pythonpath = None
    if "PYTHONPATH" in os.environ:
        saved_pythonpath = os.environ["PYTHONPATH"]
        # Filter out any paths containing pip_prebundle (pre-bundled packages that pip shouldn't manage)
        paths = saved_pythonpath.split(os.pathsep)
        filtered_paths = [p for p in paths if p and "pip_prebundle" not in p]

        if len(filtered_paths) != len(paths):
            filtered_pythonpath = os.pathsep.join(filtered_paths)
            os.environ["PYTHONPATH"] = filtered_pythonpath
            filtered_count = len(paths) - len(filtered_paths)
            print_info(
                f"Temporarily filtering {filtered_count} Isaac Sim pre-bundled package path(s) from PYTHONPATH "
                "during pip operations to prevent interference with pre-bundled packages."
            )

    pip_cmd = get_pip_command(python_exe)
    using_uv = pip_cmd[0] == "uv"

    # Probe with the user's original PYTHONPATH (before pip-time filtering) so we detect
    # Isaac Sim's setup_python_env.sh ordering that prefers extsDeprecated/ml_archive.
    probe_env = {**os.environ}
    if saved_pythonpath is not None:
        probe_env["PYTHONPATH"] = saved_pythonpath

    try:
        # Upgrade pip first to avoid compatibility issues (skip when using uv).
        if not using_uv:
            print_info("Upgrading pip...")
            run_command(pip_cmd + ["install", "--upgrade", "pip"])

        # Pin setuptools to avoid issues with pkg_resources removal in 82.0.0.
        run_command(pip_cmd + ["install", "setuptools<82.0.0"])

        # On ARM Linux pre-install nlopt to dodge its from-source build fallback.
        _maybe_preinstall_arm_nlopt(pip_cmd)

        # Drop pip-installed torch if Isaac Sim's deprecated ML prebundle would shadow it.
        _maybe_uninstall_prebundled_torch(python_exe, pip_cmd, using_uv, probe_env=probe_env)

        # Install Isaac Sim if requested.
        if install_isaacsim:
            _install_isaacsim()

        # Install pytorch (version based on arch).
        _ensure_cuda_torch()

        # Install the python modules for the extensions in Isaac Lab.
        _install_isaaclab_submodules(isaaclab_submodules, submodule_extras, exclude)

        # Install the python packages for supported reinforcement learning frameworks.
        print_info("Installing extra requirements such as learning frameworks...")
        _install_extra_frameworks(framework_type)

        # In some rare cases, torch might not be installed properly by setup.py, add one more check here.
        # Can prevent that from happening.
        _ensure_cuda_torch()

        # Ensure ``pinocchio`` is actually importable.  The kit-bundled ``pin-pink`` in recent
        # Isaac Sim images ships without its cmeel ``pin`` dependency, so the transitive
        # requirement from ``pip install -e source/isaaclab`` can be silently skipped.
        _ensure_pinocchio_installed(python_exe, pip_cmd, probe_env=probe_env)

        # Repoint prebundled packages in Isaac Sim to the environment's copies so
        # the active venv/conda versions are always loaded regardless of PYTHONPATH
        # ordering (e.g. torch+cu130 in venv vs torch+cu128 in prebundle on aarch64).
        _repoint_prebundle_packages()

    finally:
        # Restore LD_PRELOAD if we cleared it.
        if saved_ld_preload:
            os.environ["LD_PRELOAD"] = saved_ld_preload
        # Restore PYTHONPATH if we filtered it.
        if saved_pythonpath is not None:
            os.environ["PYTHONPATH"] = saved_pythonpath

    # Install vscode update unless we're in docker.
    if not (os.path.exists("/.dockerenv") or os.path.exists("/run/.containerenv")):
        command_vscode_settings()
