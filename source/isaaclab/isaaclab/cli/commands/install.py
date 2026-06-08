# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import re
import shutil
import sys
from pathlib import Path

import tomllib

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
            if os.geteuid() != 0 and not shutil.which("sudo"):
                print_info(
                    "swig is missing and sudo is unavailable; skipping swig install. "
                    "Pre-install swig in your image if you need to build nlopt from source."
                )
            else:
                print_info("Installing swig (required for building nlopt on ARM)...")
                cmd = ["apt-get", "update"]
                run_command(["sudo"] + cmd if os.geteuid() != 0 else cmd)
                cmd = ["apt-get", "install", "-y", "--no-install-recommends", "swig"]
                run_command(["sudo"] + cmd if os.geteuid() != 0 else cmd)

        # imgui-bundle has no aarch64 manylinux wheel, so pip falls back to a
        # CMake source build that needs GL/X11 dev headers (via glfw).
        # Mirrors the apt step in docker/Dockerfile.base.
        _gl_x11_packages = [
            "libgl1-mesa-dev",
            "libopengl-dev",
            "libglx-dev",
            "libx11-dev",
            "libxcursor-dev",
            "libxi-dev",
            "libxinerama-dev",
            "libxrandr-dev",
        ]
        if not os.path.isfile("/usr/include/X11/Xlib.h"):
            if os.geteuid() != 0 and not shutil.which("sudo"):
                print_info(
                    "GL/X11 dev headers are missing and sudo is unavailable; "
                    "skipping install.  Pre-install " + " ".join(_gl_x11_packages) + " "
                    "if you need to build imgui-bundle from source."
                )
            else:
                print_info("Installing GL/X11 dev headers (required for building imgui-bundle on ARM)...")
                cmd = ["apt-get", "update"]
                run_command(["sudo"] + cmd if os.geteuid() != 0 else cmd)
                cmd = ["apt-get", "install", "-y", "--no-install-recommends", *_gl_x11_packages]
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


# Dependency stack required by isaaclab.controllers.pink_ik. Pinocchio is installed
# via the cmeel ``pin`` wheel, which provides the ``pinocchio`` Python module under
# ``cmeel.prefix/lib/python3.12/site-packages/`` and registers it on sys.path via a
# ``cmeel.pth`` hook. DAQP provides the QP solver selected by the Pink IK controller.
_PINK_IK_STACK = ("pin", "pin-pink==3.1.0", "daqp==0.8.5")


def _ensure_pink_ik_dependencies_installed(python_exe: str, pip_cmd: list[str], *, probe_env: dict[str, str]) -> None:
    """Ensure the Pink IK dependency stack is importable, force-installing it if not.

    Recent Isaac Sim base images preinstall ``pin-pink`` into the kit's bundled
    ``site-packages`` without its ``pin`` (cmeel pinocchio) dependency.  Pip then
    treats the ``pin-pink`` requirement as satisfied and never resolves the
    transitive ``pin`` dep, leaving ``import pinocchio`` broken.  This checks
    the runtime dependencies and force-installs the cmeel stack when needed so
    the pink IK controller and its tests work out of the box.

    Only runs on Linux x86_64 / aarch64 — the same platforms that have
    pinocchio listed in :mod:`isaaclab`'s ``pyproject.toml`` install requirements.
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
        [
            python_exe,
            "-c",
            "import inspect, pinocchio, daqp, qpsolvers; "
            "assert 'daqp' in qpsolvers.available_solvers; "
            "assert 'primal_start' in inspect.signature(daqp.solve).parameters",
        ],
        env=probe_env,
        check=False,
        capture_output=True,
        text=True,
    )
    if probe_result.returncode == 0:
        return

    print_info("Pink IK dependency probe failed. Force-installing the cmeel pinocchio and DAQP stack.")
    install_result = run_command(
        pip_cmd + ["install", "--upgrade", "--force-reinstall", *_PINK_IK_STACK],
        check=False,
    )
    if install_result.returncode != 0:
        print_warning(
            "Force-installing the cmeel pinocchio and DAQP stack failed (returncode "
            f"{install_result.returncode}). The pink IK controller and its tests will not be"
            " usable until ``pin pin-pink==3.1.0 daqp==0.8.5`` is installed manually."
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


def _normalize_package_name(name: str) -> str:
    """Normalize a Python package name for metadata comparisons."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _requirement_name(requirement: str) -> str:
    """Extract the distribution name from a requirement string."""
    requirement = requirement.split(";", 1)[0].strip()
    return re.split(r"\s|<|>|=|!|~|\[|@", requirement, maxsplit=1)[0]


def _get_installed_distribution_requirements(python_exe: str, distribution_name: str) -> list[str]:
    """Return installed ``Requires-Dist`` requirements for a distribution."""
    probe = """import importlib.metadata
import sys

try:
    dist = importlib.metadata.distribution(sys.argv[1])
except importlib.metadata.PackageNotFoundError:
    sys.exit(1)

for requirement in dist.requires or []:
    print(requirement)
"""
    result = run_command(
        [python_exe, "-c", probe, distribution_name],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        print_warning(f"Could not read installed metadata for {distribution_name}; skipping dependency upgrades.")
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _get_extension_pip_upgrade_dependencies(extension_dir: Path) -> list[str]:
    """Read dependency names opted into targeted pip upgrades from ``extension.toml``."""
    extension_toml = extension_dir / "config" / "extension.toml"
    if not extension_toml.is_file():
        return []

    try:
        with extension_toml.open("rb") as fd:
            extension_data = tomllib.load(fd)
    except tomllib.TOMLDecodeError as exc:
        print_warning(f"Could not parse {extension_toml}: {exc}; skipping targeted dependency upgrades.")
        return []

    isaac_lab_settings = extension_data.get("isaac_lab_settings", {})
    if not isinstance(isaac_lab_settings, dict):
        print_warning(
            f"Ignoring invalid isaac_lab_settings in {extension_toml}; expected a table with pip_upgrade_dependencies."
        )
        return []

    upgrade_dependencies = isaac_lab_settings.get("pip_upgrade_dependencies", [])
    if not isinstance(upgrade_dependencies, list) or not all(isinstance(item, str) for item in upgrade_dependencies):
        print_warning(f"Ignoring invalid pip_upgrade_dependencies in {extension_toml}; expected a list of strings.")
        return []

    return upgrade_dependencies


def _get_pip_upgrade_command(pip_cmd: list[str], dependency_name: str, requirement: str) -> list[str]:
    """Return a pip command that upgrades one dependency requirement."""
    if pip_cmd[0] == "uv":
        return pip_cmd + ["install", "--upgrade-package", dependency_name, requirement]
    return pip_cmd + ["install", "--upgrade", requirement]


def _upgrade_extension_pip_dependencies(
    python_exe: str,
    pip_cmd: list[str],
    distribution_name: str,
    dependency_names: list[str],
) -> None:
    """Upgrade selected dependencies using installed distribution metadata requirements."""
    if not dependency_names:
        return

    requirements = _get_installed_distribution_requirements(python_exe, distribution_name)
    seen_dependency_names = set()

    for dependency_name in dependency_names:
        normalized_dependency_name = _normalize_package_name(dependency_name)
        if normalized_dependency_name in seen_dependency_names:
            continue
        seen_dependency_names.add(normalized_dependency_name)

        matching_requirements = [
            req for req in requirements if _normalize_package_name(_requirement_name(req)) == normalized_dependency_name
        ]
        if not matching_requirements:
            print_warning(
                f"Could not find dependency '{dependency_name}' in installed metadata for {distribution_name}; "
                "skipping targeted upgrade."
            )
            continue

        for requirement in matching_requirements:
            print_info(f"Upgrading {dependency_name} for {distribution_name}: {requirement}")
            run_command(_get_pip_upgrade_command(pip_cmd, dependency_name, requirement))


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


# Source directories installed on every ./isaaclab.sh -i invocation (even "core").
# Order matters: isaaclab must be first so dependents resolve against the local copy,
# and isaaclab_ppisp should precede renderer backends (newton/ov/physx) so local
# camera ISP support is available when CameraCfg.isp_cfg is used.
CORE_ISAACLAB_SUBMODULES: list[str] = [
    "isaaclab",
    "isaaclab_ppisp",
    "isaaclab_assets",
    "isaaclab_contrib",
    "isaaclab_experimental",
    "isaaclab_newton",
    "isaaclab_ov",
    "isaaclab_ovphysx",
    "isaaclab_physx",
    "isaaclab_rl",
    "isaaclab_tasks",
    "isaaclab_tasks_experimental",
    "isaaclab_visualizers",
]

# Optional submodules — only installed when explicitly requested or with 'all'.
# Maps the short CLI name to one or more source directory names under source/.
OPTIONAL_ISAACLAB_SUBMODULES: dict[str, tuple[str, ...]] = {
    "mimic": ("isaaclab_teleop", "isaaclab_mimic"),
    "teleop": ("isaaclab_teleop",),
}

# Extra feature sets that install optional heavy dependencies on top of the
# always-installed core submodules. Each name corresponds to one or more
# 'pip install --editable path[extra]' calls against packages already in the
# core set.
VALID_EXTRA_FEATURES: set[str] = {
    "contrib",
    "newton",
    "ov",
    "rl",
    "visualizer",
}

# Extra features excluded from the automatic ``-i all`` / ``-i`` install.
MANUAL_EXTRA_FEATURES: set[str] = {"contrib", "ov"}


def split_install_items(install_type: str) -> list[str]:
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


def _install_isaaclab_submodules(isaaclab_submodules: list[str]) -> None:
    """Install Isaac Lab submodules from the source directory as editable packages.

    Args:
        isaaclab_submodules: Ordered list of source directory names to install
            (e.g. ``["isaaclab", "isaaclab_assets", ...]``). ``isaaclab`` must
            appear first so downstream packages resolve against the local copy.
    """
    python_exe = extract_python_exe()
    source_dir = ISAACLAB_ROOT / "source"

    if not source_dir.exists():
        print_warning(f"Source directory not found: {source_dir}")
        return

    pip_cmd = get_pip_command(python_exe)
    for pkg_name in isaaclab_submodules:
        item = source_dir / pkg_name
        if not item.is_dir() or not ((item / "pyproject.toml").exists() or (item / "setup.py").exists()):
            print_warning(f"Submodule directory not found or missing pyproject.toml: {item}")
            continue
        print_info(f"Installing submodule: {pkg_name}")
        run_command(pip_cmd + ["install", "--editable", str(item)])
        _upgrade_extension_pip_dependencies(
            python_exe,
            pip_cmd,
            pkg_name,
            _get_extension_pip_upgrade_dependencies(item),
        )


def _install_optional_submodule_extra_dependencies(submodule_name: str, selector: str) -> None:
    """Install optional dependency extras for an optional submodule.

    Args:
        submodule_name: One of :data:`OPTIONAL_ISAACLAB_SUBMODULES`.
        selector: Extra selector from a token such as ``mimic[foo]``.
    """
    if not selector:
        return

    print_warning(f"Optional submodule '{submodule_name}' does not support selectors (got '{selector}').")


def _install_contrib_extra_dependencies(selector: str) -> None:
    """Install optional contrib runtime dependencies.

    Args:
        selector: Contrib extra selector, currently ``rlinf``.
    """
    if not selector:
        print_info(
            "Contrib source package is installed with the core submodules. "
            "Use 'contrib[rlinf]' to install contrib runtime dependencies."
        )
        return

    python_exe = extract_python_exe()
    pip_cmd = get_pip_command(python_exe)
    source_dir = ISAACLAB_ROOT / "source"

    print_info(f"Installing contrib optional dependencies: {selector}...")
    run_command(pip_cmd + ["install", "--editable", f"{source_dir}/isaaclab_contrib[{selector}]"])


def _install_ov_extra_dependencies(selector: str) -> None:
    """Install optional OV runtime dependencies.

    Args:
        selector: One or more OV selectors from ``ov[ovrtx]``,
            ``ov[ovphysx]``, or ``ov[all]``.
    """
    if not selector:
        print_info(
            "OV source packages are installed with the core submodules. "
            "Use 'ov[ovrtx]', 'ov[ovphysx]', or 'ov[all]' to install OV runtime dependencies."
        )
        return

    python_exe = extract_python_exe()
    pip_cmd = get_pip_command(python_exe)
    source_dir = ISAACLAB_ROOT / "source"

    selectors = {item.strip().lower() for item in selector.split(",") if item.strip()}
    valid_selectors = {"all", "ovrtx", "ovphysx"}
    unknown_selectors = selectors - valid_selectors
    if unknown_selectors:
        print_warning(
            f"Unknown ov selector(s): {', '.join(sorted(unknown_selectors))}. "
            f"Valid selectors: {', '.join(sorted(valid_selectors))}."
        )
    if "all" in selectors:
        selectors.update({"ovrtx", "ovphysx"})
    if "ovrtx" in selectors:
        print_info("Installing OVRTX optional dependency...")
        run_command(pip_cmd + ["install", "--editable", f"{source_dir}/isaaclab_ov[ovrtx]"])
    if "ovphysx" in selectors:
        print_info("Installing OVPhysX optional dependency...")
        run_command(pip_cmd + ["install", "--editable", f"{source_dir}/isaaclab_ovphysx[ovphysx]"])


def _install_extra_feature(feature_name: str, selector: str = "") -> None:
    """Install optional extra dependencies for a feature set.

    Each feature maps to one or more editable installs with extras applied to
    packages that are already part of the core set.

    Args:
        feature_name: One of :data:`VALID_EXTRA_FEATURES`.
        selector: Optional extra selector (e.g. ``"rsl-rl"`` for
            ``rl[rsl-rl]``). When empty a sensible default is chosen per
            feature (``"all"`` for ``rl`` and ``visualizer``).
    """
    python_exe = extract_python_exe()
    pip_cmd = get_pip_command(python_exe)
    source_dir = ISAACLAB_ROOT / "source"

    if feature_name == "contrib":
        _install_contrib_extra_dependencies(selector)
    elif feature_name == "newton":
        if selector:
            print_warning(f"'newton' does not support selectors (got '{selector}'). Installing all newton extras.")
        print_info("Installing newton extras (newton[sim], PyOpenGL-accelerate, imgui-bundle, typing-extensions)...")
        run_command(pip_cmd + ["install", "--editable", f"{source_dir}/isaaclab_newton[all]"])
        run_command(pip_cmd + ["install", "--editable", f"{source_dir}/isaaclab_physx[newton]"])
        run_command(pip_cmd + ["install", "--editable", f"{source_dir}/isaaclab_visualizers[newton]"])
    elif feature_name == "rl":
        extra = selector if selector else "all"
        print_info(f"Installing RL framework extras: {extra}...")
        run_command(pip_cmd + ["install", "--editable", f"{source_dir}/isaaclab_rl[{extra}]"])
    elif feature_name == "visualizer":
        extra = selector if selector else "all"
        print_info(f"Installing visualizer extras: {extra}...")
        run_command(pip_cmd + ["install", "--editable", f"{source_dir}/isaaclab_visualizers[{extra}]"])
    elif feature_name == "ov":
        _install_ov_extra_dependencies(selector)
    else:
        print_warning(
            f"Unknown extra feature '{feature_name}'. "
            f"Valid features: {', '.join(sorted(VALID_EXTRA_FEATURES))}. Skipping."
        )


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
    "attr",
    "attrs",
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
    """Install Isaac Lab extensions and optional extras.

    All core submodules are always installed. Optional submodules, optional
    submodule extras, and extra feature dependencies are installed based on
    *install_type*.

    Args:
        install_type: Controls which optional submodules and extra feature
            dependencies to install on top of the always-installed core set.

            * ``"all"`` (default) — install core submodules + optional
              submodules (``mimic``, ``teleop``) + all automatic
              extra features.
            * ``"core"`` — install core submodules only; no optional
              submodules, no extra feature dependencies.
            * Comma-separated tokens — install core submodules plus the listed
              optional submodules and extra features. Valid tokens:

              - Optional submodules: ``mimic``, ``teleop``
              - Extra features: ``contrib[rlinf]``, ``newton``, ``rl[<framework>]``,
                ``visualizer[<backend>]``, ``ov[ovrtx|ovphysx|all]``
              - Special: ``isaacsim``

              Examples::

                  ./isaaclab.sh -i newton,rl[rsl-rl]
                  ./isaaclab.sh -i mimic,visualizer[rerun]
                  ./isaaclab.sh -i teleop,rl[skrl],newton
    """

    # Install system dependencies first.
    _install_system_deps()

    print_info("Installing extensions inside the Isaac Lab repository...")
    python_exe = extract_python_exe()

    if os.environ.get("VIRTUAL_ENV"):
        print_info(f"Using uv/venv environment: {os.environ['VIRTUAL_ENV']}")
    elif os.environ.get("CONDA_PREFIX"):
        print_info(f"Using conda environment: {os.environ['CONDA_PREFIX']}")
    print_info(f"Python executable: {python_exe}")

    install_isaacsim = False
    # Always start with the full core set (isaaclab must be first).
    submodules_to_install: list[str] = list(CORE_ISAACLAB_SUBMODULES)
    # List of (feature_name, selector) tuples to apply after the base install.
    extra_features: list[tuple[str, str]] = []
    # List of (submodule_name, selector) tuples for optional submodule extras.
    optional_submodule_extra_dependencies: list[tuple[str, str]] = []

    def append_submodules_once(package_dirs: tuple[str, ...]) -> None:
        for pkg_dir in package_dirs:
            if pkg_dir not in submodules_to_install:
                submodules_to_install.append(pkg_dir)

    # back-compat: "none" is the old name for "core"
    if install_type == "none":
        install_type = "core"

    if install_type == "all":
        for package_dirs in OPTIONAL_ISAACLAB_SUBMODULES.values():
            append_submodules_once(package_dirs)
        extra_features = [(name, "") for name in sorted(VALID_EXTRA_FEATURES - MANUAL_EXTRA_FEATURES)]
    elif install_type == "core":
        # Core only — no optional submodules, no extra features.
        pass
    else:
        for token in split_install_items(install_type):
            if "[" in token:
                bracket_pos = token.index("[")
                name = token[:bracket_pos].strip()
                if "]" not in token:
                    print_warning(f"Malformed install token '{token}': missing closing ']'. Skipping.")
                    continue
                selector = token[bracket_pos + 1 : token.index("]")].strip()
            else:
                name = token.strip()
                selector = ""

            if name == "isaacsim":
                install_isaacsim = True
            elif name in OPTIONAL_ISAACLAB_SUBMODULES:
                append_submodules_once(OPTIONAL_ISAACLAB_SUBMODULES[name])
                if selector:
                    optional_submodule_extra_dependencies.append((name, selector))
            elif name in VALID_EXTRA_FEATURES:
                extra_features.append((name, selector))
            else:
                valid = sorted(OPTIONAL_ISAACLAB_SUBMODULES) + sorted(VALID_EXTRA_FEATURES) + ["isaacsim"]
                print_warning(f"Unknown install token '{name}'. Valid values: {', '.join(valid)}. Skipping.")

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

        # Install all submodules (core set + any explicitly requested optional ones).
        _install_isaaclab_submodules(submodules_to_install)

        # Install requested optional submodule dependency extras.
        if optional_submodule_extra_dependencies:
            print_info("Installing optional submodule dependencies...")
            for submodule_name, selector in optional_submodule_extra_dependencies:
                _install_optional_submodule_extra_dependencies(submodule_name, selector)

        # Install requested extra feature dependencies.
        if extra_features:
            print_info("Installing extra feature dependencies...")
            for feature_name, selector in extra_features:
                _install_extra_feature(feature_name, selector)

        # In some rare cases, torch might not be installed properly by pyproject.toml, add one more check here.
        # Can prevent that from happening.
        _ensure_cuda_torch()

        # Ensure Pink IK's runtime dependencies are actually importable.  The kit-bundled
        # ``pin-pink`` in recent Isaac Sim images can cause transitive dependencies from
        # ``pip install -e source/isaaclab`` to be silently skipped.
        _ensure_pink_ik_dependencies_installed(python_exe, pip_cmd, probe_env=probe_env)

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
