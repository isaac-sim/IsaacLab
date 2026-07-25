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
    # to build Python packages with native extensions. They are typically
    # pre-installed in x86 Docker images but missing on bare-metal ARM systems.
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


def _ensure_swig_installed() -> bool:
    """Install ``swig`` via apt when missing so the nlopt source build can run.

    Returns:
        ``True`` when this call installed ``swig`` (so the caller is responsible
        for purging it afterwards), ``False`` when ``swig`` was already present or
        could not be installed.
    """
    if shutil.which("swig"):
        return False
    if os.geteuid() != 0 and not shutil.which("sudo"):
        print_warning(
            "swig is required to build nlopt==2.6.2 from source on ARM but is missing and sudo is "
            "unavailable. Pre-install swig (or nlopt==2.6.2) manually; the build below will fail otherwise."
        )
        return False
    print_info("Temporarily installing swig to build nlopt==2.6.2 from source on ARM...")
    update = ["apt-get", "update"]
    run_command(["sudo"] + update if os.geteuid() != 0 else update)
    install = ["apt-get", "install", "-y", "--no-install-recommends", "swig"]
    run_command(["sudo"] + install if os.geteuid() != 0 else install)
    return shutil.which("swig") is not None


def _purge_swig() -> None:
    """Remove the ``swig`` package that was installed for the nlopt build.

    ``swig`` is GPL-licensed and must not be shipped (e.g. in the Docker image),
    so it is purged immediately after nlopt is built. ``nlopt`` is already a
    compiled wheel at this point and does not need ``swig`` at runtime.
    Best-effort: failures are logged but do not abort the install.
    """
    print_info("Removing swig now that nlopt is built (it must not remain installed)...")
    purge = ["apt-get", "purge", "-y", "--auto-remove", "swig"]
    run_command(["sudo"] + purge if os.geteuid() != 0 else purge, check=False)


def _maybe_preinstall_arm_nlopt(python_exe: str, pip_cmd: list[str]) -> None:
    """Pre-install ``nlopt==2.6.2`` on ARM Linux to skip the source-build fallback.

    There is no aarch64 manylinux wheel for the ``nlopt 2.6.2`` version pinned
    by ``isaacteleop[retargeters]``, so pip falls back to a CMake source build
    that hides the host-Python ``numpy`` from its isolated build env. Mirror
    the docker/Dockerfile.base arm64 step: install ``setuptools wheel numpy``
    in the host Python first, then ``--no-build-isolation`` install nlopt so
    later submodule installs see it as already satisfied.

    The source build requires ``swig``. When it is missing it is installed via
    apt only for the duration of the build and purged afterwards, so the
    GPL-licensed ``swig`` package is never left behind — in particular it is
    never shipped in the Docker image. In the Docker build nlopt is pre-installed,
    so this function returns early and never touches ``swig`` (the Dockerfile
    manages its own temporary swig install and purge).
    """
    if is_windows() or not is_arm():
        return

    probe_result = run_command(
        [
            python_exe,
            "-c",
            "import importlib.metadata as metadata; import nlopt; "
            "raise SystemExit(0 if metadata.version('nlopt') == '2.6.2' else 1)",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if probe_result.returncode == 0:
        print_info("nlopt==2.6.2 is already installed on ARM.")
        return

    # The from-source build needs swig; install it only if missing and purge it
    # afterwards so swig is never left behind (it is GPL and must not ship).
    swig_installed_by_us = _ensure_swig_installed()
    try:
        print_info("Pre-installing nlopt==2.6.2 on ARM (no-build-isolation)...")
        print_info("  step 1/2: ensure setuptools/wheel/numpy are importable for the no-build-isolation backend")
        run_command(pip_cmd + ["install", "setuptools", "wheel", "numpy"])
        print_info("  step 2/2: install nlopt==2.6.2 with --no-build-isolation")
        run_command(pip_cmd + ["install", "--no-build-isolation", "nlopt==2.6.2"])
    finally:
        if swig_installed_by_us:
            _purge_swig()


# Packages forming the Pink IK dependency stack. Pinocchio is installed via the
# cmeel ``pin`` wheel, which provides the ``pinocchio`` Python module under
# ``cmeel.prefix/lib/python3.12/site-packages/`` and registers it on sys.path via a
# ``cmeel.pth`` hook. DAQP provides the QP solver selected by the Pink IK controller.
# Versions (e.g. the pink 3.3.x window required by Isaac Sim 6.x) are pinned in the
# root pyproject.toml; :func:`_pink_ik_stack` derives the requirements from there.
_PINK_IK_PACKAGES = ("pin", "pin-pink", "daqp")


def _pink_ik_stack() -> tuple[str, ...]:
    """Return the Pink IK stack requirements pinned in the root ``pyproject.toml``.

    Derives the requirement strings for :data:`_PINK_IK_PACKAGES` from the
    centralized ``[project.dependencies]`` table so the pins live in one place.
    Environment markers are stripped because
    :func:`_ensure_pink_ik_dependencies_installed` gates on platform itself.

    Raises:
        KeyError: If a stack package is missing from the root dependencies.
    """
    dependencies = _load_root_pyproject().get("project", {}).get("dependencies", [])
    requirements = {_requirement_name(r): r.split(";", 1)[0].strip() for r in dependencies}
    missing = [name for name in _PINK_IK_PACKAGES if name not in requirements]
    if missing:
        raise KeyError(f"{missing} missing from [project.dependencies] in the root pyproject.toml.")
    return tuple(requirements[name] for name in _PINK_IK_PACKAGES)


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
    pink_ik_stack = _pink_ik_stack()
    install_result = run_command(
        pip_cmd + ["install", "--upgrade", "--force-reinstall", *pink_ik_stack],
        check=False,
    )
    if install_result.returncode != 0:
        print_warning(
            "Force-installing the cmeel pinocchio and DAQP stack failed (returncode "
            f"{install_result.returncode}). The pink IK controller and its tests will not be"
            f" usable until ``{' '.join(pink_ik_stack)}`` is installed manually."
        )


def _ensure_cuda_torch() -> None:
    """Ensure correct PyTorch and CUDA versions are installed."""
    python_exe = extract_python_exe()
    pip_cmd = get_pip_command(python_exe)
    using_uv = pip_cmd[0] == "uv"

    # Base index for torch.
    base_index = "https://download.pytorch.org/whl"

    # Pinned versions (single source of truth: [tool.isaaclab.versions]).
    torch_ver = _pinned_version("torch")
    tv_ver = _pinned_version("torchvision")

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


def _ensure_newton() -> None:
    """Install the pinned Newton git build, replacing any index version.

    Isaac Sim bundles ``newton[sim]==1.2.0``, which satisfies the loose core bound in
    the root pyproject, so the centralized install would otherwise keep the older
    Newton. Isaac Lab owns the exact commit via ``[tool.uv].override-dependencies``
    (``uv sync`` honors it, ``pip``/``uv pip`` installs do not), so force it in here
    from that single source.
    """
    overrides = _load_root_pyproject().get("tool", {}).get("uv", {}).get("override-dependencies", [])
    requirement = next((r for r in overrides if _requirement_name(r) == "newton"), None)
    if not requirement:
        raise KeyError("Newton git pin is missing from [tool.uv].override-dependencies in the root pyproject.toml.")
    commit = _pinned_version("newton")
    # Newton-matched schemas (isaacsim pins the older ==0.2.0); force it alongside newton.
    schemas = next((r for r in overrides if _requirement_name(r) == "newton-usd-schemas"), None)

    python_exe = extract_python_exe()
    pip_cmd = get_pip_command(python_exe)
    using_uv = pip_cmd[0] == "uv"

    # git installs record the commit in freeze output; skip if it is already present.
    frozen = run_command(pip_cmd + ["freeze"], capture_output=True, text=True, check=False)
    if frozen.returncode == 0 and any(
        _requirement_name(line) == "newton" and commit in line for line in frozen.stdout.splitlines()
    ):
        print_info(f"Newton git build ({commit[:10]}) already installed.")
        return

    print_info(f"Installing pinned Newton git build ({commit[:10]})...")
    uninstall_flags = ["-y"] if not using_uv else []
    run_command(pip_cmd + ["uninstall"] + uninstall_flags + ["newton"], check=False)
    run_command(pip_cmd + ["install", requirement, *([schemas] if schemas else [])])


# Isaac Sim install settings.
NVIDIA_INDEX_URL = "https://pypi.nvidia.com"


def _normalize_package_name(name: str) -> str:
    """Normalize a Python package name for metadata comparisons."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _requirement_name(requirement: str) -> str:
    """Extract the distribution name from a requirement string."""
    requirement = requirement.split(";", 1)[0].strip()
    return re.split(r"\s|<|>|=|!|~|\[|@", requirement, maxsplit=1)[0]


# Distributions installed from the PyTorch index by :func:`_ensure_cuda_torch`;
# excluded from the centralized core-dependency install so they are not pulled
# from PyPI first.
_TORCH_DISTRIBUTIONS = {"torch", "torchvision", "torchaudio"}


def _is_isaaclab_requirement(requirement: str) -> bool:
    """Return True for ``isaaclab*`` self-references (installed as editable submodules)."""
    return _normalize_package_name(_requirement_name(requirement)).startswith("isaaclab")


def _load_root_pyproject() -> dict:
    """Load the root development ``pyproject.toml`` (single source of dependency truth)."""
    with (ISAACLAB_ROOT / "pyproject.toml").open("rb") as fd:
        return tomllib.load(fd)


def _pinned_version(package: str) -> str:
    """Return the pinned version for ``package`` from ``[tool.isaaclab.versions]``.

    This table is the single source of truth for externally-pinned versions; the
    literal pins in the extras and uv constraints mirror it.

    Args:
        package: Key in the ``[tool.isaaclab.versions]`` table (e.g. ``"torch"``).
    """
    versions = _load_root_pyproject().get("tool", {}).get("isaaclab", {}).get("versions", {})
    version = versions.get(package)
    if not version:
        raise KeyError(f"'{package}' is missing from [tool.isaaclab.versions] in the root pyproject.toml.")
    return version


def _isaacsim_requirement() -> str:
    """Return the pinned ``isaacsim`` requirement from the root ``isaacsim`` extra."""
    optional = _load_root_pyproject().get("project", {}).get("optional-dependencies", {})
    requirement = next((r for r in optional.get("isaacsim", []) if _requirement_name(r) == "isaacsim"), None)
    if not requirement:
        raise KeyError(
            "The 'isaacsim' extra is missing from [project.optional-dependencies] in the root pyproject.toml."
        )
    return requirement


def _root_core_dependencies() -> list[str]:
    """Return the third-party core requirements declared in the root pyproject.

    Workspace members (installed as editable submodules) and the torch stack
    (installed by :func:`_ensure_cuda_torch`) are excluded.
    """
    project = _load_root_pyproject().get("project", {})
    dependencies = []
    for requirement in project.get("dependencies", []):
        if _is_isaaclab_requirement(requirement):
            continue
        if _normalize_package_name(_requirement_name(requirement)) in _TORCH_DISTRIBUTIONS:
            continue
        dependencies.append(requirement)
    return dependencies


def _root_extra_dependencies(extra: str) -> list[str]:
    """Return the third-party requirements for a root ``optional-dependencies`` group.

    Workspace member self-references are stripped (the editable submodules are
    installed separately).

    Args:
        extra: Name of the optional-dependency group in the root pyproject.
    """
    optional = _load_root_pyproject().get("project", {}).get("optional-dependencies", {})
    if extra not in optional:
        print_warning(f"Unknown root extra '{extra}'. Available: {', '.join(sorted(optional))}. Skipping.")
        return []
    return [requirement for requirement in optional[extra] if not _is_isaaclab_requirement(requirement)]


def _install_root_extra(extra: str) -> None:
    """Install the third-party dependencies of a root ``optional-dependencies`` group."""
    dependencies = _root_extra_dependencies(extra)
    if not dependencies:
        return
    python_exe = extract_python_exe()
    pip_cmd = get_pip_command(python_exe)
    print_info(f"Installing '{extra}' extra dependencies from the root pyproject...")
    run_command(pip_cmd + ["install"] + dependencies)


def _install_centralized_dependencies(pip_cmd: list[str], optional_submodules: list[str]) -> None:
    """Install the centralized third-party dependencies for the current install.

    The editable sub-packages no longer declare dependencies, so the core
    requirements come from the root pyproject; the runtime extras for any
    requested optional submodules are installed on top.

    Args:
        pip_cmd: Base pip command (e.g. ``["uv", "pip"]`` or ``["python", "-m", "pip"]``).
        optional_submodules: Names of requested optional submodules whose root
            extras should also be installed.
    """
    core_dependencies = _root_core_dependencies()
    if core_dependencies:
        print_info("Installing core dependencies from the root pyproject...")
        run_command(pip_cmd + ["install"] + core_dependencies)
    # dict preserves order while de-duplicating extras shared across submodules.
    extras: dict[str, None] = {}
    for submodule_name in optional_submodules:
        for extra in OPTIONAL_SUBMODULE_ROOT_EXTRAS.get(submodule_name, ()):
            extras.setdefault(extra)
    for extra in extras:
        _install_root_extra(extra)


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
    """Read dependency names opted into targeted pip upgrades from ``[tool.isaaclab]`` in ``pyproject.toml``."""
    pyproject_toml = extension_dir / "pyproject.toml"
    if not pyproject_toml.is_file():
        return []

    try:
        with pyproject_toml.open("rb") as fd:
            project_data = tomllib.load(fd)
    except tomllib.TOMLDecodeError as exc:
        print_warning(f"Could not parse {pyproject_toml}: {exc}; skipping targeted dependency upgrades.")
        return []

    upgrade_dependencies = project_data.get("tool", {}).get("isaaclab", {}).get("pip_upgrade_dependencies", [])
    if not isinstance(upgrade_dependencies, list) or not all(isinstance(item, str) for item in upgrade_dependencies):
        print_warning(f"Ignoring invalid pip_upgrade_dependencies in {pyproject_toml}; expected a list of strings.")
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
            _isaacsim_requirement(),
            "--extra-index-url",
            NVIDIA_INDEX_URL,
        ]
        + extra_flags
    )


# Source directories installed on every ./isaaclab.sh -i invocation (even "core").
# Order must respect inter-package dependencies (topological sort):
#   isaaclab first, then ppisp (no inter-package deps, precedes renderer backends),
#   then contrib (needed by assets), then assets, then tasks (needed by rl),
#   then rl. Packages with only an isaaclab dep can go anywhere after isaaclab.
CORE_ISAACLAB_SUBMODULES: list[str] = [
    "isaaclab",
    "isaaclab_ppisp",
    "isaaclab_contrib",
    "isaaclab_assets",
    "isaaclab_experimental",
    "isaaclab_newton",
    "isaaclab_ov",
    "isaaclab_ovphysx",
    "isaaclab_physx",
    "isaaclab_tasks",
    "isaaclab_tasks_experimental",
    "isaaclab_rl",
    "isaaclab_visualizers",
]

# Optional submodules — only installed when explicitly requested or with 'all'.
# Maps the short CLI name to one or more source directory names under source/.
OPTIONAL_ISAACLAB_SUBMODULES: dict[str, tuple[str, ...]] = {
    "mimic": ("isaaclab_teleop", "isaaclab_mimic"),
    "teleop": ("isaaclab_teleop",),
}

# Root pyproject optional-dependency groups that carry the third-party runtime
# requirements for each optional submodule (the submodules themselves no longer
# declare dependencies). Derived from OPTIONAL_ISAACLAB_SUBMODULES rather than
# redefined: each ``isaaclab_<name>`` source dir maps to the same-named root
# extra (so ``mimic`` pulls in the ``teleop`` stack as well, matching the
# editable-install behavior it replaces). The extra names are validated against
# the root pyproject by :func:`_root_extra_dependencies` at install time.
OPTIONAL_SUBMODULE_ROOT_EXTRAS: dict[str, tuple[str, ...]] = {
    submodule: tuple(directory.removeprefix("isaaclab_") for directory in directories)
    for submodule, directories in OPTIONAL_ISAACLAB_SUBMODULES.items()
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

    print_info(f"Installing contrib optional dependencies: {selector}...")
    _install_root_extra(selector)


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
    # The ov[ovrtx] selector maps to the root 'rtx' extra; ov[ovphysx] to 'ov'.
    if "ovrtx" in selectors:
        print_info("Installing OVRTX optional dependency...")
        _install_root_extra("rtx")
    if "ovphysx" in selectors:
        print_info("Installing OVPhysX optional dependency...")
        _install_root_extra("ov")


def _install_extra_feature(feature_name: str, selector: str = "") -> None:
    """Install optional extra dependencies for a feature set.

    Each feature maps the CLI token to one or more root ``optional-dependencies``
    groups and installs their third-party requirements.

    Args:
        feature_name: One of :data:`VALID_EXTRA_FEATURES`.
        selector: Optional extra selector (e.g. ``"rsl-rl"`` for
            ``rl[rsl-rl]``). When empty a sensible default is chosen per
            feature (``"all"`` for ``rl`` and ``visualizer``).
    """
    if feature_name == "contrib":
        _install_contrib_extra_dependencies(selector)
    elif feature_name == "newton":
        if selector:
            print_warning(f"'newton' does not support selectors (got '{selector}').")
        # The Newton physics engine and its interactive viewer GUI (imgui-bundle,
        # typing-extensions) are part of the base install; this token is a no-op.
        print_info("Newton (engine + viewer) is part of the base install; nothing to install.")
    elif feature_name == "rl":
        extra = selector if selector else "all"
        # rl[all] installs every RL framework extra; other selectors map by name
        # (rsl_rl -> rsl-rl, skrl, sb3, rl-games).
        frameworks = {"sb3", "skrl", "rl-games", "rsl-rl"} if extra == "all" else {extra.replace("_", "-")}
        print_info(f"Installing RL framework extras: {extra}...")
        for framework in sorted(frameworks):
            _install_root_extra(framework)
    elif feature_name == "visualizer":
        extra = selector if selector else "all"
        backends = {"newton", "rerun", "viser"} if extra == "all" else {extra}
        print_info(f"Installing visualizer extras: {extra}...")
        for backend in sorted(backends):
            # 'kit' (Omniverse-provided) and 'newton' (part of the base install)
            # have no extra to install.
            if backend in {"kit", "newton"}:
                continue
            _install_root_extra(backend)
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


def _force_remove(path: Path) -> None:
    """Recursively remove a file, directory, or symlink. A missing path is a no-op.

    Uses absolute-path :func:`os.unlink` / :func:`os.rmdir` rather than the
    ``dir_fd``-relative operations :func:`shutil.rmtree` performs internally. On
    an overlayfs *lower* layer (e.g. inside a Docker image build) the ``dir_fd``
    variant raises ``EINVAL``, whereas the plain ``unlink(2)`` / ``rmdir(2)``
    syscalls create the proper whiteout. This makes prebundle neutralization
    behave identically on a normal filesystem and on an overlayfs lower layer.
    """
    if path.is_symlink() or path.is_file():
        os.unlink(path)
    elif path.is_dir():
        for child in path.iterdir():
            _force_remove(child)
        os.rmdir(path)


def _discover_prebundle_dirs() -> set[Path]:
    """Find every ``pip_prebundle`` directory under the Isaac Sim installation.

    Searches both the Isaac Sim tree and the Omniverse cache roots — some Isaac
    Sim directories are symlinked into ``~/.local/share/ov`` and would be missed
    by a plain ``rglob()`` on ``_isaac_sim``. Returns an empty set when no Isaac
    Sim installation is present.
    """
    isaacsim_path = extract_isaacsim_path(required=False)
    if isaacsim_path is None or not isaacsim_path.exists():
        return set()

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
    return prebundle_dirs


def _find_dangling_prebundle_symlinks() -> set[Path]:
    """Find symlinks under Isaac Sim prebundles whose targets do not resolve.

    Isaac Sim deduplicates packages shared by several extensions as per-file
    symlink farms between ``pip_prebundle`` directories. pip operations routinely
    replace prebundled distributions with copies in ``site-packages`` — harmless
    on its own — but deleting a copy that other prebundles link into leaves
    dangling symlinks that break extension startup at runtime.
    """
    dangling: set[Path] = set()
    for prebundle_dir in _discover_prebundle_dirs():
        for root, _dirs, files in os.walk(prebundle_dir):
            for name in files:
                path = Path(root) / name
                if path.is_symlink() and not path.exists():
                    dangling.add(path)
    return dangling


def _assert_no_new_dangling_prebundle_symlinks(before: set[Path]) -> None:
    """Fail when the installation broke a prebundled package's symlinked ``__init__.py``.

    A new dangling symlink means a pip operation deleted a prebundled package
    that other extensions reference through Isaac Sim's symlink farms — the
    failure mode behind the ``packaging`` removal cascade in nvbugs 6343978
    (14 extensions failing to start). Routine pip replacements do leave a few
    dozen dangling links to files Python never imports at startup (test modules,
    ``WHEEL``/license files, cmake hooks), so only a dangling ``__init__.py`` —
    which makes the whole package unimportable — fails the install; other new
    dangling links are reported as warnings.

    Args:
        before: Dangling symlinks from :func:`_find_dangling_prebundle_symlinks`,
            collected before the pip operations.

    Raises:
        RuntimeError: If the installation left a prebundled package with a
            dangling ``__init__.py``.
    """
    introduced = sorted(_find_dangling_prebundle_symlinks() - before)
    if not introduced:
        return
    broken_packages = [p for p in introduced if p.name == "__init__.py"]
    if broken_packages:
        shown = "\n  ".join(str(p) for p in broken_packages)
        raise RuntimeError(
            f"Installation broke {len(broken_packages)} prebundled package(s) in Isaac Sim"
            f" (dangling __init__.py, {len(introduced)} new dangling symlink(s) total):\n  "
            + shown
            + "\nA pip operation deleted a prebundled package that other Isaac Sim extensions share"
            " via symlinks; extensions will fail to start at runtime (see nvbugs 6343978). This"
            " usually means a dependency pin forced pip to downgrade/replace the prebundled copy —"
            " fix that pin instead of shipping a broken prebundle, and restore the Isaac Sim"
            " installation before retrying."
        )
    print_warning(
        f"Installation left {len(introduced)} new dangling symlink(s) in Isaac Sim prebundles"
        " (no package __init__.py affected — extensions should still start). First few: "
        + ", ".join(str(p) for p in introduced[:5])
    )


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

    prebundle_dirs = _discover_prebundle_dirs()
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
                # Already repointed to the right place — nothing to do.
                if prebundled.is_symlink() and prebundled.resolve() == venv_pkg.resolve():
                    continue
                # Replace the prebundled copy (a stale symlink or a real directory)
                # with a symlink to the active environment. We remove rather than
                # rename-to-``.bak``: the env copy is the symlink target, so the
                # prebundle content is redundant, and renaming a directory on an
                # overlayfs lower layer (Docker image build) fails with ``EXDEV``.
                _force_remove(prebundled)
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

    # Fail loud: a real (non-symlink) prebundled ``torch`` left behind shadows the
    # pip-installed torch on launch paths that do not import ``isaaclab`` (e.g.
    # ``isaac-sim.streaming.sh``), pulling a mismatched NCCL and crashing with
    # ``undefined symbol: ncclDevCommCreate``. Never let that state ship silently.
    # Only relevant when symlinking (Linux); the Windows branch deliberately copies the
    # env package into the prebundle, which is a real directory by design.
    if use_symlinks and (site_packages / "torch").exists():
        shadowing = [
            prebundle_dir / "torch"
            for prebundle_dir in prebundle_dirs
            if (prebundle_dir / "torch").is_dir() and not (prebundle_dir / "torch").is_symlink()
        ]
        if shadowing:
            raise RuntimeError(
                "Failed to neutralize prebundled torch under Isaac Sim; the following would shadow the "
                "pip-installed torch and crash non-isaaclab launches:\n  " + "\n  ".join(str(p) for p in shadowing)
            )


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
              - Extra features: ``contrib[rlinf]``, ``rl[<framework>]``,
                ``visualizer[<backend>]``, ``ov[ovrtx|ovphysx|all]``
              - Special: ``isaacsim``

              Examples::

                  ./isaaclab.sh -i rl[rsl-rl]
                  ./isaaclab.sh -i mimic,visualizer[rerun]
                  ./isaaclab.sh -i teleop,rl[skrl],ov[ovrtx]
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
    # Names of requested optional submodules (used to install their root extras).
    requested_optional_submodules: list[str] = []

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
        requested_optional_submodules = list(OPTIONAL_ISAACLAB_SUBMODULES)
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
                requested_optional_submodules.append(name)
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

    # Baseline for the post-install integrity check: no pip operation below may
    # leave new dangling symlinks in Isaac Sim's prebundles (nvbugs 6343978).
    dangling_symlinks_before = _find_dangling_prebundle_symlinks()

    try:
        # Upgrade pip first to avoid compatibility issues (skip when using uv).
        if not using_uv:
            print_info("Upgrading pip...")
            run_command(pip_cmd + ["install", "--upgrade", "pip"])

        # Pin setuptools to avoid issues with pkg_resources removal in 82.0.0.
        run_command(pip_cmd + ["install", "setuptools<82.0.0"])

        # On ARM Linux pre-install nlopt to dodge its from-source build fallback.
        _maybe_preinstall_arm_nlopt(python_exe, pip_cmd)

        # Drop pip-installed torch if Isaac Sim's deprecated ML prebundle would shadow it.
        _maybe_uninstall_prebundled_torch(python_exe, pip_cmd, using_uv, probe_env=probe_env)

        # Install Isaac Sim if requested.
        if install_isaacsim:
            _install_isaacsim()

        # Install pytorch (version based on arch).
        _ensure_cuda_torch()

        # Install all submodules (core set + any explicitly requested optional ones).
        _install_isaaclab_submodules(submodules_to_install)

        # The submodules no longer declare third-party dependencies; install the
        # centralized core requirements (and optional-submodule extras) from the
        # root pyproject. torch is excluded — it is handled by _ensure_cuda_torch.
        _install_centralized_dependencies(pip_cmd, requested_optional_submodules)

        # Isaac Sim's bundled newton==1.2.0 satisfies the loose core bound, so force the
        # pinned Newton git build (the default physics engine) over it.
        _ensure_newton()

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

        # Fail loud if any pip operation above broke Isaac Sim's cross-extension
        # symlink farms. Prebundle deletions on their own are routine (pip
        # replaces those packages in site-packages, which shadows the prebundle
        # at runtime); only newly dangling symlinks break extension startup.
        _assert_no_new_dangling_prebundle_symlinks(dangling_symlinks_before)

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
