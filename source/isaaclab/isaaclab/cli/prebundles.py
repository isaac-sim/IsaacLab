# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Keep container-provided Isaac Sim packages aligned with the uv environment."""

import os
import shutil
from pathlib import Path

from .utils import (
    extract_isaacsim_path,
    extract_python_exe,
    is_windows,
    print_debug,
    print_info,
    print_warning,
    run_command,
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

When a local ``_isaac_sim`` symlink exists, its ``setup_python_env.sh`` injects
``pip_prebundle`` paths into ``PYTHONPATH``.  These prebundled copies can shadow
the versions installed in the active uv environment (e.g. ``torch+cu128``
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


def repoint_prebundle_packages() -> None:
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

    # Extras are expanded as wheel trees nested below pip_prebundle.
    package_roots = prebundle_dirs | {
        path for prebundle_dir in prebundle_dirs for path in prebundle_dir.glob("*[[]*[]]/*") if path.is_dir()
    }
    repointed = 0
    for package_root in package_roots:
        for pkg_name in _PREBUNDLE_REPOINT_PACKAGES:
            prebundled = package_root / pkg_name
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
            package_root / "torch"
            for package_root in package_roots
            if (package_root / "torch").is_dir() and not (package_root / "torch").is_symlink()
        ]
        if shadowing:
            raise RuntimeError(
                "Failed to neutralize prebundled torch under Isaac Sim; the following would shadow the "
                "pip-installed torch and crash non-isaaclab launches:\n  " + "\n  ".join(str(p) for p in shadowing)
            )
