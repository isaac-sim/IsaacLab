# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""On-demand installer for the RLinf VLA post-training demo.

The demo needs three things that ``pyproject.toml`` cannot express: package pins that must bypass the
dependency resolver, a source checkout of Isaac-GR00T, and a pretrained checkpoint. Baking them into the
Isaac Lab container image would cost every user the download, so they are fetched here instead, only when
the demo is actually run.

Install the root ``rlinf`` extra first, then run this script::

    ./isaaclab.sh -i contrib[rlinf]
    ./isaaclab.sh -p scripts/reinforcement_learning/rlinf/setup_rlinf.py

Every step is idempotent and skips work that is already done, so the script can be re-run to repair a
partial install. Afterwards, train and evaluate with::

    ./isaaclab.sh train --rl_library rlinf --config_name isaaclab_ppo_gr00t_assemble_trocar
    ./isaaclab.sh play --rl_library rlinf --config_name isaaclab_ppo_gr00t_assemble_trocar --video
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import subprocess
from pathlib import Path

from isaaclab.cli.utils import ISAACLAB_ROOT, get_pip_command, print_info, print_warning, run_command

# Pins validated against the reference run of the assemble_trocar demo. RLinf and GR00T are installed with
# ``--no-deps`` because their own pins clash with the Isaac Sim ones already resolved in the environment.
RLINF_VERSION = "0.2.0dev2"
# Matches GR00T's own pin. Newer transformers drop ``VideoInput`` from ``transformers.image_utils``,
# which the checkpoint's remote processing code imports.
TRANSFORMERS_VERSION = "4.51.3"
TOKENIZERS_SPEC = "tokenizers>=0.21,<0.22"
FLASH_ATTN_VERSION = "2.8.3"
# Required: gr00t.data.transform.state_action imports pytorch3d.transforms. RLinf pins the
# ``pipablepytorch3d`` wheel, but that distribution caps out at Python 3.11, so on the Python 3.12
# interpreter Isaac Lab ships this is built from source instead, as the GR00T docs already do.
PYTORCH3D_SPEC = "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.9"

GR00T_REPO_URL = "https://github.com/NVIDIA/Isaac-GR00T.git"
# Pinned: later commits restructure ``gr00t.experiment``, dropping the ``data_config`` module that
# the task's gr00t_config.py imports.
GR00T_COMMIT = "4af2b622892f7dcb5aae5a3fb70bcb02dc217b96"

CHECKPOINT_REPO_ID = "nvidia/Assemble_Trocar"

DEFAULT_GR00T_DIR = ISAACLAB_ROOT.parent / "Isaac-GR00T"
DEFAULT_CHECKPOINT_DIR = ISAACLAB_ROOT / ".pretrained_checkpoints" / "rlinf" / "Assemble_Trocar"
NO_FLASH_ATTN_PATCH = ISAACLAB_ROOT / "scripts/imitation_learning/locomanipulation_sdg/gr00t/no_flash_attn.patch"

# Marker packages from the root ``rlinf`` extra, used to detect that ``-i contrib[rlinf]`` has not run.
RLINF_EXTRA_MARKERS = ("ray", "decord", "peft", "diffusers", "timm")


def is_importable(module: str) -> bool:
    """Return whether a module can be resolved without importing it.

    Args:
        module: Top-level module name to look up.

    Returns:
        Whether the module was found on the import path.
    """
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def installed_version(distribution: str) -> str | None:
    """Return the installed version of a distribution, or ``None`` when absent.

    Args:
        distribution: Distribution name as known to the package metadata.

    Returns:
        The installed version string, or ``None`` if the distribution is not installed.
    """
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def check_rlinf_extra() -> bool:
    """Report whether the root ``rlinf`` extra is installed.

    Returns:
        Whether every marker package from the extra is importable.
    """
    missing = [name for name in RLINF_EXTRA_MARKERS if not is_importable(name)]
    if missing:
        print_warning(
            f"Missing packages from the root 'rlinf' extra: {', '.join(missing)}."
            " Run './isaaclab.sh -i contrib[rlinf]' first, then re-run this script."
        )
        return False
    print_info("Root 'rlinf' extra is already installed.")
    return True


def install_pinned_packages(pip_cmd: list[str], force: bool) -> None:
    """Install the packages whose pins must bypass the dependency resolver.

    Args:
        pip_cmd: Base pip command tokens for the active environment.
        force: Whether to reinstall even when the pins are already satisfied.
    """
    if not force and is_importable("rlinf") and installed_version("transformers") == TRANSFORMERS_VERSION:
        print_info(f"rlinf and transformers=={TRANSFORMERS_VERSION} are already installed.")
        return
    run_command(
        pip_cmd
        + [
            "install",
            "--no-deps",
            f"rlinf=={RLINF_VERSION}",
            f"transformers=={TRANSFORMERS_VERSION}",
            TOKENIZERS_SPEC,
        ]
    )


def install_pytorch3d(pip_cmd: list[str]) -> None:
    """Build and install pytorch3d, which GR00T's state/action transforms import.

    Args:
        pip_cmd: Base pip command tokens for the active environment.
    """
    if is_importable("pytorch3d"):
        print_info("pytorch3d is already installed.")
        return
    print_info("Building pytorch3d from source; this takes several minutes...")
    run_command(pip_cmd + ["install", "--no-build-isolation", "--no-deps", PYTORCH3D_SPEC])


def install_gr00t(pip_cmd: list[str], gr00t_dir: Path, commit: str) -> None:
    """Clone Isaac-GR00T at a pinned commit and install it in editable mode.

    Args:
        pip_cmd: Base pip command tokens for the active environment.
        gr00t_dir: Directory the repository is cloned into.
        commit: Commit SHA to check out.
    """
    if not gr00t_dir.exists():
        print_info(f"Cloning Isaac-GR00T into {gr00t_dir}...")
        run_command(["git", "clone", GR00T_REPO_URL, str(gr00t_dir)])

    if is_worktree_dirty(gr00t_dir):
        print_warning(f"{gr00t_dir} has local changes; leaving the checkout untouched.")
    else:
        run_command(["git", "fetch", "origin", commit], cwd=gr00t_dir, check=False)
        run_command(["git", "checkout", "--detach", commit], cwd=gr00t_dir)

    # ``--no-deps`` keeps GR00T's own pins from downgrading the resolved Isaac Sim environment. The
    # dependencies it actually needs come from the root ``rlinf`` extra.
    run_command(pip_cmd + ["install", "--editable", str(gr00t_dir), "--no-deps"])


def is_worktree_dirty(repo_dir: Path) -> bool:
    """Return whether a git worktree has uncommitted changes.

    Args:
        repo_dir: Path to the git repository.

    Returns:
        Whether ``git status --porcelain`` reported any entries.
    """
    result = run_command(["git", "status", "--porcelain"], cwd=repo_dir, check=False, stdout=subprocess.PIPE, text=True)
    return bool(result.stdout and result.stdout.strip())


def install_flash_attn(pip_cmd: list[str], gr00t_dir: Path) -> None:
    """Install flash-attn, falling back to GR00T's PyTorch SDPA patch when the build fails.

    Args:
        pip_cmd: Base pip command tokens for the active environment.
        gr00t_dir: Directory holding the Isaac-GR00T checkout.
    """
    if is_importable("flash_attn"):
        print_info("flash-attn is already installed.")
        return
    result = run_command(
        pip_cmd + ["install", f"flash-attn=={FLASH_ATTN_VERSION}", "--no-build-isolation", "--no-deps"], check=False
    )
    if result.returncode == 0:
        return
    print_warning("flash-attn could not be built; switching GR00T to PyTorch SDPA instead.")
    apply_no_flash_attn_patch(gr00t_dir)


def apply_no_flash_attn_patch(gr00t_dir: Path) -> None:
    """Apply the bundled patch that removes GR00T's flash-attn requirement.

    Args:
        gr00t_dir: Directory holding the Isaac-GR00T checkout.
    """
    already_applied = run_command(
        ["git", "apply", "--reverse", "--check", str(NO_FLASH_ATTN_PATCH)], cwd=gr00t_dir, check=False
    )
    if already_applied.returncode == 0:
        print_info("GR00T is already patched for PyTorch SDPA.")
        return
    run_command(["git", "apply", str(NO_FLASH_ATTN_PATCH)], cwd=gr00t_dir)


def download_checkpoint(checkpoint_dir: Path, repo_id: str) -> None:
    """Download the pretrained VLA checkpoint from the Hugging Face Hub.

    Args:
        checkpoint_dir: Directory the snapshot is written to.
        repo_id: Hugging Face Hub model repository identifier.
    """
    from huggingface_hub import snapshot_download

    print_info(f"Downloading '{repo_id}' into {checkpoint_dir} (several GB; resumes if interrupted)...")
    snapshot_download(repo_id=repo_id, local_dir=str(checkpoint_dir))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line arguments.

    Args:
        argv: Argument list to parse. Defaults to ``sys.argv[1:]`` when ``None``.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--gr00t_dir",
        type=Path,
        default=DEFAULT_GR00T_DIR,
        help="Directory to clone Isaac-GR00T into.",
    )
    parser.add_argument(
        "--gr00t_commit",
        type=str,
        default=GR00T_COMMIT,
        help="Isaac-GR00T commit to check out.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR,
        help="Directory to download the pretrained checkpoint into.",
    )
    parser.add_argument(
        "--skip_checkpoint",
        action="store_true",
        help="Skip the checkpoint download, e.g. when it is already available elsewhere.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reinstall the pinned packages even when they are already satisfied.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the on-demand RLinf demo setup.

    Args:
        argv: Argument list to parse. Defaults to ``sys.argv[1:]`` when ``None``.

    Returns:
        Process exit code.
    """
    args = parse_args(argv)
    if not check_rlinf_extra():
        return 1

    pip_cmd = get_pip_command()
    install_pinned_packages(pip_cmd, args.force)
    install_pytorch3d(pip_cmd)
    install_gr00t(pip_cmd, args.gr00t_dir.resolve(), args.gr00t_commit)
    install_flash_attn(pip_cmd, args.gr00t_dir.resolve())
    if not args.skip_checkpoint:
        download_checkpoint(args.checkpoint_dir.resolve(), CHECKPOINT_REPO_ID)

    print_info("RLinf demo setup complete. Train with:")
    print_info("  ./isaaclab.sh train --rl_library rlinf --config_name isaaclab_ppo_gr00t_assemble_trocar")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
