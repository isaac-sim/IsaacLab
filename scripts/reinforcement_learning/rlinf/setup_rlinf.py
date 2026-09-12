# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""On-demand installer for the RLinf VLA post-training demo.

The demo needs three things that ``pyproject.toml`` cannot express: package pins that must bypass the
dependency resolver, a source checkout of Isaac-GR00T, and a pretrained checkpoint. Baking them into the
Isaac Lab container image would cost every user the download, so they are fetched here instead, only when
the demo is actually run.

Install the root ``rlinf`` extra first, then run this script for the GR00T generation you need::

    ./isaaclab.sh -i contrib[rlinf]
    ./isaaclab.sh -p scripts/reinforcement_learning/rlinf/setup_rlinf.py --gr00t n15   # assemble_trocar
    ./isaaclab.sh -p scripts/reinforcement_learning/rlinf/setup_rlinf.py --gr00t n17   # pick_and_place_apple, pack_agx

Both GR00T generations install as the ``gr00t`` package with incompatible APIs, so one Python
environment holds one generation at a time; re-running with the other ``--gr00t`` value swaps it.

Every step is idempotent and skips work that is already done, so the script can be re-run to repair a
partial install. Afterwards, train and evaluate with::

    uv run --no-sync isaaclab train --rl_library rlinf --config_name isaaclab_ppo_gr00t_assemble_trocar
    uv run --no-sync isaaclab play --rl_library rlinf --config_name isaaclab_ppo_gr00t_assemble_trocar --video
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import subprocess
from dataclasses import dataclass
from pathlib import Path

from isaaclab.cli.utils import (
    ISAACLAB_ROOT,
    extract_python_exe,
    get_pip_command,
    print_info,
    print_warning,
    run_command,
)

# RLinf and GR00T are installed with ``--no-deps`` because their own pins clash with the Isaac Sim ones
# already resolved in the environment. One RLinf serves both GR00T generations: this commit dispatches
# on ``actor.model.model_type`` (``gr00t``/``gr00t_n1d5`` vs ``gr00t_n1d7``).
RLINF_SPEC = "git+https://github.com/RLinf/RLinf.git@0f9ea98c7a6d9e3ade24e8f4846c64d3b135dbcc"
RLINF_VERSION = "0.3.0"
FLASH_ATTN_VERSION = "2.8.3"
COSMOS_BACKBONE_REPO_ID = "nvidia/Cosmos-Reason2-2B"


@dataclass(frozen=True)
class Gr00tProfile:
    """Everything that differs between the supported GR00T generations."""

    commit: str
    """Isaac-GR00T commit to check out."""
    transformers: str
    """``transformers`` version GR00T's checkpoint code was written against."""
    tokenizers: str
    """``tokenizers`` requirement matching :attr:`transformers`."""
    needs_pytorch3d: bool
    """Whether ``gr00t.data.transform`` imports ``pytorch3d`` (N1.5 only)."""
    gr00t_dirname: str
    """Checkout directory name, next to the Isaac Lab repository."""
    checkpoints: tuple[tuple[str, str | None], ...]
    """``(repo_id, subfolder)`` pairs downloaded under ``--checkpoint_root/<repo tail>``."""


PROFILES: dict[str, Gr00tProfile] = {
    # Newer transformers drop ``VideoInput`` from ``transformers.image_utils``, which the N1.5
    # checkpoint's remote processing code imports. Later GR00T commits restructure
    # ``gr00t.experiment`` and drop the ``data_config`` module the N1.5 task configs import.
    "n15": Gr00tProfile(
        commit="4af2b622892f7dcb5aae5a3fb70bcb02dc217b96",
        transformers="4.51.3",
        tokenizers="tokenizers>=0.21,<0.22",
        needs_pytorch3d=True,
        gr00t_dirname="Isaac-GR00T",
        checkpoints=(("nvidia/Assemble_Trocar", None),),
    ),
    # GR00T N1.7 General Release: the first N1.7 commit whose ``requires-python`` admits the
    # Python 3.12 interpreter Isaac Lab ships (earlier N1.7 commits pin ``==3.10.*`` and refuse
    # to install). Its checkpoints reference the gated ``nvidia/Cosmos-Reason2-2B`` backbone.
    "n17": Gr00tProfile(
        commit="1a1837f20538b7d7e21f977a11a5aee14f99803c",
        transformers="4.57.3",
        tokenizers="tokenizers>=0.22,<0.23",
        needs_pytorch3d=False,
        gr00t_dirname="Isaac-GR00T-N1.7",
        checkpoints=(
            ("LiFanxing/pnp_apple", "gen_data_SFT_l40s-2/checkpoint-30000"),
            ("LiFanxing/pack_agx", "pack_agx_mimicgen_sft_l40s-1/checkpoint-40000"),
            (COSMOS_BACKBONE_REPO_ID, None),
        ),
    ),
}
# N1.5's gr00t.data.transform.state_action imports pytorch3d.transforms. RLinf pins the
# ``pipablepytorch3d`` wheel, but that distribution caps out at Python 3.11, so on the Python 3.12
# interpreter Isaac Lab ships this is built from source instead, as the GR00T docs already do.
PYTORCH3D_SPEC = "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.9"

GR00T_REPO_URL = "https://github.com/NVIDIA/Isaac-GR00T.git"

DEFAULT_CHECKPOINT_ROOT = ISAACLAB_ROOT / ".pretrained_checkpoints" / "rlinf"
NO_FLASH_ATTN_PATCH = ISAACLAB_ROOT / "scripts/imitation_learning/locomanipulation_sdg/gr00t/no_flash_attn.patch"

# Marker packages from the root ``rlinf`` extra, used to detect that ``-i contrib[rlinf]`` has not run.
RLINF_EXTRA_MARKERS = ("ray", "decord", "peft", "diffusers", "timm")


def is_importable(module: str) -> bool:
    """Return whether a module can be resolved without importing it."""
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def installed_version(distribution: str) -> str | None:
    """Return the installed version of a distribution, or ``None`` when absent."""
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def installed_gr00t_generation() -> str | None:
    """Return ``"n15"``, ``"n17"``, or ``None`` for the ``gr00t`` package currently importable.

    The two releases share the package name, so the API surface identifies the generation.
    """
    if is_importable("gr00t.data.types"):
        return "n17"
    if is_importable("gr00t.experiment.data_config"):
        return "n15"
    return None


def check_rlinf_extra() -> bool:
    """Report whether the root ``rlinf`` extra is installed."""
    missing = [name for name in RLINF_EXTRA_MARKERS if not is_importable(name)]
    if missing:
        print_warning(
            f"Missing packages from the root 'rlinf' extra: {', '.join(missing)}."
            " Run './isaaclab.sh -i contrib[rlinf]' first, then re-run this script."
        )
        return False
    print_info("Root 'rlinf' extra is already installed.")
    return True


def install_pinned_packages(pip_cmd: list[str], profile: Gr00tProfile, force: bool) -> None:
    """Install the packages whose pins must bypass the dependency resolver."""
    satisfied = (
        installed_version("rlinf") == RLINF_VERSION and installed_version("transformers") == profile.transformers
    )
    if not force and satisfied:
        print_info(f"rlinf=={RLINF_VERSION} and transformers=={profile.transformers} are already installed.")
        return
    run_command(
        pip_cmd + ["install", "--no-deps", RLINF_SPEC, f"transformers=={profile.transformers}", profile.tokenizers]
    )


def install_pytorch3d(pip_cmd: list[str]) -> None:
    """Build and install pytorch3d, which GR00T's state/action transforms import."""
    if is_importable("pytorch3d"):
        print_info("pytorch3d is already installed.")
        return
    print_info("Building pytorch3d from source; this takes several minutes...")
    run_command(pip_cmd + ["install", "--no-build-isolation", "--no-deps", PYTORCH3D_SPEC])


def install_gr00t(pip_cmd: list[str], gr00t_dir: Path, commit: str, generation: str) -> None:
    """Clone Isaac-GR00T at a pinned commit and install it in editable mode."""
    installed = installed_gr00t_generation()
    if installed is not None and installed != generation:
        print_info(f"Replacing the installed GR00T {installed} package with {generation}...")
        run_command(pip_cmd + ["uninstall", "gr00t"] + ([] if pip_cmd[0] == "uv" else ["-y"]), check=False)

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
    """Return whether a git worktree has uncommitted changes."""
    result = run_command(["git", "status", "--porcelain"], cwd=repo_dir, check=False, stdout=subprocess.PIPE, text=True)
    return bool(result.stdout and result.stdout.strip())


def check_gr00t_resolves_to(gr00t_dir: Path) -> None:
    """Warn when ``import gr00t`` resolves somewhere other than the checkout just installed.

    A stale ``PYTHONPATH`` entry pointing at another Isaac-GR00T clone takes precedence over the
    editable install and silently swaps the GR00T generation the tasks run against.
    """
    result = run_command(
        [extract_python_exe(), "-c", "import gr00t, os; print(os.path.dirname(os.path.dirname(gr00t.__file__)))"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    resolved = (result.stdout or "").strip()
    if result.returncode != 0 or not resolved:
        print_warning("Could not import gr00t after installation; check the output above.")
        return
    if Path(resolved).resolve() != gr00t_dir.resolve():
        print_warning(
            f"'import gr00t' resolves to {resolved}, not the checkout just installed ({gr00t_dir}). "
            "Remove that path from PYTHONPATH, otherwise the tasks run against the wrong GR00T generation."
        )


def install_flash_attn(pip_cmd: list[str], gr00t_dir: Path) -> None:
    """Install flash-attn, falling back to GR00T's PyTorch SDPA patch when the build fails."""
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
    """Apply the bundled patch that removes GR00T's flash-attn requirement."""
    already_applied = run_command(
        ["git", "apply", "--reverse", "--check", str(NO_FLASH_ATTN_PATCH)], cwd=gr00t_dir, check=False
    )
    if already_applied.returncode == 0:
        print_info("GR00T is already patched for PyTorch SDPA.")
        return
    run_command(["git", "apply", str(NO_FLASH_ATTN_PATCH)], cwd=gr00t_dir)


def download_checkpoints(checkpoint_root: Path, profile: Gr00tProfile) -> None:
    """Download the profile's pretrained checkpoints (and backbone) from the Hugging Face Hub.

    Each repository lands under ``checkpoint_root/<repo tail>`` with its Hub layout intact, so a
    task config points at ``<repo tail>/<subfolder>``. Gated repositories such as the Cosmos
    backbone need a logged-in account with access.
    """
    from huggingface_hub import snapshot_download

    for repo_id, subfolder in profile.checkpoints:
        local_dir = checkpoint_root / repo_id.split("/")[-1]
        patterns = [f"{subfolder}/*"] if subfolder else None
        print_info(f"Downloading '{repo_id}' into {local_dir} (several GB; resumes if interrupted)...")
        snapshot_download(repo_id=repo_id, local_dir=str(local_dir), allow_patterns=patterns)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--gr00t",
        choices=sorted(PROFILES),
        default="n15",
        help="GR00T generation to install: n15 for assemble_trocar, n17 for the H2 + Sharpa tasks.",
    )
    parser.add_argument(
        "--gr00t_dir",
        type=Path,
        default=None,
        help="Directory to clone Isaac-GR00T into. Defaults to ../<profile dirname> next to the repository.",
    )
    parser.add_argument(
        "--gr00t_commit",
        type=str,
        default=None,
        help="Isaac-GR00T commit to check out. Defaults to the selected profile's pin.",
    )
    parser.add_argument(
        "--checkpoint_root",
        type=Path,
        default=DEFAULT_CHECKPOINT_ROOT,
        help="Directory the pretrained checkpoints are downloaded under.",
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
    """Run the on-demand RLinf demo setup."""
    args = parse_args(argv)
    if not check_rlinf_extra():
        return 1

    profile = PROFILES[args.gr00t]
    gr00t_dir = (args.gr00t_dir or ISAACLAB_ROOT.parent / profile.gr00t_dirname).resolve()
    commit = args.gr00t_commit or profile.commit

    pip_cmd = get_pip_command()
    install_pinned_packages(pip_cmd, profile, args.force)
    if profile.needs_pytorch3d:
        install_pytorch3d(pip_cmd)
    install_gr00t(pip_cmd, gr00t_dir, commit, args.gr00t)
    check_gr00t_resolves_to(gr00t_dir)
    install_flash_attn(pip_cmd, gr00t_dir)
    if not args.skip_checkpoint:
        download_checkpoints(args.checkpoint_root.resolve(), profile)

    config = (
        "isaaclab_ppo_gr00t_assemble_trocar" if args.gr00t == "n15" else "isaaclab_ppo_gr00t_pick_and_place_apple_n17"
    )
    print_info(f"RLinf demo setup complete for GR00T {args.gr00t}. Train with:")
    print_info(f"  uv run --no-sync isaaclab train --rl_library rlinf --config_name {config}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
