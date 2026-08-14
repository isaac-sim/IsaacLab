# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Misc commands"""

import json
import re
import shutil
import zipfile
from pathlib import Path

import tomllib

from ..utils import (
    ISAACLAB_ROOT,
    extract_isaacsim_exe,
    is_windows,
    print_error,
    print_info,
    print_warning,
    run_command,
    run_python_command,
)


def command_run_isaacsim(sim_args: list[str]) -> None:
    """Run Isaac Sim (-s).

    Args:
        sim_args: Additional arguments passed to the Isaac Sim executable.
    """

    isaacsim_exe = extract_isaacsim_exe()
    print_info(f"Running Isaac Sim from: {isaacsim_exe}")

    isaacsim_exe.append("--ext-folder")
    isaacsim_exe.append(str(ISAACLAB_ROOT / "source"))
    isaacsim_exe.extend(sim_args)

    run_command(isaacsim_exe, check=False)


def command_new(new_args: list[str]) -> None:
    """Create a new external project or internal task from template (-n).

    Args:
        new_args: Arguments forwarded to the template generator CLI.
    """

    print_info("Installing template dependencies...")
    reqs = ISAACLAB_ROOT / "tools" / "template" / "requirements.txt"
    run_python_command("pip", ["install", "-q", "-r", str(reqs)], is_module=True)

    print_info("Running template generator...")
    cli_script = ISAACLAB_ROOT / "tools" / "template" / "cli.py"
    run_python_command(cli_script, new_args)


def command_test(test_args: list[str]) -> None:
    """Run pytest for Isaac Lab tests (-t).

    Args:
        test_args: Additional pytest arguments.
    """
    run_python_command("-m", ["pytest", str(ISAACLAB_ROOT / "tools")] + test_args)


def command_vscode_settings() -> None:
    """Update the vscode settings from template and Isaac Sim settings"""

    print_info("Setting up vscode settings...")

    # Path to setup_vscode.py.
    setup_vscode_script = ISAACLAB_ROOT / ".vscode" / "tools" / "setup_vscode.py"

    # Check if the file exists before attempting to run it.
    if setup_vscode_script.exists():
        run_python_command(setup_vscode_script, [])
        print_info("VS Code settings generated successfully.")
    else:
        print_warning("Unable to find the script 'setup_vscode.py'. Aborting vscode settings setup.")


def command_build_docs() -> None:
    """Build the documentation."""
    print_info("Building documentation...")
    docs_dir = ISAACLAB_ROOT / "docs"

    uv_exe = shutil.which("uv")
    if uv_exe is None:
        print_error("uv could not be found. Please install uv and try again.")
        print_error("https://docs.astral.sh/uv/getting-started/installation/")
        raise SystemExit(1)

    out_dir = docs_dir / "_build" / "current"
    cmd = [
        uv_exe,
        "run",
        "--isolated",
        "--extra",
        "test",
        "--",
        "python",
        "-m",
        "sphinx",
        "-W",
        "--keep-going",
        "-j",
        "auto",
        "-b",
        "html",
        "-d",
        "_build/doctrees",
        ".",
        str(out_dir),
    ]
    run_command(cmd, cwd=docs_dir)

    index_path = out_dir / "index.html"
    print_info(f"Documentation built at {index_path}")
    if not is_windows():
        print_info(f"Open with: xdg-open {index_path}")


def command_build_isaacsim(source_path: str) -> None:
    """Build Isaac Sim from source and make it usable through ``uv`` (--isaacsim_source).

    Runs an incremental Isaac Sim build, packages the resulting Python wheels, links them into the
    Isaac Lab repository as ``_isaac_sim_wheels``, points uv at that directory through
    ``find-links``, pins the ``isaacsim-local`` extra to the version that was built, and re-resolves
    Isaac Sim from those wheels. Afterwards, run Isaac Lab against the build with
    ``uv run --extra isaacsim-local``.

    The pin is required: source builds produce pre-release local versions that sort below the
    published release, so an unpinned extra resolves back to the registry wheels instead. It
    edits ``pyproject.toml``, which (like ``uv.lock``) must not be committed.

    Args:
        source_path: Path to an Isaac Sim source checkout.
    """
    isaacsim_root = Path(source_path).expanduser().resolve()
    build_script = isaacsim_root / ("build.bat" if is_windows() else "build.sh")
    repo_script = isaacsim_root / ("repo.bat" if is_windows() else "repo.sh")

    if not build_script.is_file():
        print_error(f"'{isaacsim_root}' is not an Isaac Sim source checkout ({build_script.name} not found).")
        print_info("Clone it first with: git clone https://github.com/isaac-sim/IsaacSim.git")
        raise SystemExit(1)

    build_dir = isaacsim_root / "_build"
    print_info("Incrementally building Isaac Sim from source. This may take a while...")
    run_command([str(build_script)], cwd=isaacsim_root)

    print_info("Packaging the Isaac Sim build as Python wheels...")
    for repo_args in (["python_package", "--create"], ["comment_archive_deps"], ["python_package", "--wheel"]):
        run_command([str(repo_script)] + repo_args, cwd=isaacsim_root)

    wheel_dir = build_dir / "packages" / "dist"
    wheels = sorted(wheel_dir.glob("*.whl")) if wheel_dir.is_dir() else []
    if not wheels:
        print_error(f"No wheels were produced in {wheel_dir}.")
        raise SystemExit(1)
    print_info(f"Built {len(wheels)} wheel(s) in {wheel_dir}.")

    local_version = _extract_local_isaacsim_version(wheel_dir)
    _check_kernel_abi(wheel_dir, build_script)

    # A stable, git-ignored path inside the repository keeps the find-links entry short and
    # valid across shells, the same way ``_isaac_sim`` does for the Kit build.
    link_path = ISAACLAB_ROOT / "_isaac_sim_wheels"
    if link_path.is_symlink() or link_path.exists():
        if link_path.is_symlink():
            link_path.unlink()
        else:
            print_error(f"{link_path} exists and is not a symbolic link. Remove it and re-run.")
            raise SystemExit(1)
    try:
        link_path.symlink_to(wheel_dir, target_is_directory=True)
        # Store the absolute symlink path.  Unlike a bare ``_isaac_sim_wheels`` path, it remains
        # valid when the command is run from another directory.
        find_links = str(link_path)
        print_info(f"Linked {link_path} -> {wheel_dir}")
    except OSError as error:
        # Windows requires elevation (or developer mode) to create symbolic links.
        find_links = str(wheel_dir)
        print_warning(f"Could not create {link_path} ({error}). Using the absolute wheel path instead.")

    # Point uv at the wheels from the project configuration so the setting survives shell restarts
    # and applies to every later ``uv run``/``uv sync``.
    _set_uv_find_links(find_links)

    # Source builds carry a pre-release local version (``6.0.1rc7+develop.<hash>.local``) that
    # sorts *below* the published release, so an unpinned ``isaacsim-local`` extra always resolves
    # back to the registry. uv only honors a version this specific per conflicting-extra fork when
    # it is written into the requirement itself, so pin the extra to the version just built.
    _pin_isaacsim_local_extra(local_version)
    _add_isaacsim_local_conflicts()

    # Re-resolve Isaac Sim so the lock file picks the local wheels over the published release.
    if shutil.which("uv") is not None:
        print_info("Re-resolving Isaac Sim from the local wheels...")
        run_command(["uv", "lock", "--upgrade-package", "isaacsim"], cwd=ISAACLAB_ROOT)
    else:
        print_warning("uv was not found on PATH. Run 'uv lock --upgrade-package isaacsim' once uv is available.")

    print_info("Isaac Sim is ready. Run Isaac Lab against it with:")
    print_info(
        "  uv run --extra isaacsim-local isaaclab train --rl_library rsl_rl --task Isaac-Cartpole-Direct"
        " physics=isaacsim_physx"
    )
    print_warning(
        f"pyproject.toml now points uv at '{find_links}' and pins the 'isaacsim-local' extra to your build"
        f" ({local_version}), and uv.lock resolves it from there. Both are local-only; revert with"
        " 'git checkout pyproject.toml uv.lock' before committing."
    )


def _extract_local_isaacsim_version(wheel_dir: Path) -> str:
    """Read the version of the locally built ``isaacsim`` wheel.

    Args:
        wheel_dir: Directory holding the wheels produced by the Isaac Sim source build.

    Returns:
        The version of the top-level ``isaacsim`` wheel, e.g. ``6.0.1rc7+develop.0.98701505.local``.
    """
    # Wheel filenames are ``<name>-<version>-<python tag>-...``; the sibling extension wheels
    # normalize to ``isaacsim_<component>``, so only the top-level distribution matches here.
    candidates = sorted(wheel_dir.glob("isaacsim-*.whl"))
    if not candidates:
        print_error(f"No top-level 'isaacsim' wheel found in {wheel_dir}.")
        raise SystemExit(1)
    return candidates[-1].name.split("-")[1]


def _check_kernel_abi(wheel_dir: Path, build_script: Path) -> None:
    """Verify the packaged Kit kernel matches the Python ABI its wheel is tagged for.

    ``repo.sh python_package`` tags the wheels from ``repo.toml`` rather than from the binaries it
    packages, so a build tree left over from an older Kit (a different Python) is repackaged under
    the wrong tag. ``uv`` then installs it happily and Isaac Sim only fails much later with
    ``No module named 'carb._carb'``.

    Args:
        wheel_dir: Directory holding the wheels produced by the Isaac Sim source build.
        build_script: Path to the Isaac Sim ``build.sh``/``build.bat``, quoted in the error message.
    """
    kernels = sorted(wheel_dir.glob("isaacsim_kernel-*.whl"))
    if not kernels:
        # Older Isaac Sim layouts may not split the kernel out; nothing to check against.
        return
    kernel = kernels[-1]
    # ``<name>-<version>-<python tag>-<abi tag>-<platform tag>.whl``
    wheel_tag = kernel.stem.split("-")[2]

    with zipfile.ZipFile(kernel) as archive:
        # ``carb._carb`` is the module Kit bootstraps first, and it ships as a single ABI build.
        found = [re.search(r"/carb/_carb\.(cp\d+)", name.replace("cpython-", "cp")) for name in archive.namelist()]
    built_tags = {match.group(1) for match in found if match}
    if not built_tags or wheel_tag in built_tags:
        return

    print_error(
        f"{kernel.name} is tagged '{wheel_tag}' but its carb kernel was built for"
        f" {'/'.join(sorted(built_tags))}. The build tree is stale: it predates the Python version"
        " the checkout now targets."
    )
    print_info(f"Remove '{wheel_dir.parents[1]}' and rebuild with: {build_script}")
    raise SystemExit(1)


def _set_uv_find_links(find_links: str) -> None:
    """Point uv at a wheel directory by writing ``find-links`` into ``[tool.uv]`` in ``pyproject.toml``.

    The entry is part of the project configuration, so it survives shell restarts and applies to
    every later ``uv`` invocation in the checkout. Re-running the build rewrites the existing entry.

    Args:
        find_links: Directory holding the locally built Isaac Sim wheels, either the repository-relative
            ``_isaac_sim_wheels`` link or an absolute path when that link could not be created.
    """
    pyproject = ISAACLAB_ROOT / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")

    header = re.search(r"^\[tool\.uv\]$", text, flags=re.MULTILINE)
    if header is None:
        print_error(f"Could not find the '[tool.uv]' table in {pyproject}.")
        raise SystemExit(1)

    config = tomllib.loads(text)
    existing_links = config.get("tool", {}).get("uv", {}).get("find-links", [])
    if not isinstance(existing_links, list) or not all(isinstance(link, str) for link in existing_links):
        print_error(f"The 'find-links' entry in {pyproject} must be an array of strings.")
        raise SystemExit(1)

    # Confine the edit to the ``[tool.uv]`` table so a ``find-links`` entry in another table is
    # neither read nor rewritten. Preserve user-provided entries: this command only adds its own
    # local wheel directory. Windows paths are normalized to forward slashes, which uv accepts.
    start = header.end()
    following = re.search(r"^\[", text[start:], flags=re.MULTILINE)
    end = start + (following.start() if following is not None else len(text) - start)
    section = text[start:end]
    find_links = Path(find_links).as_posix()
    links = [link for link in existing_links if link != find_links] + [find_links]
    entry = f"find-links = [{', '.join(json.dumps(link) for link in links)}]"

    # ``find-links`` may be formatted over several lines. Re-rendering just this array retains all
    # entries while avoiding duplicate TOML keys on repeated source-build invocations.
    find_links_match = re.search(r"^find-links\s*=\s*\[(?:[^\]]|\n)*?\]", section, flags=re.MULTILINE)
    if find_links_match is None:
        comment = "# Locally built Isaac Sim wheels ('isaaclab --isaacsim_source'). Local-only, do not commit."
        section = f"\n{comment}\n{entry}{section}"
    else:
        section = section[: find_links_match.start()] + entry + section[find_links_match.end() :]

    updated = text[:start] + section + text[end:]
    if updated != text:
        pyproject.write_text(updated, encoding="utf-8")
    print_info(f"Pointed uv at the local wheels via 'find-links' in {pyproject.name} ({find_links}).")


def _pin_isaacsim_local_extra(version: str) -> None:
    """Pin the ``isaacsim-local`` extra in ``pyproject.toml`` to a locally built Isaac Sim.

    Args:
        version: Version of the locally built ``isaacsim`` wheel.
    """
    pyproject = ISAACLAB_ROOT / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    pinned = f'isaacsim-local = ["isaacsim[all,extscache]=={version}"]'
    updated, count = re.subn(r"^isaacsim-local = \[.*\]$", pinned, text, count=1, flags=re.MULTILINE)
    if count == 0:
        print_error(f"Could not find the 'isaacsim-local' extra in {pyproject}.")
        raise SystemExit(1)
    if updated != text:
        pyproject.write_text(updated, encoding="utf-8")
    print_info(f"Pinned the 'isaacsim-local' extra to isaacsim=={version}.")


def _add_isaacsim_local_conflicts() -> None:
    """Add local-only conflicts for extras that pin the published Isaac Sim wheel.

    A source build has a pre-release local version, whereas these extras require the published
    release exactly. The normal project deliberately has no such conflicts; they are needed only
    after :func:`_pin_isaacsim_local_extra` changes the local checkout.
    """
    pyproject = ISAACLAB_ROOT / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    config = tomllib.loads(text)
    existing_conflicts = config.get("tool", {}).get("uv", {}).get("conflicts", [])
    if not isinstance(existing_conflicts, list):
        print_error(f"The 'conflicts' entry in {pyproject} must be an array.")
        raise SystemExit(1)

    conflicts: list[list[dict[str, str]]] = []
    for conflict in existing_conflicts:
        if not (
            isinstance(conflict, list)
            and all(isinstance(term, dict) and isinstance(term.get("extra"), str) for term in conflict)
        ):
            print_error(f"The 'conflicts' entry in {pyproject} must contain extra selectors.")
            raise SystemExit(1)
        conflicts.append([{"extra": term["extra"]} for term in conflict])

    existing_pairs = {frozenset(term["extra"] for term in conflict) for conflict in conflicts}
    for extra in ("isaacsim", "teleop", "all"):
        pair = frozenset(("isaacsim-local", extra))
        if pair not in existing_pairs:
            conflicts.append([{"extra": "isaacsim-local"}, {"extra": extra}])

    conflict_lines = []
    for conflict in conflicts:
        terms = ", ".join(f"{{ extra = {json.dumps(term['extra'])} }}" for term in conflict)
        conflict_lines.append(f"    [{terms}],")
    entry = "conflicts = [\n" + "\n".join(conflict_lines) + "\n]"
    header = re.search(r"^\[tool\.uv\]$", text, flags=re.MULTILINE)
    if header is None:
        print_error(f"Could not find the '[tool.uv]' table in {pyproject}.")
        raise SystemExit(1)
    start = header.end()
    following = re.search(r"^\[", text[start:], flags=re.MULTILINE)
    end = start + (following.start() if following is not None else len(text) - start)
    section = text[start:end]
    conflict_match = re.search(r"^conflicts\s*=\s*\[\n.*?^\]$", section, flags=re.MULTILINE | re.DOTALL)
    if conflict_match is None:
        section = (
            f"\n# Local source builds cannot co-resolve with extras that pin published Isaac Sim.\n{entry}{section}"
        )
    else:
        section = section[: conflict_match.start()] + entry + section[conflict_match.end() :]
    pyproject.write_text(text[:start] + section + text[end:], encoding="utf-8")
    print_info("Added local-only uv conflicts for the pinned 'isaacsim-local' extra.")


def command_run_docker(args: list[str]) -> None:
    """Run the docker container helper script (docker/container.py).

    Args:
        args: Arguments forwarded to ``docker/container.py``.
    """
    script_path = ISAACLAB_ROOT / "docker" / "container.py"
    print_info(f"Running docker utility script from: {script_path}")
    run_python_command(script_path, args)
