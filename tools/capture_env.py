#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Capture an Isaac Lab installation into a portable bundle.

A lockfile records none of what makes one installation differ from another: the GPU and its driver,
the distributions as they exist on disk, the environment variables Isaac Lab reads, and the symlinks
and ``.pth`` files that decide which code is imported. This writes those into a zip, with
``pyproject.toml`` and ``uv.lock`` verbatim, the git state, the ``uv sync`` command whose derived
extras rebuild the environment, and a short ``REPRODUCE.md``. It uses the standard library only and
never imports ``isaaclab``: the installation it describes frequently cannot import anything.
"""

from __future__ import annotations  # noqa: I001  (ruff's target-version predates tomllib in the stdlib)

import argparse
import json
import os
import re
import shlex
import socket
import subprocess
import tomllib
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ISAAC_LAB_REMOTE = "https://github.com/isaac-sim/IsaacLab.git"  # no remote is recorded, so the clone is upstream
# Isaac Sim stamps its provenance into the local version segment: 6.1.0-alpha.59+develop.0.4877ef77.local
_ISAAC_SIM_BUILD = re.compile(r"\+(?P<branch>[^.+]+)\.\d+\.(?P<revision>[0-9a-f]{7,40})")
MACHINE_OWNED_ENV_VARS = frozenset("CONDA_PREFIX ISAACLAB_PATH TMPDIR USER USERNAME VIRTUAL_ENV".split())  # noqa: SIM905
ISAAC_LAB_ENV_VARS = frozenset(
    """
    CARB_APP_PATH CI CI_MARKER CMAKE_POLICY_VERSION_MINIMUM CONDA_PREFIX CUBLAS_WORKSPACE_CONFIG
    CUDA_VISIBLE_DEVICES DEBUG DEBUG_TIMERS DEBUG_TIMER_RESET DEBUG_TIMER_STEP DISPLAY EXP_PATH GITHUB_ACTIONS
    GITLAB_CI HEADLESS ISAACLAB_CHANGELOG_BASE_REF ISAACLAB_CXR_SKIP_AUTOLAUNCH ISAACLAB_DISABLE_LIVE_PLOTS
    ISAACLAB_DUMP_ARTICULATION_PARTITION_IMAGES ISAACLAB_FRANKA_POUR_CUPS_USD_PATH
    ISAACLAB_FRANKA_POUR_ROBOT_USD_PATH ISAACLAB_PATH ISAACLAB_RUN_STANDALONE_SCRIPT_TESTS
    ISAACLAB_STANDALONE_SCREENSHOT_DELAY ISAACLAB_STANDALONE_SCREENSHOT_DIR ISAACLAB_STANDALONE_SCRIPT_RUNTIME_GROUP
    ISAACLAB_STANDALONE_SCRIPT_SCOPE ISAACLAB_STANDALONE_SOAK_TIME ISAACLAB_STANDALONE_STARTUP_TIMEOUT
    ISAACLAB_STANDALONE_VISUALIZER ISAACLAB_TEST_DEVICES ISAACLAB_TEST_QUEUE ISAACLAB_WHEEL ISAACSIM_ASSET_ROOT
    ISAACSIM_CI_SHORT ISAAC_LAB_ENABLE_ISAAC_RTX_PER_ENV_SCENE_PARTITION ISAAC_LAB_OVRTX_USE_OVSTAGE
    ISAAC_LAB_SAVE_STAGES ISAAC_PATH JAX_LOCAL_RANK JAX_RANK LD_LIBRARY_PATH LD_PRELOAD LIVESTREAM LOCAL_RANK
    LOCAL_WORLD_SIZE NEWTON_ASSET_DIR NO_COLOR OMNI_KIT_ACCEPT_EULA OPENBLAS_NUM_THREADS OVRTX_SKIP_USD_CHECK
    PUBLIC_IP PXR_PLUGINPATH_NAME PXR_WORK_THREAD_LIMIT PYTEST_CURRENT_TEST PYTHONHASHSEED PYTHONPATH
    PYTHONUNBUFFERED RANK RAY_AIR_NEW_OUTPUT RLINF_CONFIG_FILE TERM TEST_CUROBO_ONLY TEST_EXCLUDE_PATTERN
    TEST_FILTER_PATTERN TEST_INCLUDE_FILES TEST_K_EXPR TEST_NODE_IDS TEST_NODE_IDS_FILE TEST_NODE_IDS_KEY
    TEST_QUARANTINED_ONLY TEST_RESULT_FILE TEST_SHARD_COUNT TEST_SHARD_INDEX TMPDIR
    TUNE_DISABLE_STRICT_METRIC_CHECKING USER USERNAME USE_RELATIVE_MODE UV_PYTHON VIRTUAL_ENV WANDB_DISABLED
    WARP_CACHE_PATH WAYLAND_DISPLAY WORLD_SIZE XR
    """.split()  # noqa: SIM905
)
"""The exact names Isaac Lab reads or sets: a closed list, matched by exact name and nothing else, so
a variable this project does not read cannot reach a bundle -- neither its value nor its name."""


def _normalize_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).strip().lower()  # PEP 503 normalization


def _run(command: list[str], timeout: int = 60) -> str | None:
    try:
        done = subprocess.run(command, capture_output=True, text=True, errors="replace", timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout + done.stderr or None


def _read_text(path: Path, limit: int | None = None) -> str | None:
    """Return ``path`` as text. ``limit`` truncates; the files copied verbatim pass none, because a
    lockfile cut at a byte offset is no longer a lockfile the reproduction can sync from."""
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if limit is None or len(data) <= limit:
        return data.decode("utf-8", errors="replace")
    return data[:limit].decode("utf-8", errors="replace") + "\n... [truncated]\n"


def _requirement_nodes(requirements: list[dict]) -> set[str]:
    nodes: set[str] = set()
    for requirement in requirements:
        name = _normalize_name(requirement.get("name", ""))
        extras = [*requirement.get("extra", []), *requirement.get("extras", [])]  # both spellings occur
        nodes.update({name, *(f"{name}[{_normalize_name(extra)}]" for extra in extras)})
    return nodes


def parse_lock(lock: str) -> dict:
    """Return ``uv.lock`` as ``root`` plus ``packages``, each with ``versions``, ``requires``, ``extras``."""
    packages: dict[str, dict] = {}
    root: str | None = None
    for entry in tomllib.loads(lock).get("package", []):
        name = _normalize_name(entry.get("name", ""))
        record = packages.setdefault(name, {"versions": set(), "requires": set(), "extras": {}})
        record["versions"] |= {entry["version"]} if entry.get("version") else set()
        source = entry.get("source", {})  # the root project is the one placed at the checkout itself
        root = name if source.get("virtual") == "." or source.get("editable") == "." else root
        record["requires"] |= _requirement_nodes(entry.get("dependencies", []))
        for extra, requirements in entry.get("optional-dependencies", {}).items():
            record["extras"].setdefault(_normalize_name(extra), set()).update(_requirement_nodes(requirements))
    return {"root": root, "packages": packages}


def lock_extras(graph: dict) -> dict[str, set[str]]:
    """Return the root project's extras, mapped to the distributions each one directly requires."""
    root = graph.get("root")
    if not root:
        return {}
    declared, own = graph["packages"][root]["extras"], f"{root}["
    # An extra defined as other extras (`all`) expands into what they require, repeatedly, so that
    # an alias of an alias resolves; a node's own extras drop out, since disk holds only `isaacsim`.
    resolved = {
        name: {node.partition("[")[0] for node in nodes if node != root and not node.startswith(own)}
        for name, nodes in declared.items()
    }
    aliases = {name: [n[len(own) : -1] for n in nodes if n.startswith(own)] for name, nodes in declared.items()}
    for _ in range(len(aliases) + 1):
        for alias, referenced in aliases.items():
            resolved[alias] |= set().union(set(), *(resolved.get(target, set()) for target in referenced))
    return resolved


def select_sync_extras(extras: dict[str, set[str]], installed: set[str]) -> list[str]:
    """Return the fewest extras whose direct requirements are all installed, sorted by name."""
    satisfied = {name: required for name, required in extras.items() if required and required <= installed}
    # Extras with identical requirements are interchangeable; the reverse sort keeps the first name.
    chosen = set({frozenset(required): name for name, required in sorted(satisfied.items(), reverse=True)}.values())
    return sorted(n for n in chosen if not any(satisfied[n] < satisfied[other] for other in chosen if other != n))


def resolve_sync_plan(lock: str | None, distributions: list[dict]) -> dict:
    """Return the ``uv sync`` that rebuilds this environment, its extras derived from the lockfile."""
    if not lock:
        return {"lock_available": False, "extras": [], "command": "uv sync --locked"}
    extras = select_sync_extras(lock_extras(parse_lock(lock)), {dist["key"] for dist in distributions})
    command = "uv sync --locked" + "".join(f" --extra {name}" for name in extras)
    return {"lock_available": True, "extras": extras, "command": command}


def scan_distributions(site_packages: Path) -> list[dict]:
    """Return every distribution in ``site_packages``, read from disk rather than through imports."""
    distributions: list[dict] = []
    for info in sorted(site_packages.iterdir()):
        if info.suffix not in (".dist-info", ".egg-info") or not info.is_dir():
            continue
        metadata = info / ("METADATA" if info.suffix == ".dist-info" else "PKG-INFO")
        headers: dict[str, str] = {}
        for line in (_read_text(metadata, limit=64 << 10) or "").splitlines():
            if not line.strip():
                break
            key, _, value = line.partition(":")
            headers.setdefault(key.strip(), value.strip())
        if not headers.get("Name"):
            continue  # unparsable metadata costs one distribution, not the inventory
        record = {"name": headers["Name"], "key": _normalize_name(headers["Name"]), "location": info.name}
        record["version"] = headers.get("Version") or "unknown"
        record["installer"] = (_read_text(info / "INSTALLER", limit=1024) or "").strip() or None
        distributions.append(record)
    return sorted(distributions, key=lambda dist: dist["key"])


def collect_environment() -> tuple[dict, dict[str, str]]:
    """Return the allowlisted variables, plus a count of the ones present but never named."""
    variables = {name: value for name, value in os.environ.items() if name in ISAAC_LAB_ENV_VARS}
    omitted = len(os.environ) - len(variables)
    rendered = (
        "# Only the exact variables Isaac Lab reads or sets are captured.\n"
        f"# {omitted} other variable(s) were present and deliberately not collected.\n"
        + "".join(f"{name}={variables[name]}\n" for name in sorted(variables))
    )
    return {"variables": variables, "omitted_count": omitted}, {"env/environment.txt": rendered}


def collect_repo(repo_root: Path) -> tuple[dict, dict[str, str]]:
    """Return the checked-out revision, with ``pyproject.toml`` and ``uv.lock`` copied verbatim.

    Read out of ``.git`` rather than through the git binary, which an installation broken enough to
    need capturing may not have, and which a read-only mount will not run.
    """
    artifacts = {f"files/{name}": _read_text(repo_root / name) or "" for name in ("pyproject.toml", "uv.lock")}
    dot_git, git_dir = repo_root / ".git", None
    if dot_git.is_dir():
        git_dir = dot_git
    elif (redirect := (_read_text(dot_git, limit=4096) or "").strip()).startswith("gitdir:"):
        git_dir = Path(redirect.removeprefix("gitdir:").strip())  # a worktree points at its real state
    head = ((_read_text(git_dir / "HEAD", limit=4096) if git_dir else None) or "").strip()
    branch = head.partition("ref: refs/heads/")[2] or None
    commit = None if branch else head or None
    if branch and git_dir:
        # A worktree keeps its own HEAD but shares refs with the checkout it was made from.
        common = (_read_text(git_dir / "commondir", limit=4096) or ".").strip()
        roots = [git_dir, (git_dir / common).resolve()]
        loose = (_read_text(d / "refs" / "heads" / branch, limit=4096) for d in roots)
        packed = "".join(_read_text(d / "packed-refs", limit=1 << 20) or "" for d in roots)
        commit = next((ref.strip() for ref in loose if ref), None) or next(
            (line.split()[0] for line in packed.splitlines() if line.endswith(f" refs/heads/{branch}")), None
        )
    return {"root": str(repo_root), "git": {"commit": commit, "branch": branch}}, {
        name: content for name, content in artifacts.items() if content
    }


def collect_isaac_sim(repo_root: Path, distributions: list[dict]) -> tuple[dict, dict[str, str]]:
    """Return which Isaac Sim this installation reaches and how it was obtained.

    A wheel is already described by the lockfile; a downloaded package and a local build are not,
    and only the ``VERSION`` string says which one to fetch or rebuild.
    """
    wheel = next((dist["version"] for dist in distributions if dist["key"] == "isaacsim"), None)
    env_path = os.environ.get("ISAAC_PATH")
    candidates = [repo_root / "_isaac_sim", *([Path(env_path)] if env_path else [])]
    kit = next((path for path in candidates if path.is_dir()), None)
    version = ((_read_text(kit / "VERSION", limit=4096) or "").strip() if kit else "") or None
    build = _ISAAC_SIM_BUILD.search(version or "")
    # A path through a build tree, or a version stamped `.local`, is a Kit compiled on that machine.
    local = bool(kit) and ("_build" in kit.resolve().parts or (version or "").endswith(".local"))
    section = {
        "install_method": ("source_build" if local else "binary") if kit else ("wheel" if wheel else "none"),
        "path": str(kit.resolve()) if kit else None,
        "version": version,
        "branch": build.group("branch") if build else None,
        "revision": build.group("revision") if build else None,
        "wheel_version": wheel,
    }
    return section, {"isaacsim/VERSION": version + "\n"} if version else {}


def build_manifest(
    repo_root: Path,
    venv: Path | None,
    command: str | None = None,
) -> tuple[dict, dict[str, str]]:
    """Capture an environment into a manifest and the files stored alongside it."""
    capture = {"hostname": socket.gethostname(), "command_under_test": command}
    manifest: dict = {"captured_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "capture": capture}
    manifest["repo"], artifacts = collect_repo(repo_root)
    manifest["environment"], files = collect_environment()
    artifacts.update(files)
    query = "index,name,uuid,driver_version,vbios_version,memory.total"
    fields = query.split(",")
    try:
        rows = (_run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"]) or "").splitlines()
        # A driver that cannot be reached makes nvidia-smi print its failure on the same channel as
        # the CSV, so a row is a device only when it carries every field that was asked for.
        cells = ([value.strip() for value in row.split(",")] for row in rows)
        devices = [dict(zip(fields, values)) for values in cells if len(values) == len(fields)]
        driver_version = devices[0]["driver_version"] if devices else None
    except (KeyError, IndexError, ValueError):  # a bundle without the GPU beats no bundle at all
        devices, driver_version = [], None
    manifest["gpu"] = {"devices": devices, "driver_version": driver_version}
    site_packages = next((p for p in (venv or Path()).glob("[Ll]ib*/**/site-packages") if p.is_dir()), None)
    manifest["python"] = {"venv": str(venv) if venv else None, "distributions": []}
    if site_packages is not None:
        manifest["python"]["distributions"] = scan_distributions(site_packages)
        artifacts["files/pyvenv.cfg"] = _read_text(venv / "pyvenv.cfg", limit=64 << 10) or ""
        # A .pth file runs at interpreter start and can put anything on sys.path, so it is copied.
        for pth in sorted(site_packages.glob("*.pth")):
            artifacts[f"files/pth/{pth.name}"] = _read_text(pth, limit=256 << 10) or ""
    manifest["sync"] = resolve_sync_plan(artifacts.get("files/uv.lock"), manifest["python"]["distributions"])
    manifest["isaac_sim"], files = collect_isaac_sim(repo_root, manifest["python"]["distributions"])
    artifacts.update(files)
    # Only the top level: a link that redirects imports sits at the root (`_isaac_sim`) or in
    # site-packages, and everything deeper in this tree is tracked and arrives with the clone.
    symlinks = [
        {"path": str(entry), "target": os.readlink(entry), "exists": entry.exists(), "in_virtualenv": in_venv}
        for directory, in_venv in ((repo_root, False), *([(site_packages, True)] if site_packages else ()))
        for entry in sorted(directory.iterdir())
        if entry.is_symlink()
    ]
    manifest["links"] = {"symlinks": symlinks}
    return manifest, {name: content for name, content in artifacts.items() if content}


def render_document(manifest: dict, artifacts: dict[str, str]) -> str:
    """Render the reproduction document, written into the bundle and beside it."""
    repo, sync, gpu = manifest.get("repo", {}), manifest.get("sync", {}), manifest.get("gpu", {})
    git, environment = repo.get("git", {}), manifest.get("environment", {})
    variables, omitted = environment.get("variables", {}), environment.get("omitted_count", 0)
    root = str(repo.get("root", ""))
    command = manifest.get("capture", {}).get("command_under_test")
    hand_made = [
        link
        for link in manifest.get("links", {}).get("symlinks", [])
        if link["path"].startswith(root) and not link.get("in_virtualenv")
    ]
    exportable = {name: value for name, value in variables.items() if name not in MACHINE_OWNED_ENV_VARS}
    owned = ", ".join(f"`{name}`" for name in sorted(set(variables) & MACHINE_OWNED_ENV_VARS))
    steps = [
        "unzip <this-bundle>.zip -d bundle",
        f"git clone {ISAAC_LAB_REMOTE} IsaacLab-repro && cd IsaacLab-repro",
        *([f"git checkout {git['commit']}"] if git.get("commit") else []),
        "cp ../bundle/files/pyproject.toml ../bundle/files/uv.lock .",
        sync.get("command", "uv sync --locked"),
        *(
            f"ln -s {link['target']} {link['path'][len(root) :].lstrip('/')}"
            + ("" if link["exists"] else "   # BROKEN on the captured machine")
            for link in hand_made[:20]
        ),
        *(f"export {name}={shlex.quote(exportable[name])}" for name in sorted(exportable)),
    ]
    notes = []
    if sync.get("extras"):
        notes.append("The extras are derived from what is installed; `uv sync` deletes whatever they leave out.")
    if git.get("commit"):
        notes.append("The clone URL is Isaac Lab itself; a commit missing from it came from a fork.")
    if owned:
        notes.append(f"Recorded but never exported, because they describe the machine, not the run: {owned}.")
    summary = (
        f"Captured {manifest.get('captured_at', '?')} on {manifest.get('capture', {}).get('hostname', '?')}:"
        f" Isaac Lab {(git.get('commit') or 'unknown')[:12]} on {git.get('branch') or '?'}"
        f", driver {gpu.get('driver_version') or 'none'} on"
        f" {len(gpu.get('devices', []))} GPU(s),"
        f" {len(manifest.get('python', {}).get('distributions', []))} packages installed."
        + (f" The command under test was `{command}`." if command else "")
    )
    return f"""# Isaac Lab environment capture

{summary}

## Rebuilding it, beside the unpacked bundle (symlink targets are the captured machine's paths)

```bash
{chr(10).join(steps)}
```

{chr(10).join(notes)}

## What this bundle cannot reproduce

The GPU and its driver, and anything outside the repository and the environment: system packages and
whatever `PYTHONPATH` and `LD_LIBRARY_PATH` reach are recorded by path only. The process environment
is captured by allowlist -- {len(variables)} variable(s) from the closed list of names Isaac Lab reads
or sets, while {omitted} variable(s) were present but not collected, and are named nowhere here.
Paths, hostnames, and usernames are captured as-is, so review `env/environment.txt` before sending
this anywhere.
"""


def write_bundle(destination: Path, manifest: dict, artifacts: dict[str, str], document: str) -> None:
    """Write the manifest, the document, and every artifact into a zip at ``destination``."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archive.writestr("manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        archive.writestr("REPRODUCE.md", document)
        for name, content in sorted(artifacts.items()):
            archive.writestr(name, content)


def main(argv: list[str] | None = None) -> int:
    """Parse the command line, capture this machine, and write the bundle and its document."""
    parser = argparse.ArgumentParser(prog="capture_env.py", description=__doc__.splitlines()[0])
    capture = parser.add_subparsers(dest="command_name", required=True).add_parser(
        "capture", help="Capture this machine's environment into a bundle."
    )
    capture.add_argument("--repo_root", default=None, help="Repository to describe. Default: auto-detected.")
    capture.add_argument("--venv", default=None, help="Environment to describe. Default: $VIRTUAL_ENV, else .venv.")
    capture.add_argument("--output_dir", default=".", help="Where to write the bundle. Default: the current directory.")
    capture.add_argument("--command", default=None, help="The command that prompted this capture, recorded verbatim.")
    args = parser.parse_args(argv)
    marked = (path for path in [Path.cwd(), *Path.cwd().parents] if (path / "pyproject.toml").is_file())
    repo_root = Path(args.repo_root).resolve() if args.repo_root else next(marked, Path.cwd())
    venv = Path(args.venv or os.environ.get("VIRTUAL_ENV") or repo_root / ".venv")
    print(f"capturing {repo_root} with {venv} ...")
    manifest, artifacts = build_manifest(repo_root, venv if venv.is_dir() else None, args.command)
    document = render_document(manifest, artifacts)
    stem = f"isaaclab-env-{socket.gethostname()}-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
    bundle = Path(args.output_dir).resolve() / f"{stem}.zip"
    write_bundle(bundle, manifest, artifacts, document)
    bundle.with_suffix(".md").write_text(document)
    print(document.split("## Rebuilding")[0].strip(), "\n\nbundle:", bundle)
    print("Review it before sending: paths, hostnames, and usernames are captured as-is.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
