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

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path

MAX_COPIED_FILE_BYTES = 16 << 20  # sized for `uv.lock`, the largest file copied verbatim
SYMLINK_SCAN_DEPTH = 3  # reaches source/<package>/<module> without walking into an asset tree
PRUNED_DIRECTORIES = frozenset(  # skipped by the symlink scan for size, not for lack of interest
    ".git .venv __pycache__ .pytest_cache .ruff_cache node_modules logs outputs _build".split()  # noqa: SIM905
)
# An HTTPS remote authenticates with what its URL carries; an SSH remote's userinfo is a login name.
CREDENTIAL_BEARING_URL_SCHEMES = frozenset({"ftp", "ftps", "http", "https"})
# Collected, but never exported by the document: these describe the machine, not the run.
MACHINE_OWNED_ENV_VARS = frozenset("CONDA_PREFIX ISAACLAB_PATH TMPDIR USER USERNAME VIRTUAL_ENV".split())  # noqa: SIM905
ISAAC_LAB_ENV_VARS = frozenset(
    """
    CARB_APP_PATH CI CI_MARKER CMAKE_POLICY_VERSION_MINIMUM CONDA_PREFIX CUBLAS_WORKSPACE_CONFIG CUDA_VISIBLE_DEVICES
    DEBUG DEBUG_TIMERS DEBUG_TIMER_RESET DEBUG_TIMER_STEP DISPLAY EXP_PATH GITHUB_ACTIONS GITLAB_CI HEADLESS
    ISAACLAB_CHANGELOG_BASE_REF ISAACLAB_CXR_SKIP_AUTOLAUNCH ISAACLAB_DISABLE_LIVE_PLOTS
    ISAACLAB_DUMP_ARTICULATION_PARTITION_IMAGES ISAACLAB_FRANKA_POUR_CUPS_USD_PATH
    ISAACLAB_FRANKA_POUR_ROBOT_USD_PATH ISAACLAB_PATH ISAACLAB_PPISP_PERF ISAACLAB_PPISP_PERF_DEVICE
    ISAACLAB_PPISP_PERF_MEASURE_ITERS ISAACLAB_PPISP_PERF_MEMORY_FRACTION ISAACLAB_PPISP_PERF_NUM_ENVS
    ISAACLAB_PPISP_PERF_OUTPUT ISAACLAB_PPISP_PERF_RESOLUTIONS ISAACLAB_PPISP_PERF_VARIANTS
    ISAACLAB_PPISP_PERF_WARMUP_ITERS ISAACLAB_RUN_STANDALONE_SCRIPT_TESTS ISAACLAB_STANDALONE_SCREENSHOT_DELAY
    ISAACLAB_STANDALONE_SCREENSHOT_DIR ISAACLAB_STANDALONE_SCRIPT_RUNTIME_GROUP ISAACLAB_STANDALONE_SCRIPT_SCOPE
    ISAACLAB_STANDALONE_SOAK_TIME ISAACLAB_STANDALONE_STARTUP_TIMEOUT ISAACLAB_STANDALONE_VISUALIZER
    ISAACLAB_TEST_DEVICES ISAACLAB_TEST_QUEUE ISAACLAB_WHEEL ISAACSIM_ASSET_ROOT ISAACSIM_CI_SHORT ISAAC_PATH
    ISAAC_LAB_ENABLE_ISAAC_RTX_PER_ENV_SCENE_PARTITION ISAAC_LAB_OVRTX_USE_OVSTAGE ISAAC_LAB_SAVE_STAGES
    JAX_LOCAL_RANK JAX_RANK LD_LIBRARY_PATH LD_PRELOAD LIVESTREAM LOCAL_RANK LOCAL_WORLD_SIZE NEWTON_ASSET_DIR
    NO_COLOR OMNI_KIT_ACCEPT_EULA OPENBLAS_NUM_THREADS OVRTX_SKIP_USD_CHECK PUBLIC_IP PXR_PLUGINPATH_NAME
    PXR_WORK_THREAD_LIMIT PYTEST_CURRENT_TEST PYTHONHASHSEED PYTHONPATH PYTHONUNBUFFERED RANK RAY_AIR_NEW_OUTPUT
    RLINF_CONFIG_FILE TERM TEST_CUROBO_ONLY TEST_EXCLUDE_PATTERN TEST_FILTER_PATTERN TEST_INCLUDE_FILES TEST_K_EXPR
    TEST_NODE_IDS TEST_NODE_IDS_FILE TEST_NODE_IDS_KEY TEST_QUARANTINED_ONLY TEST_RESULT_FILE TEST_SHARD_COUNT
    TEST_SHARD_INDEX TMPDIR TUNE_DISABLE_STRICT_METRIC_CHECKING USER USERNAME USE_RELATIVE_MODE UV_PYTHON
    VIRTUAL_ENV WANDB_DISABLED WARP_CACHE_PATH WAYLAND_DISPLAY WORLD_SIZE XR
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


def _read_text(path: Path, limit: int = MAX_COPIED_FILE_BYTES) -> str | None:
    try:
        data = path.read_bytes()
    except OSError:
        return None
    text, marker = data[:limit].decode("utf-8", errors="replace"), "\n... [truncated]\n"
    return text + marker if len(data) > limit else text


def _git(repo_root: Path, args: list[str], timeout: int = 15) -> str | None:
    options = {"capture_output": True, "text": True, "errors": "replace", "timeout": timeout, "check": False}
    try:
        done = subprocess.run(["git", *args], cwd=str(repo_root), **options)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout if done.returncode == 0 else None


def sanitize_remote_url(url: str) -> str:
    """Return ``url`` without the userinfo that can hold a credential; bundles go onto public issues."""
    scheme, separator, rest = url.partition("://")
    if not separator:
        # SCP-style or a local path. Only the `user:password@host` form holds a secret.
        userinfo, at_sign, tail = url.rpartition("@")
        return tail if at_sign and ":" in userinfo else url
    authority, slash, path = rest.partition("/")
    userinfo, at_sign, host = authority.rpartition("@")
    if not at_sign:
        return url
    if scheme.lower() in CREDENTIAL_BEARING_URL_SCHEMES or ":" in userinfo:
        return f"{scheme}://{host}{slash}{path}"
    return url


def _requirement_nodes(requirements: list[dict]) -> set[str]:
    nodes: set[str] = set()
    for requirement in requirements:
        name = _normalize_name(requirement.get("name", ""))
        extras = [*requirement.get("extra", []), *requirement.get("extras", [])]  # both spellings occur
        nodes.update({name, *(f"{name}[{_normalize_name(extra)}]" for extra in extras)})
    return nodes


def parse_lock(lock: str) -> dict:
    """Return ``uv.lock`` as ``root`` plus ``packages``, each with ``versions``, ``requires``, ``extras``."""
    import tomllib  # imported here so an interpreter without it still produces the rest of a capture

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
    try:
        graph = parse_lock(lock) if lock else None
    except (ImportError, ValueError):
        graph = None
    if graph is None:
        return {"lock_available": False, "extras": [], "command": "uv sync --locked"}
    extras = select_sync_extras(lock_extras(graph), {dist["key"] for dist in distributions})
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


def is_collected_env_var(name: str) -> bool:
    """Return whether ``name`` is in the allowlist. Exact membership only: a rule clever enough to
    admit an unforeseen name is also clever enough to admit a secret."""
    return name in ISAAC_LAB_ENV_VARS


def collect_environment() -> tuple[dict, dict[str, str]]:
    """Return the allowlisted variables, plus a count of the ones present but never named."""
    variables = {name: value for name, value in os.environ.items() if is_collected_env_var(name)}
    omitted = len(os.environ) - len(variables)
    rendered = (
        "# Only the exact variables Isaac Lab reads or sets are captured.\n"
        f"# {omitted} other variable(s) were present and deliberately not collected.\n"
        + "".join(f"{name}={variables[name]}\n" for name in sorted(variables))
    )
    return {"variables": variables, "omitted_count": omitted}, {"env/environment.txt": rendered}


def collect_repo(repo_root: Path, include_diff: bool, include_remotes: bool = False) -> tuple[dict, dict[str, str]]:
    """Return the repository's git state, with ``pyproject.toml`` and ``uv.lock`` copied verbatim.

    Both flags are off by default: a dirty tree can hold code the sender may not share, and a fork's
    URL names an organisation and a host. The diffstat is stored either way, so the gap stays visible.
    """
    artifacts = {f"files/{name}": _read_text(repo_root / name) or "" for name in ("pyproject.toml", "uv.lock")}
    artifacts["repo/git-status.txt"] = status = _git(repo_root, ["status", "--porcelain"]) or ""
    artifacts["repo/git-diffstat.txt"] = _git(repo_root, ["diff", "--stat", "HEAD"]) or ""
    git_info: dict = {"dirty": bool(status.strip()), "diff_included": False, "remotes_included": include_remotes}
    git_info["commit"] = (_git(repo_root, ["rev-parse", "HEAD"]) or "").strip() or None
    git_info["branch"] = (_git(repo_root, ["rev-parse", "--abbrev-ref", "HEAD"]) or "").strip() or None
    if include_diff and git_info["dirty"]:
        artifacts["repo/git-diff.patch"] = _git(repo_root, ["diff", "HEAD"], timeout=60) or ""
        git_info["diff_included"] = True
    # Redacted on the way in, so the manifest, the listing, and the clone command all read the same.
    rows = [(line, line.split()) for line in (_git(repo_root, ["remote", "-v"]) or "").splitlines() if include_remotes]
    if rows:
        urls = {fields[1]: sanitize_remote_url(fields[1]) for _, fields in rows if len(fields) > 1}
        artifacts["repo/git-remote.txt"] = "\n".join(
            "\t".join([f[0], " ".join([urls[f[1]], *f[2:]])]) if len(f) > 1 else line for line, f in rows
        )
        git_info["remotes"] = sorted(set(urls.values()))
        git_info["remotes_redacted"] = any(raw != clean for raw, clean in urls.items())
    return {"root": str(repo_root), "git": git_info}, {k: v for k, v in artifacts.items() if v}


def build_manifest(
    repo_root: Path,
    venv: Path | None,
    command: str | None = None,
    include_diff: bool = False,
    include_remotes: bool = False,
) -> tuple[dict, dict[str, str]]:
    """Capture an environment into a manifest and the files stored alongside it."""
    capture = {"hostname": socket.gethostname(), "command_under_test": command}
    manifest: dict = {"captured_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "capture": capture}
    manifest["repo"], artifacts = collect_repo(repo_root, include_diff, include_remotes)
    manifest["environment"], files = collect_environment()
    artifacts.update(files)
    query = "index,name,uuid,driver_version,vbios_version,memory.total"
    rows = (_run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"]) or "").splitlines()
    devices = [dict(zip(query.split(","), [value.strip() for value in row.split(",")])) for row in rows]
    manifest["gpu"] = {"devices": devices, "driver_version": devices[0]["driver_version"] if devices else None}
    site_packages = next((p for p in (venv or Path()).glob("[Ll]ib*/**/site-packages") if p.is_dir()), None)
    manifest["python"] = {"venv": str(venv) if venv else None, "distributions": []}
    if site_packages is not None:
        manifest["python"]["distributions"] = scan_distributions(site_packages)
        artifacts["files/pyvenv.cfg"] = _read_text(venv / "pyvenv.cfg", limit=64 << 10) or ""
        # A .pth file runs at interpreter start and can put anything on sys.path, so it is copied.
        for pth in sorted(site_packages.glob("*.pth")):
            artifacts[f"files/pth/{pth.name}"] = _read_text(pth, limit=256 << 10) or ""
    manifest["sync"] = resolve_sync_plan(artifacts.get("files/uv.lock"), manifest["python"]["distributions"])
    tracked = {
        str(repo_root / line.split("\t", 1)[1])
        for line in (_git(repo_root, ["ls-files", "-s"]) or "").splitlines()
        if line.startswith("120000 ") and "\t" in line
    }
    symlinks: list[dict] = []
    frontier = [(repo_root, SYMLINK_SCAN_DEPTH), *([(site_packages, 0)] if site_packages else [])]
    while frontier:
        directory, depth = frontier.pop()
        for entry in sorted(directory.iterdir()) if directory.is_dir() else []:
            if entry.is_symlink():
                link = {"path": str(entry), "target": os.readlink(entry), "exists": entry.exists()}
                link["tracked"] = link["path"] in tracked
                link["in_virtualenv"] = any((parent / "pyvenv.cfg").is_file() for parent in entry.parents)
                symlinks.append(link)
            elif depth > 0 and entry.is_dir() and entry.name not in PRUNED_DIRECTORIES:
                frontier.append((entry, depth - 1))
    manifest["links"] = {"symlinks": symlinks}
    return manifest, {name: content for name, content in artifacts.items() if content}


def render_document(manifest: dict, artifacts: dict[str, str]) -> str:
    """Render the reproduction document, written into the bundle and beside it."""
    repo, sync, gpu = manifest.get("repo", {}), manifest.get("sync", {}), manifest.get("gpu", {})
    git, environment = repo.get("git", {}), manifest.get("environment", {})
    variables, omitted = environment.get("variables", {}), environment.get("omitted_count", 0)
    root = str(repo.get("root", ""))
    remote = (git.get("remotes") or ["https://github.com/isaac-sim/IsaacLab.git"])[0]
    command = manifest.get("capture", {}).get("command_under_test")
    hand_made = [
        link
        for link in manifest.get("links", {}).get("symlinks", [])
        if link["path"].startswith(root) and not link.get("tracked") and not link.get("in_virtualenv")
    ]
    exportable = {name: value for name, value in variables.items() if name not in MACHINE_OWNED_ENV_VARS}
    owned = ", ".join(f"`{name}`" for name in sorted(set(variables) & MACHINE_OWNED_ENV_VARS))
    steps = [
        "unzip <this-bundle>.zip -d bundle",
        f"git clone {remote} IsaacLab-repro && cd IsaacLab-repro",
        *([f"git checkout {git['commit']}"] if git.get("commit") else []),
        *(["git apply ../bundle/repo/git-diff.patch"] if git.get("diff_included") else []),
        "cp ../bundle/files/pyproject.toml ../bundle/files/uv.lock .",
        sync.get("command", "uv sync --locked"),
        *(
            f"ln -s {link['target']} {link['path'][len(root) :].lstrip('/')}"
            + ("" if link["exists"] else "   # BROKEN on the captured machine")
            for link in hand_made[:20]
        ),
        *(f"export {name}={exportable[name]!r}" for name in sorted(exportable)),
    ]
    notes = []
    if sync.get("extras"):
        notes.append("The extras are derived from what is installed; `uv sync` deletes whatever they leave out.")
    if not git.get("remotes_included"):
        notes.append("Remotes were not recorded: the clone URL is Isaac Lab; a commit missing from it is a fork's.")
    if owned:
        notes.append(f"Recorded but never exported, because they describe the machine, not the run: {owned}.")
    summary = (
        f"Captured {manifest.get('captured_at', '?')} on {manifest.get('capture', {}).get('hostname', '?')}:"
        f" Isaac Lab {(git.get('commit') or 'unknown')[:12]} on {git.get('branch') or '?'}"
        f"{' (dirty)' if git.get('dirty') else ''}, driver {gpu.get('driver_version') or 'none'} on"
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
Paths, hostnames, and usernames are captured as-is, so review `env/environment.txt` and, when it is
present, `repo/git-diff.patch` before sending this anywhere.
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
    capture.add_argument("--include_diff", action="store_true", help="Attach the uncommitted working-tree patch.")
    capture.add_argument("--include_remotes", action="store_true", help="Attach the configured git remote URLs.")
    args = parser.parse_args(argv)
    marked = (path for path in [Path.cwd(), *Path.cwd().parents] if (path / "pyproject.toml").is_file())
    repo_root = Path(args.repo_root).resolve() if args.repo_root else next(marked, Path.cwd())
    venv = Path(args.venv or os.environ.get("VIRTUAL_ENV") or repo_root / ".venv")
    print(f"capturing {repo_root} with {venv} ...")
    manifest, artifacts = build_manifest(
        repo_root, venv if venv.is_dir() else None, args.command, args.include_diff, args.include_remotes
    )
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
