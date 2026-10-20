# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Check the selected Git checkout and the benchmark's imported source paths."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path, PurePosixPath

RUNTIME_MODULE = "isaaclab.benchmark.entrypoints.runtime"


def _git(root: Path, *args: str) -> bytes:
    result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, check=False)
    if result.returncode:
        raise ValueError(f"Git source inspection failed: {result.stderr.decode(errors='replace').strip()}")
    return result.stdout


def check_checkout(root: Path, expected_commit: str | None = None) -> str:
    """Check the selected revision and tracked Python source on the host."""
    commit = _git(root, "rev-parse", "HEAD").decode().strip()
    if expected_commit is not None and commit != expected_commit:
        raise ValueError(f"Checked-out revision {commit} differs from expected commit {expected_commit}.")
    changed = _git(
        root,
        "-c",
        "diff.autoRefreshIndex=true",
        "-c",
        "core.filemode=false",
        "diff",
        "--name-only",
        "-z",
        "--no-renames",
        commit,
        "--",
        "source",
        "scripts",
    )
    for path in changed.decode().split("\0"):
        if path.endswith((".py", ".pyi")):
            details = _git(root, "diff", "--numstat", "--summary", "--no-renames", commit, "--", path).decode().strip()
            raise ValueError(f"Checked-out Python source differs from commit {commit}: {path}\n{details}")
    return commit


def prepare_manifest(root: Path, expected_commit: str | None = None) -> dict:
    """Record the selected commit and repository-relative Python paths.
    `files` lists paths. `namespaces` maps installed package names to their source directories.
    """
    commit = check_checkout(root, expected_commit)
    files = sorted(
        path
        for path in _git(root, "ls-tree", "-rz", "--name-only", commit, "--", "source", "scripts").decode().split("\0")
        if path.endswith((".py", ".pyi"))
    )
    if not files:
        raise ValueError("The selected commit contains no Python source under source/ or scripts/.")
    namespaces = {}
    for path in files:
        parts = PurePosixPath(path).parts
        # Test directories may contain __init__.py without being installed packages.
        if len(parts) >= 4 and parts[0] == "source" and parts[2] == parts[1].replace("-", "_"):
            prefix = "/".join(parts[:3])
            namespaces.setdefault(parts[2], set()).add(prefix)
    return {
        "schema_version": 4,
        "commit": commit,
        "files": files,
        "namespaces": {name: sorted(paths) for name, paths in sorted(namespaces.items())},
    }


def _expected_paths(name: str, namespaces: dict[str, list[str]]) -> list[str]:
    namespace, _, suffix = name.partition(".")
    paths = []
    for prefix in namespaces.get(namespace, []):
        stem = prefix + ("/" + suffix.replace(".", "/") if suffix else "")
        paths.extend((stem + ".py", stem + "/__init__.py"))
    return paths


def _inspect_modules(manifest: dict, root: Path) -> tuple[list[dict], list[dict]]:
    records, mismatches = [], []
    for name, module in sorted(list(sys.modules.items())):
        filename = getattr(module, "__file__", None)
        expected_paths = _expected_paths(name, manifest["namespaces"])
        if not filename:
            # Modules without a source file have no path to verify.
            continue
        path = Path(filename).resolve()
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError:
            relative = None
        if not expected_paths and relative not in manifest["files"]:
            continue
        record = {
            "name": name,
            "path": str(path),
            "relative_path": relative,
        }
        reasons = []
        if expected_paths and relative not in expected_paths:
            reasons.append("Imported module origin does not match the selected checkout module path.")
        if relative not in manifest["files"]:
            reasons.append("Imported Python source is absent from the selected commit manifest.")
        if reasons:
            mismatches.extend({"module": name, "path": str(path), "reason": reason} for reason in reasons)
        records.append(record)
    return records, mismatches


def _output_state(output_dir: Path) -> dict[Path, tuple[int, int]]:
    return {
        path: (path.stat().st_mtime_ns, path.stat().st_size) for path in output_dir.rglob("benchmark_runtime_*.json")
    }


def run_benchmark(manifest: dict, root: Path, output_dir: Path, argv: list[str]) -> int:
    """Verify source evidence after the CLI completes successfully."""
    root = root.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    before = _output_state(output_dir)
    with tempfile.TemporaryDirectory(prefix="isaaclab-source-pycache-") as pycache:
        sys.pycache_prefix = pycache
        try:
            sys.argv = ["isaaclab", *argv]
            from isaaclab.cli import cli

            cli()
        except SystemExit as exc:
            if exc.code not in (0, None):
                raise
        modules, mismatches = _inspect_modules(manifest, root)
        if RUNTIME_MODULE not in sys.modules:
            mismatches.append(
                {"module": RUNTIME_MODULE, "reason": "Runtime entrypoint was not imported in this process."}
            )
        outputs = []
        for path, state in sorted(_output_state(output_dir).items()):
            if before.get(path) == state:
                continue
            data = path.read_bytes()
            outputs.append(
                {
                    "path": path.relative_to(output_dir).as_posix(),
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "bytes": len(data),
                }
            )
        if not outputs:
            mismatches.append({"reason": "The benchmark invocation produced no runtime JSON output."})
        verified = not mismatches
        sidecar = {
            "schema_version": 4,
            "status": "verified" if verified else "failed",
            "commit": manifest["commit"],
            "pid": os.getpid(),
            "bytecode_policy": "fresh_process_cache",
            "modules": modules,
            "outputs": outputs,
            "mismatches": mismatches,
        }
        (output_dir / "source-revision.json").write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n")
        print(f"Source revision {manifest['commit']}: {sidecar['status']} ({len(modules)} imported modules checked)")
        for mismatch in mismatches:
            context = mismatch.get("module", mismatch.get("path", "runtime output"))
            print(f"Source verification: {context}: {mismatch['reason']}", file=sys.stderr)
    return 0 if verified else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--checkout-root", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--commit")
    check = commands.add_parser("check")
    check.add_argument("--manifest", type=Path, required=True)
    check.add_argument("--checkout-root", type=Path, required=True)
    check.add_argument("--output-dir", type=Path, required=True)
    run = commands.add_parser("run")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--checkout-root", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("benchmark_argv", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            manifest = prepare_manifest(args.checkout_root, args.commit)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
            print(f"Prepared source revision {manifest['commit']} ({len(manifest['files'])} Python files)")
            return 0
        if args.command == "check":
            try:
                manifest = json.loads(args.manifest.read_text())
                check_checkout(args.checkout_root, manifest["commit"])
            except (OSError, ValueError, KeyError, TypeError) as exc:
                proof_path = args.output_dir / "source-revision.json"
                if proof_path.is_file():
                    proof = json.loads(proof_path.read_text())
                    proof["status"] = "failed"
                    proof["mismatches"].append({"reason": f"Host Git checkout check failed: {exc}"})
                    proof_path.write_text(json.dumps(proof, indent=2, sort_keys=True) + "\n")
                raise
            return 0
        benchmark_argv = args.benchmark_argv
        if benchmark_argv[:1] == ["--"]:
            benchmark_argv = benchmark_argv[1:]
        return run_benchmark(json.loads(args.manifest.read_text()), args.checkout_root, args.output_dir, benchmark_argv)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        traceback.print_exc()
        print(f"Source revision verification failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
