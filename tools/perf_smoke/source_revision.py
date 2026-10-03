# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Check the selected Git checkout and the benchmark's imported source paths."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path, PurePosixPath

RUNTIME_MODULE = "isaaclab.benchmark.entrypoints.runtime"
MEASUREMENT_PATHS = ("source/isaaclab/isaaclab/benchmark", "source/isaaclab/isaaclab/utils/timer.py")


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git(root: Path, *args: str, data: bytes | None = None) -> bytes:
    result = subprocess.run(["git", "-C", str(root), *args], input=data, capture_output=True, check=False)
    if result.returncode:
        raise ValueError(f"Git source inspection failed: {result.stderr.decode(errors='replace').strip()}")
    return result.stdout


def _tracked_files(root: Path, commit: str, paths: tuple[str, ...]) -> dict[str, str]:
    files = {}
    for entry in _git(root, "ls-tree", "-rz", "--full-tree", commit, "--", *paths).split(b"\0")[:-1]:
        metadata, path_bytes = entry.split(b"\t", 1)
        _, kind, oid = metadata.decode().split()
        if kind == "blob":
            files[path_bytes.decode()] = oid
    return files


def measurement_digest(files: dict[str, str]) -> str:
    """Identify measurement contents independently of the commit containing them."""
    return _digest(json.dumps(files, sort_keys=True, separators=(",", ":")).encode())


def snapshot_measurement(root: Path, output_dir: Path) -> dict:
    """Save the tested revision's measurement package for both benchmark legs."""
    commit = _git(root, "rev-parse", "HEAD").decode().strip()
    entries = _tracked_files(root, commit, MEASUREMENT_PATHS)
    batch = io.BytesIO(_git(root, "cat-file", "--batch", data="".join(f"{oid}\n" for oid in entries.values()).encode()))
    files = {}
    for path in entries:
        _, _, size = batch.readline().split()
        blob = batch.read(int(size))
        batch.read(1)
        destination = output_dir / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(blob)
        files[path] = _digest(blob)
    manifest = {"schema_version": 1, "commit": commit, "files": files, "digest": measurement_digest(files)}
    (output_dir / "measurement-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


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


def prepare_manifest(
    root: Path, measurement_manifest: dict | None = None, *, expected_commit: str | None = None
) -> dict:
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
    manifest = {
        "schema_version": 4,
        "commit": commit,
        "files": files,
        "namespaces": {name: sorted(paths) for name, paths in sorted(namespaces.items())},
    }
    if measurement_manifest is not None:
        try:
            if not isinstance(measurement_manifest["files"], dict):
                raise TypeError("Measurement files must map paths to hashes.")
            if measurement_digest(measurement_manifest["files"]) != measurement_manifest["digest"]:
                raise ValueError("Measurement manifest digest does not match its files.")
        except (KeyError, TypeError) as exc:
            raise ValueError(f"Invalid measurement manifest: {exc}") from exc
        manifest["measurement"] = measurement_manifest
    return manifest


def _expected_paths(name: str, namespaces: dict[str, list[str]]) -> list[str]:
    namespace, _, suffix = name.partition(".")
    paths = []
    for prefix in namespaces.get(namespace, []):
        stem = prefix + ("/" + suffix.replace(".", "/") if suffix else "")
        paths.extend((stem + ".py", stem + "/__init__.py"))
    return paths


def _inspect_files(files: dict, root: Path, description: str) -> list[dict]:
    mismatches = []
    for relative, expected in files.items():
        try:
            if _digest((root / relative).read_bytes()) != expected:
                mismatches.append({"path": relative, "reason": f"{description} differs from the selected manifest."})
        except OSError as exc:
            mismatches.append({"path": relative, "reason": f"{description} could not be read: {exc.strerror}."})
    return mismatches


def _inspect_modules(manifest: dict, root: Path) -> tuple[list[dict], list[dict]]:
    records, mismatches = [], []
    measurement_files = manifest.get("measurement", {}).get("files", {})
    files = set(manifest["files"]) | measurement_files.keys()
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
        if not expected_paths and relative not in files:
            continue
        record = {
            "name": name,
            "path": str(path),
            "relative_path": relative,
        }
        reasons = []
        if expected_paths and relative not in expected_paths:
            reasons.append("Imported module origin does not match the selected checkout module path.")
        if relative not in files:
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
        measurement = manifest.get("measurement")
        if measurement is not None:
            if measurement_digest(measurement["files"]) != measurement["digest"]:
                mismatches.append({"reason": "Measurement manifest digest does not match its files."})
            mismatches.extend(_inspect_files(measurement["files"], root, "Measurement source"))
        outputs = []
        for path, state in sorted(_output_state(output_dir).items()):
            if before.get(path) == state:
                continue
            data = path.read_bytes()
            outputs.append(
                {"path": path.relative_to(output_dir).as_posix(), "sha256": _digest(data), "bytes": len(data)}
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
        if measurement is not None:
            sidecar["measurement_digest"] = measurement["digest"]
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
    prepare.add_argument("--measurement-manifest", type=Path)
    check = commands.add_parser("check")
    check.add_argument("--manifest", type=Path, required=True)
    check.add_argument("--checkout-root", type=Path, required=True)
    check.add_argument("--output-dir", type=Path, required=True)
    snapshot = commands.add_parser("snapshot")
    snapshot.add_argument("--checkout-root", type=Path, required=True)
    snapshot.add_argument("--output-dir", type=Path, required=True)
    run = commands.add_parser("run")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--checkout-root", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("benchmark_argv", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    try:
        if args.command == "snapshot":
            manifest = snapshot_measurement(args.checkout_root, args.output_dir)
            print(f"Prepared measurement source {manifest['commit']} ({manifest['digest']})")
            return 0
        if args.command == "prepare":
            measurement = json.loads(args.measurement_manifest.read_text()) if args.measurement_manifest else None
            manifest = prepare_manifest(args.checkout_root, measurement, expected_commit=args.commit)
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
