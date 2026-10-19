# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bind a benchmark's imported Python source to an immutable checkout revision."""

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


def _git_digest(data: bytes, object_format: str) -> str:
    return hashlib.new(object_format, b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def prepare_manifest(root: Path, expected_commit: str | None = None) -> dict:
    """Check tracked Python source with Git and record its immutable blob identities."""
    root = root.resolve()
    commit = _git(root, "rev-parse", "HEAD").decode().strip()
    if expected_commit is not None and commit != expected_commit:
        raise ValueError(f"Checked-out revision {commit} differs from expected commit {expected_commit}.")
    changed = _git(root, "diff", "--name-only", "-z", "--no-renames", commit, "--", "source", "scripts")
    for path in changed.decode().split("\0"):
        if path.endswith((".py", ".pyi")):
            raise ValueError(f"Checked-out Python source differs from commit {commit}: {path}")
    files = {}
    for entry in _git(root, "ls-tree", "-rz", "--full-tree", commit, "--", "source", "scripts").split(b"\0")[:-1]:
        metadata, path_bytes = entry.split(b"\t", 1)
        _, kind, oid = metadata.decode().split()
        path = path_bytes.decode()
        if kind == "blob" and path.endswith((".py", ".pyi")):
            files[path] = oid
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
        "schema_version": 3,
        "commit": commit,
        "git_object_format": _git(root, "rev-parse", "--show-object-format").decode().strip(),
        "files": files,
        "namespaces": {name: sorted(paths) for name, paths in sorted(namespaces.items())},
    }


def _expected_paths(name: str, namespaces: dict) -> list[str]:
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
            # Implicit namespace packages have no __init__.py to verify.
            continue
        path = Path(filename).resolve()
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError:
            relative = None
        if not expected_paths and relative not in manifest["files"]:
            continue
        expected = manifest["files"].get(relative)
        record = {
            "name": name,
            "path": str(path),
            "relative_path": relative,
            "digest": None,
            "status": "verified",
        }
        reasons = []
        if expected_paths and relative not in expected_paths:
            reasons.append("Imported module origin does not match the selected checkout module path.")
        if expected is None:
            reasons.append("Imported Python source is absent from the selected commit manifest.")
        try:
            record["digest"] = _git_digest(path.read_bytes(), manifest["git_object_format"])
        except OSError as exc:
            reasons.append(f"Imported module source could not be read: {exc.strerror}.")
        if expected is not None and record["digest"] != expected:
            reasons.append("Imported Python source bytes differ from the selected commit.")
        if reasons:
            record["status"] = "failed"
            mismatches.extend({"module": name, "path": str(path), "reason": reason} for reason in reasons)
        records.append(record)
    # lazy_export reads these import maps at runtime; they are not Python modules.
    for relative, expected in manifest["files"].items():
        if relative.endswith(".pyi"):
            try:
                if _git_digest((root / relative).read_bytes(), manifest["git_object_format"]) != expected:
                    mismatches.append({"path": relative, "reason": "Import map differs from the selected manifest."})
            except OSError as exc:
                mismatches.append({"path": relative, "reason": f"Import map could not be read: {exc.strerror}."})
    return records, mismatches


def _output_state(output_dir: Path) -> dict[Path, tuple[int, int]]:
    return {
        path: (path.stat().st_mtime_ns, path.stat().st_size) for path in output_dir.rglob("benchmark_runtime_*.json")
    }


def run_benchmark(manifest: dict, root: Path, output_dir: Path, argv: list[str]) -> int:
    """Invoke the normal installed CLI and verify evidence after it finishes."""
    root = root.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    before = _output_state(output_dir)
    old_argv = sys.argv
    old_pycache = sys.pycache_prefix
    exception = None
    exit_code = 0
    with tempfile.TemporaryDirectory(prefix="isaaclab-source-pycache-") as pycache:
        sys.pycache_prefix = pycache
        try:
            sys.argv = ["isaaclab", *argv]
            from isaaclab.cli import cli

            cli()
        except SystemExit as exc:
            exit_code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
            if exit_code:
                exception = {"type": type(exc).__name__, "message": str(exc)}
                if not isinstance(exc.code, int):
                    print(str(exc), file=sys.stderr)
        except Exception as exc:
            traceback.print_exc()
            exit_code = 1
            exception = {"type": type(exc).__name__, "message": str(exc)}
        finally:
            sys.argv = old_argv
            sys.pycache_prefix = old_pycache
        modules, mismatches = _inspect_modules(manifest, root)
        runtime = {"module": RUNTIME_MODULE} if RUNTIME_MODULE in sys.modules else None
        if runtime is None:
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
        verified = exit_code == 0 and not mismatches
        sidecar = {
            "schema_version": 3,
            "status": "verified" if verified else "failed",
            "commit": manifest["commit"],
            "pid": os.getpid(),
            "bytecode_policy": "fresh_process_cache",
            "benchmark_exit_code": exit_code,
            "exception": exception,
            "modules": modules,
            "runtime_entrypoint": runtime,
            "outputs": outputs,
            "mismatches": mismatches,
        }
        (output_dir / "source-revision.json").write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n")
        print(f"Source revision {manifest['commit']}: {sidecar['status']} ({len(modules)} imported modules checked)")
        for mismatch in mismatches:
            context = mismatch.get("module", mismatch.get("path", "runtime output"))
            print(f"Source verification: {context}: {mismatch['reason']}", file=sys.stderr)
    return exit_code or (0 if verified else 1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--checkout-root", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--commit")
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
