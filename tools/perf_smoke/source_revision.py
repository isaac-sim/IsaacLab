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
import types
from pathlib import Path, PurePosixPath

RUNTIME_MODULE = "isaaclab.benchmark.entrypoints.runtime"
MEASUREMENT_PATHS = ("source/isaaclab/isaaclab/benchmark", "source/isaaclab/isaaclab/utils/timer.py")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git(root: Path, *args: str, data: bytes | None = None) -> bytes:
    result = subprocess.run(["git", "-C", str(root), *args], input=data, capture_output=True, check=False)
    if result.returncode:
        raise ValueError(f"Git source inspection failed: {result.stderr.decode(errors='replace').strip()}")
    return result.stdout


def _tracked_blobs(root: Path, commit: str, paths: tuple[str, ...], *, python_only: bool = False) -> dict[str, bytes]:
    entries = []
    for entry in _git(root, "ls-tree", "-rz", "--full-tree", commit, "--", *paths).split(b"\0"):
        if not entry:
            continue
        metadata, path_bytes = entry.split(b"\t", 1)
        _, kind, oid = metadata.decode().split()
        path = path_bytes.decode()
        if kind == "blob" and (not python_only or path.endswith(".py")):
            entries.append((oid, path))
    batch = _git(root, "cat-file", "--batch", data="".join(f"{oid}\n" for oid, _ in entries).encode())
    files = {}
    cursor = 0
    for oid, path in entries:
        header_end = batch.index(b"\n", cursor)
        actual_oid, kind, size = batch[cursor:header_end].decode().split()
        if actual_oid != oid or kind != "blob":
            raise ValueError(f"Unexpected Git object for {path}.")
        cursor = header_end + 1
        blob = batch[cursor : cursor + int(size)]
        cursor += int(size) + 1
        files[path] = blob
    return files


def measurement_digest(files: dict[str, str]) -> str:
    """Identify measurement contents independently of the commit containing them."""
    return _sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode())


def snapshot_measurement(root: Path, output_dir: Path) -> dict:
    """Save the tested revision's measurement package for both benchmark legs."""
    commit = _git(root, "rev-parse", "HEAD").decode().strip()
    files = {}
    for path, blob in _tracked_blobs(root, commit, MEASUREMENT_PATHS).items():
        destination = output_dir / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(blob)
        files[path] = _sha256(blob)
    manifest = {"schema_version": 1, "commit": commit, "files": files, "digest": measurement_digest(files)}
    (output_dir / "measurement-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def prepare_manifest(root: Path, measurement_manifest: dict | None = None) -> dict:
    """Hash immutable Python blobs and verify their checked-out contents before any measurement overlay."""
    root = root.resolve()
    commit = _git(root, "rev-parse", "HEAD").decode().strip()
    blobs = _tracked_blobs(root, commit, ("source", "scripts"), python_only=True)
    if not blobs:
        raise ValueError("The selected commit contains no Python source under source/ or scripts/.")
    files = {}
    for path, blob in blobs.items():
        expected = _sha256(blob)
        try:
            actual = (root / path).read_bytes()
        except OSError as exc:
            raise ValueError(f"Cannot read checked-out Python source {path}: {exc.strerror}") from exc
        if _sha256(actual) != expected:
            raise ValueError(f"Checked-out Python source differs from commit {commit}: {path}")
        files[path] = expected
    namespaces = {}
    for path in files:
        parts = PurePosixPath(path).parts
        # Test directories are not installed namespaces, even when they have __init__.py.
        if len(parts) >= 4 and parts[0] == "source" and parts[2] == parts[1].replace("-", "_"):
            prefix = "/".join(parts[:3])
            namespaces.setdefault(parts[2], set()).add(prefix)
    manifest = {
        "schema_version": 1,
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
        manifest.update(schema_version=2, measurement=measurement_manifest)
    return manifest


def _expected_paths(name: str, namespaces: dict) -> list[str]:
    namespace, _, suffix = name.partition(".")
    paths = []
    for prefix in namespaces.get(namespace, []):
        stem = prefix + ("/" + suffix.replace(".", "/") if suffix else "")
        paths.extend((stem + ".py", stem + "/__init__.py"))
    return paths


def _inspect_modules(manifest: dict, root: Path) -> tuple[list[dict], list[dict]]:
    records, mismatches = [], []
    measurement_files = manifest.get("measurement", {}).get("files", {})
    files = manifest["files"] | measurement_files
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
        if not expected_paths and relative not in files:
            continue
        expected = files.get(relative)
        record = {
            "name": name,
            "path": str(path),
            "relative_path": relative,
            "sha256": None,
            "expected_sha256": expected,
            "status": "verified",
        }
        if "measurement" in manifest:
            record["origin"] = (
                "measurement"
                if relative in measurement_files or any(path in measurement_files for path in expected_paths)
                else "workload"
            )
        reasons = []
        if expected_paths and relative not in expected_paths:
            reasons.append("Imported module origin does not match the selected checkout module path.")
        if expected is None:
            reasons.append("Imported Python source is absent from the selected commit manifest.")
        try:
            record["sha256"] = _sha256(path.read_bytes())
        except OSError as exc:
            reasons.append(f"Imported module source could not be read: {exc.strerror}.")
        if expected is not None and record["sha256"] != expected:
            reasons.append("Imported Python source bytes differ from the selected commit.")
        if reasons:
            record["status"] = "failed"
            mismatches.extend({"module": name, "path": str(path), "reason": reason} for reason in reasons)
        records.append(record)
    return records, mismatches


def _inspect_measurement(manifest: dict, root: Path) -> tuple[dict, list[dict]]:
    measurement = manifest["measurement"]
    mismatches = []
    if measurement_digest(measurement["files"]) != measurement["digest"]:
        mismatches.append({"reason": "Measurement manifest digest does not match its files."})
    for relative, expected in measurement["files"].items():
        try:
            actual = _sha256((root / relative).read_bytes())
            if actual != expected:
                mismatches.append({"path": relative, "reason": "Measurement source differs from the tested revision."})
        except OSError as exc:
            mismatches.append({"path": relative, "reason": f"Measurement source could not be read: {exc.strerror}."})
    return dict(measurement), mismatches


def _inspect_runtime() -> tuple[dict | None, list[dict]]:
    module = sys.modules.get(RUNTIME_MODULE)
    function = getattr(module, "run", None)
    code = getattr(function, "__code__", None)
    if not isinstance(code, types.CodeType):
        return None, [{"module": RUNTIME_MODULE, "reason": "Runtime entrypoint was not imported as a Python function."}]
    record = {"module": RUNTIME_MODULE, "source_code_matches": False}
    mismatches = []
    try:
        source = Path(module.__file__).read_bytes()
        compiled = compile(source, code.co_filename, "exec", dont_inherit=True, optimize=sys.flags.optimize)
        expected = next(
            item for item in compiled.co_consts if isinstance(item, types.CodeType) and item.co_name == "run"
        )
        record["source_code_matches"] = code == expected
        if not record["source_code_matches"]:
            mismatches.append(
                {"module": RUNTIME_MODULE, "reason": "Live runtime function code differs from its source."}
            )
    except (OSError, SyntaxError, StopIteration, TypeError, ValueError) as exc:
        mismatches.append({"module": RUNTIME_MODULE, "reason": f"Runtime function source comparison failed: {exc}"})
    return record, mismatches


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
        runtime, runtime_mismatches = _inspect_runtime()
        mismatches.extend(runtime_mismatches)
        measurement = None
        if "measurement" in manifest:
            measurement, measurement_mismatches = _inspect_measurement(manifest, root)
            mismatches.extend(measurement_mismatches)
            measurement_failed = (
                measurement_mismatches
                or runtime_mismatches
                or any(item["origin"] == "measurement" and item["status"] != "verified" for item in modules)
            )
            measurement["status"] = "failed" if measurement_failed else "verified"
            if runtime is not None:
                runtime["origin"] = "measurement"
        outputs = []
        for path, state in sorted(_output_state(output_dir).items()):
            if before.get(path) == state:
                continue
            data = path.read_bytes()
            outputs.append(
                {"path": path.relative_to(output_dir).as_posix(), "sha256": _sha256(data), "bytes": len(data)}
            )
        if not outputs:
            mismatches.append({"reason": "The benchmark invocation produced no runtime JSON output."})
        verified = exit_code == 0 and not mismatches
        sidecar = {
            "schema_version": 1,
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
        if measurement is not None:
            sidecar.update(schema_version=2, measurement=measurement)
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
    prepare.add_argument("--measurement-manifest", type=Path)
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
            manifest = prepare_manifest(args.checkout_root, measurement)
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
