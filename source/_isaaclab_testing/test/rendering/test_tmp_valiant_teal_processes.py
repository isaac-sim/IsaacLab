# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Temporary NVBug 7071342 probe: the start of ``test_rendering_kit.py`` in fresh processes. Do not merge.

On CI the teal Valiant frame appears in the first Kit process of a job, which starts from the restored driver shader
cache, empty CUDA, OptiX and Kit caches, and no downloaded assets. This wrapper must be the first renderer test of the
job. It snapshots those caches before anything renders, then restores that state before every process and runs the 20
cases that precede Valiant followed by the Valiant RGB and HDR cases, to measure how often the teal frame appears.
"""

import glob
import json
import os
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

_HERE = Path(__file__).parent
_ROOT = Path(__file__).parents[4]
_OUT = Path.cwd() / "tests" / "comparison-images" / "teal-processes"
_SNAPSHOT = Path("/tmp/nvbug7071342-job-start-caches")
_KIT_LOG_GLOBS = (
    "{sim}/kit/logs/**/*.log",
    "~/.nvidia-omniverse/logs/**/*.log",
    "/isaac-sim/.nvidia-omniverse/logs/**/*.log",
    "~/.cache/packman/chk/kit-kernel/*/logs/**/*.log",
)
_VARIANTS = [("regular-jobstart", 45)]
# Isaac Lab copies remote assets here; a fresh CI job downloads them again.
_ASSET_CACHES = (Path("/tmp/https"), Path("/tmp/http"))


def _isaac_sim_root() -> Path | None:
    for candidate in (os.environ.get("ISAACSIM_ROOT_PATH"), _ROOT / "_isaac_sim", "/isaac-sim"):
        if candidate and (Path(candidate) / "python.sh").exists():
            return Path(candidate)
    return None


def _restored_caches() -> dict[str, Path]:
    """Caches the first renderer process of a CI job starts with, keyed by snapshot name."""
    home = Path.home()
    caches = {
        "home_compute": home / ".nv" / "ComputeCache",
        "isaacsim_compute": Path("/isaac-sim/.nv/ComputeCache"),
        "gl": Path(os.environ.get("XDG_CACHE_HOME", home / ".cache")) / "nvidia",
    }
    sim = _isaac_sim_root()
    if sim is not None:
        caches["nv_shadercache"] = sim / "kit" / "cache" / "nv_shadercache"
    for path in glob.glob("/var/tmp/OptixCache*"):
        caches[f"optix_{Path(path).name}"] = Path(path)
    return caches


def _size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) if path.exists() else 0


def _clear(path: Path) -> None:
    # Several caches are bind mounts, so empty them rather than removing the directory.
    if not path.exists():
        return
    for child in path.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink(missing_ok=True)


def _reset_to_job_start() -> dict[str, int]:
    # Never touch a developer's caches; on CI the container and runner are disposable.
    if not os.environ.get("GITHUB_ACTIONS"):
        return {}
    caches = _restored_caches()
    if not _SNAPSHOT.exists():
        for name, path in caches.items():
            if path.exists():
                shutil.copytree(path, _SNAPSHOT / name, symlinks=True)
    sizes = {}
    for name, path in caches.items():
        _clear(path)
        if (_SNAPSHOT / name).exists():
            path.mkdir(parents=True, exist_ok=True)
            shutil.copytree(_SNAPSHOT / name, path, symlinks=True, dirs_exist_ok=True)
        sizes[name] = _size(path)
    for path in _ASSET_CACHES:
        sizes[f"assets_{path.name}"] = _size(path)
        shutil.rmtree(path, ignore_errors=True)
    sim = _isaac_sim_root()
    if sim is not None:
        # Everything else under Kit's cache directory starts empty in a CI job.
        for child in (sim / "kit" / "cache").iterdir():
            if child.name == "nv_shadercache":
                continue
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)
    return sizes


def _kit_logs_since(start: float) -> list[Path]:
    sim = _isaac_sim_root() or Path("/nonexistent")
    logs = set()
    for pattern in _KIT_LOG_GLOBS:
        for path in glob.glob(os.path.expanduser(pattern.format(sim=sim)), recursive=True):
            if os.path.getmtime(path) >= start:
                logs.add(Path(path))
    return sorted(logs)


def _growth(before: dict[str, int]) -> dict[str, int]:
    if not before:
        return {}
    return {name: _size(path) - before.get(name, 0) for name, path in _restored_caches().items()}


def _run_isaaclab(label: str, cases: str) -> dict:
    xml = _OUT / f"{label}.xml"
    env = dict(os.environ, TEAL_ORDER_CASES=cases, TEAL_ORDER_REPEATS="1", TEAL_ORDER_LABEL=label)
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        f"--config-file={_ROOT}/pyproject.toml",
        f"--junitxml={xml}",
        str(_HERE / "test_tmp_valiant_teal_order.py"),
    ]
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=1500, check=False)
    (_OUT / f"{label}.log").write_text(result.stdout[-20000:] + result.stderr[-20000:])
    outputs = {"returncode": result.returncode}
    if xml.exists():
        for case in ET.parse(xml).getroot().iter("testcase"):
            props = {p.get("name"): p.get("value") for p in case.iter("property")}
            for key, value in props.items():
                if key.startswith("ssim:") and "valiant_1env" in key and "isaac_rtx" in key:
                    kind = "hdr" if key.endswith("rgb_hdr") else "rgb"
                    outputs[kind] = {"ssim": float(value), "seconds": float(case.get("time") or 0)}
    outputs["teal"] = any(outputs.get(kind, {}).get("ssim", 1.0) < 0.5 for kind in ("rgb", "hdr"))
    return outputs


@pytest.mark.parametrize(("variant", "processes"), _VARIANTS, ids=[variant for variant, _ in _VARIANTS])
def test_valiant_teal_processes(variant: str, processes: int, request: pytest.FixtureRequest) -> None:
    processes = int(os.environ.get("TEAL_PROCESSES", processes))
    _OUT.mkdir(parents=True, exist_ok=True)
    records = []
    for index in range(processes):
        label = f"{variant}-p{index:02d}"
        restored = _reset_to_job_start()
        start = time.time()
        outputs = _run_isaaclab(label, "all")
        kit_logs = _kit_logs_since(start)
        outputs["kit_logs"] = [str(path) for path in kit_logs]
        # Keep the full Kit log of every teal process, and of the first one as a clean reference.
        if outputs["teal"] or index == 0:
            for path in kit_logs:
                shutil.copy2(path, _OUT / f"{label}-{path.name}")
        records.append({"process": index, "restored_bytes": restored, "growth_bytes": _growth(restored), **outputs})
        (_OUT / f"summary-{variant}.json").write_text(json.dumps(records, indent=1, default=str))
    teal = sum(record["teal"] for record in records)
    request.node.user_properties.extend([("processes", str(len(records))), ("teal_processes", str(teal))])
    print(f"{variant}: teal in {teal} of {len(records)} processes")
