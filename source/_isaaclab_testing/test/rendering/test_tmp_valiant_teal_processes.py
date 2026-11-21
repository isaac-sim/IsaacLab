# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Temporary NVBug 7071342 probe: Valiant in fresh processes that start from the CI job's initial cache state.

On CI, ``test_rendering_kit.py``'s Valiant RGB case is the first Gaussian-splat load of the job: it compiles the splat
shaders and kernels (about 12 s, against 2-4 s for later loads), because the restored driver shader cache comes from a
cache-warming job that renders no splats and the CUDA and OptiX caches start empty. This wrapper must be the first
renderer test of the job. It snapshots those caches before any splat has loaded, then restores the snapshot before
every process, so each process compiles the splat shaders and kernels again. Do not merge.
"""

import glob
import json
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

_HERE = Path(__file__).parent
_ROOT = Path(__file__).parents[4]
_OUT = Path.cwd() / "tests" / "comparison-images" / "teal-processes"
_SNAPSHOT = Path("/tmp/nvbug7071342-job-start-caches")
_VARIANTS = [("isolated-jobstart", 16), ("kit-none-jobstart", 16), ("kit-all-jobstart", 6)]


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


def _run_isolated(label: str) -> dict:
    sim = _isaac_sim_root()
    if sim is None:
        pytest.skip("Isaac Sim python.sh not found")
    out = _OUT / label
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMNI_KIT_ACCEPT_EULA="YES")
    # Isaac Sim's bundled PyTorch can fail to resolve the CUDA runtime at SimulationApp startup; Kit ships one.
    cudart = sorted(glob.glob(str(sim / "extscache/omni.gpu_foundation-*/bin/deps/libcudart.so.12*")))
    if cudart:
        env["LD_PRELOAD"] = cudart[-1]
    command = [str(sim / "python.sh"), str(_HERE / "tmp_valiant_teal_probe_isolated.py"), "--iterations", "1"]
    result = subprocess.run(
        [*command, "--out", str(out)], env=env, capture_output=True, text=True, timeout=1200, check=False
    )
    (out / "probe.log").write_text(result.stdout[-20000:] + result.stderr[-20000:])
    summary_path = out / "summary.json"
    if not summary_path.exists():
        return {"returncode": result.returncode, "teal": False}
    record = json.loads(summary_path.read_text())["records"][0]
    hdr = record["first"]["hdr"]
    return {
        "returncode": result.returncode,
        "seconds": record["seconds"],
        "first": record["first"],
        "later_ldr_teal": record["later"]["ldr"]["teal"],
        "teal": record["first"]["ldr"]["teal"] or hdr["greenish"] or hdr["luma_contrast"] < 0.4,
    }


@pytest.mark.parametrize(("variant", "processes"), _VARIANTS, ids=[variant for variant, _ in _VARIANTS])
def test_valiant_teal_processes(variant: str, processes: int, request: pytest.FixtureRequest) -> None:
    processes = int(os.environ.get("TEAL_PROCESSES", processes))
    _OUT.mkdir(parents=True, exist_ok=True)
    records = []
    for index in range(processes):
        label = f"{variant}-p{index:02d}"
        restored = _reset_to_job_start()
        if variant == "isolated-jobstart":
            outputs = _run_isolated(label)
        else:
            outputs = _run_isaaclab(label, "none" if variant == "kit-none-jobstart" else "all")
        records.append({"process": index, "restored_bytes": restored, "growth_bytes": _growth(restored), **outputs})
        (_OUT / f"summary-{variant}.json").write_text(json.dumps(records, indent=1, default=str))
    teal = sum(record["teal"] for record in records)
    request.node.user_properties.extend([("processes", str(len(records))), ("teal_processes", str(teal))])
    print(f"{variant}: teal in {teal} of {len(records)} processes")
