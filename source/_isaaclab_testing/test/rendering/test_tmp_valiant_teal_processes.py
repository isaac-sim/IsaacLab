# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Temporary NVBug 7071342 probe: Valiant in fresh processes that start with cold Kit caches. Do not merge.

On CI, ``test_rendering_kit.py`` is the first Kit process on a fresh runner, so its Valiant RGB case is the first load
of the Gaussian asset, with caches still cold. Every earlier probe ran after it and found warm caches. Each process
started here first wipes the caches that the first Kit process of a CI job starts without (CI restores only the
NVIDIA driver shader cache), then renders Valiant once.
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
_VARIANTS = [("isolated-cold", 30), ("cold-none", 20), ("cold-all", 10)]


def _isaac_sim_root() -> Path | None:
    for candidate in (os.environ.get("ISAACSIM_ROOT_PATH"), _ROOT / "_isaac_sim", "/isaac-sim"):
        if candidate and (Path(candidate) / "python.sh").exists():
            return Path(candidate)
    return None


def _cache_dirs() -> list[Path]:
    roots = [Path.home() / ".cache" / "ov"]
    sim = _isaac_sim_root()
    if sim is not None:
        roots += [sim / "kit" / "cache", sim / ".cache"]
    if os.environ.get("XDG_CACHE_HOME"):
        roots.append(Path(os.environ["XDG_CACHE_HOME"]))
    found = set()
    for root in roots:
        for name in ("DerivedDataCache", "shadercache", "texturecache"):
            for path in glob.glob(str(root / "**" / name), recursive=True):
                if "nv_shadercache" not in path:
                    found.add(Path(path))
    return sorted(found)


def _wipe_caches() -> dict[str, int]:
    # Never touch a developer's caches; on CI the container and runner are disposable.
    if not os.environ.get("GITHUB_ACTIONS"):
        return {}
    wiped = {}
    for path in _cache_dirs():
        wiped[str(path)] = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        shutil.rmtree(path, ignore_errors=True)
    return wiped


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
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=1200, check=False)
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
        [*command, "--out", str(out)], env=env, capture_output=True, text=True, timeout=900, check=False
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
        wiped = _wipe_caches()
        if variant == "isolated-cold":
            outputs = _run_isolated(label)
        else:
            outputs = _run_isaaclab(label, "none" if variant == "cold-none" else "all")
        records.append({"process": index, "wiped_bytes": wiped, **outputs})
        (_OUT / f"summary-{variant}.json").write_text(json.dumps(records, indent=1, default=str))
    teal = sum(record["teal"] for record in records)
    request.node.user_properties.extend([("processes", str(len(records))), ("teal_processes", str(teal))])
    print(f"{variant}: teal in {teal} of {len(records)} processes")
