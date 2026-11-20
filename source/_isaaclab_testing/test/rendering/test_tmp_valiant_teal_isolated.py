# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Temporary NVBug 7071342 check: run the Isaac Sim-only Valiant probe on the CI GPU. Do not merge."""

import glob
import json
import os
import subprocess
from pathlib import Path

import pytest

_PROBE = Path(__file__).with_name("tmp_valiant_teal_probe_isolated.py")
_OUT = Path.cwd() / "tests" / "comparison-images"


def _isaac_sim_root() -> Path:
    for candidate in (os.environ.get("ISAACSIM_ROOT_PATH"), Path(__file__).parents[4] / "_isaac_sim", "/isaac-sim"):
        if candidate and (Path(candidate) / "python.sh").exists():
            return Path(candidate)
    pytest.skip("Isaac Sim python.sh not found")


@pytest.mark.parametrize(("profile", "iterations"), [("isaaclab", 150), ("isaacsim", 100)])
def test_valiant_teal_isolated(profile: str, iterations: int, request: pytest.FixtureRequest) -> None:
    root = _isaac_sim_root()
    out = _OUT / f"teal-isolated-{profile}"
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMNI_KIT_ACCEPT_EULA="YES")
    # Isaac Sim's bundled PyTorch can fail to resolve the CUDA runtime at SimulationApp startup; Kit ships one.
    cudart = sorted(glob.glob(str(root / "extscache/omni.gpu_foundation-*/bin/deps/libcudart.so.12*")))
    if cudart:
        env["LD_PRELOAD"] = cudart[-1]
    iterations = int(os.environ.get("TEAL_PROBE_ITERATIONS", iterations))
    command = [str(root / "python.sh"), str(_PROBE), "--iterations", str(iterations), "--profile", profile]
    result = subprocess.run(
        [*command, "--out", str(out)], env=env, capture_output=True, text=True, timeout=3600, check=False
    )
    (out / "probe.log").write_text(result.stdout + result.stderr)
    summary_path = out / "summary.json"
    assert summary_path.exists(), f"probe produced no summary (exit {result.returncode}); see {out / 'probe.log'}"
    summary = json.loads(summary_path.read_text())
    request.node.user_properties.extend(
        [
            ("kit", summary["kit"]),
            ("iterations", str(summary["iterations"])),
            ("teal_first", str(summary["teal_first"])),
            ("teal_later", str(summary["teal_later"])),
        ]
    )
    print(f"{profile}: teal first {summary['teal_first']}/{summary['iterations']}, later {summary['teal_later']}")
