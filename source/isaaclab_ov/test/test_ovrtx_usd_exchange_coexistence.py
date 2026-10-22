# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ovrtx renders in a process that has loaded usd-exchange, in either order.

usd-exchange 3.0.0 and ovrtx 0.5.0 shipped differently built MaterialX DLLs under the same names, and on Windows
whichever loaded second failed (nvbug 6817794). ovrtx loads its runtime only when a renderer is created, so every
case creates one, each in a fresh interpreter so that the load order is the case's own.
"""

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.windows_ci,
    pytest.mark.skipif(sys.platform != "win32", reason="the DLL name clash is specific to the Windows loader"),
]


def _load_usd():
    from pxr import Usd  # noqa: F401


def _load_materialx():
    import usdex.core  # noqa: F401

    from pxr import UsdMtlx  # noqa: F401


def _create_renderer():
    import ovrtx

    ovrtx.Renderer()


_CASES = {
    "ovrtx-alone": (_create_renderer,),
    "usd-first": (_load_usd, _create_renderer),
    "materialx-first": (_load_materialx, _create_renderer),
    "ovrtx-first": (_create_renderer, _load_materialx),
}

# Renders camera Cartpole through Isaac Lab, which loads both libraries.
_CARTPOLE_SMOKE = Path(__file__).resolve().parents[2] / "isaaclab/test/install_ci/misc/cartpole_training_smoke.py"


@pytest.mark.parametrize("case", [*_CASES, "isaaclab-render"])
def test_ovrtx_coexists_with_usd_exchange(case):
    script = [_CARTPOLE_SMOKE, "newton_mjwarp", "ovrtx"] if case == "isaaclab-render" else [__file__, case]
    result = subprocess.run([sys.executable, *script], capture_output=True, text=True, timeout=600, check=False)
    assert result.returncode == 0, f"{case} exited with {result.returncode}:\n{result.stdout}\n{result.stderr}"


if __name__ == "__main__":
    for step in _CASES[sys.argv[1]]:
        step()
