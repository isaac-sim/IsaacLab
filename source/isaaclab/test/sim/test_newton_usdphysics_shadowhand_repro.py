# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression reproducer for the Newton / UsdPhysics rigid-body-descriptor native crash.

Building an Isaac Lab Newton (``physics=newton_mjwarp``) environment whose USD carries the Shadow
Hand rigid-body set intermittently crashes the whole process inside OpenUSD / usd-exchange
``UsdPhysics::moveDescsToDict<UsdPhysicsRigidBodyDesc>``, reached from
``UsdPhysics.LoadUsdPhysicsFromRange`` (called by ``newton.ModelBuilder.add_usd``). It is a native
memory fault, so it surfaces either as a SIGSEGV NULL-deref in ``moveDescsToDict`` or as a glibc
heap-corruption abort (``double free`` / ``unaligned tcache`` / ``free(): invalid next size``) --
both are the same out-of-bounds access.

Because the fault kills the process it cannot be caught in-process, so this test drives
``check_newton_usdphysics_shadowhand_crash.py`` as a subprocess. The helper amplifies the offending
call, turning the intermittent (~2-3 %/call) fault into a ~100 %/run one, and prints
``COMPLETED_NO_CRASH`` only if the environment builds without crashing.

Semantics: on the current (unfixed) OpenUSD / usd-exchange stack the subprocess is killed and this
test FAILS -- that failure *is* the reproduction. A fix in OpenUSD/usd-exchange ``UsdPhysics`` (or a
defensive guard in Newton ``import_usd``) makes the helper reach ``COMPLETED_NO_CRASH`` and this
test pass. The bug is NOT in Isaac Lab; the test lives here so a Newton/Isaac Sim bump that pulls in
the fix can be verified, and is meant to ship in the same PR as that fix.

The crash is Ada-only in practice (observed on L40 / B40, absent on Blackwell / Hopper), so the test
skips on non-Ada (non ``sm_89``) GPUs to avoid a misleading green result.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

_THIS_DIR = Path(__file__).resolve().parent
_HELPER = _THIS_DIR / "check_newton_usdphysics_shadowhand_crash.py"

# Substrings that indicate the native crash reproduced (any manifestation).
_CRASH_MARKERS = (
    "moveDescsToDict",            # NULL-deref manifestation (UsdPhysicsRigidBodyDesc)
    "LoadUsdPhysicsFromRange",    # offending call, present in the crash backtrace
    "crashreporter-breakpad",     # Kit breakpad engaged (both SIGSEGV and SIGABRT paths)
    "double free or corruption",  # glibc heap-corruption manifestations
    "unaligned tcache",
    "free(): invalid",
    "malloc(): ",
    "corrupted",
    "Segmentation fault",
)


def _find_isaaclab_sh() -> Path | None:
    for parent in _THIS_DIR.parents:
        candidate = parent / "isaaclab.sh"
        if candidate.is_file():
            return candidate
    return None


def _gpu_is_ada_only() -> bool:
    """True only if every visible GPU is Ada (compute capability 8.9)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            text=True,
            timeout=30,
        )
    except Exception:
        return False
    caps = [line.strip() for line in out.splitlines() if line.strip()]
    return bool(caps) and all(cap == "8.9" for cap in caps)


@pytest.mark.skipif(not _HELPER.is_file(), reason="reproducer helper script not found")
def test_newton_usdphysics_shadowhand_no_native_crash():
    """Isaac-Shadow-Handover-Direct on Newton must build without the UsdPhysics rigid-body crash."""
    try:
        import isaaclab  # noqa: F401
        import newton  # noqa: F401
    except Exception as exc:
        pytest.skip(f"Isaac Sim / Newton not importable: {exc}")

    if not _gpu_is_ada_only():
        pytest.skip("Newton/UsdPhysics rigid-body crash reproduces only on NVIDIA Ada (sm_89) GPUs")

    isaaclab_sh = _find_isaaclab_sh()
    if isaaclab_sh is None:
        pytest.skip("isaaclab.sh launcher not found; run under the Isaac Lab environment")

    env = dict(os.environ)
    env.setdefault("NEWTON_REPRO_LOOP_N", "100")
    num_envs = env.get("NEWTON_REPRO_NUM_ENVS", "4")

    proc = subprocess.run(
        [
            str(isaaclab_sh),
            "-p",
            str(_HELPER),
            "--task",
            "Isaac-Shadow-Handover-Direct",
            "--num_envs",
            num_envs,
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    output = (proc.stdout or "") + (proc.stderr or "")

    if "COMPLETED_NO_CRASH" in output:
        return  # environment built cleanly -> bug fixed / not present

    tail = output[-4000:]
    crashed = any(marker in output for marker in _CRASH_MARKERS)
    assert not crashed, (
        "Reproduced the Newton/UsdPhysics rigid-body-descriptor native crash "
        f"(launcher exit={proc.returncode}). NULL-deref / heap OOB in "
        "UsdPhysics::moveDescsToDict<UsdPhysicsRigidBodyDesc> via LoadUsdPhysicsFromRange "
        "(newton.ModelBuilder.add_usd). This is the bug under test; a fix in OpenUSD/usd-exchange "
        "UsdPhysics (or a Newton import_usd guard) makes this pass.\n"
        f"--- reproducer output tail ---\n{tail}"
    )
    pytest.fail(
        "Reproducer neither completed nor produced a known crash signature "
        f"(launcher exit={proc.returncode}); treat as an environment/setup failure, "
        f"not the target crash.\n--- reproducer output tail ---\n{tail}"
    )
