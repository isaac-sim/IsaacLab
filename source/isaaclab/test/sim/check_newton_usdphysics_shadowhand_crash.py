# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Standalone reproducer for the Newton/UsdPhysics rigid-body-descriptor native crash.

Builds ``Isaac-Shadow-Handover-Direct`` on the Newton (``physics=newton_mjwarp``) backend and
amplifies ``UsdPhysics.LoadUsdPhysicsFromRange`` on the environment subtree so the intermittent
(~2-3%/call) native NULL-deref in OpenUSD / usd-exchange
``UsdPhysics::moveDescsToDict<UsdPhysicsRigidBodyDesc>`` becomes near-deterministic in a single run.

This is a *process-crash* reproducer: on the unfixed OpenUSD/usd-exchange stack it terminates the
whole process with SIGSEGV / SIGABRT (exit 134 / 139 / 245). It therefore prints
``COMPLETED_NO_CRASH`` and exits 0 ONLY when the underlying bug is fixed. It is meant to be driven
as a subprocess by ``test_newton_usdphysics_shadowhand_repro.py`` so the parent can assert on the
exit code.

The amplification wraps the ``pxr`` binding in-process only -- it does NOT modify any installed
Newton/USD file.
"""

from __future__ import annotations

import argparse
import os
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=os.environ.get("NEWTON_REPRO_TASK", "Isaac-Shadow-Handover-Direct"))
    parser.add_argument("--num_envs", type=int, default=int(os.environ.get("NEWTON_REPRO_NUM_ENVS", "4")))
    parser.add_argument("--loops", type=int, default=int(os.environ.get("NEWTON_REPRO_LOOP_N", "60")))

    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # Optional: point the asset root at an offline mirror (e.g. staging S3) so the ShadowHand USD
    # resolves without Nucleus auth. No-op if the env var is unset.
    asset_root = os.environ.get("ISAAC_ASSET_ROOT")
    if asset_root:
        import carb

        carb.settings.get_settings().set("/persistent/isaac/asset_root/default", asset_root)
        carb.settings.get_settings().set("/rtx/verifyDriverVersion/enabled", False)

    # --- amplification: repeat the env-subtree physics load to make the rare crash near-certain.
    from pxr import UsdPhysics

    _orig_load = UsdPhysics.LoadUsdPhysicsFromRange
    _loops = max(1, int(args.loops))

    def _amplified_load(stage, include_paths, *rest, **kw):
        result = _orig_load(stage, include_paths, *rest, **kw)
        try:
            roots = [str(p) for p in include_paths]
        except TypeError:
            roots = [str(include_paths)]
        if any("/World/envs/env" in r for r in roots):
            for _ in range(_loops):
                _orig_load(stage, include_paths, *rest, **kw)
        return result

    UsdPhysics.LoadUsdPhysicsFromRange = _amplified_load

    # --- build the Newton env; the Newton model import (add_usd -> LoadUsdPhysicsFromRange) runs here.
    import gymnasium as gym

    import isaaclab_tasks  # noqa: F401  (registers Isaac Lab tasks)
    from isaaclab_tasks.utils import resolve_task_config

    env_cfg, _ = resolve_task_config(args.task, None, overrides=("physics=newton_mjwarp",))
    env_cfg.scene.num_envs = args.num_envs
    if hasattr(env_cfg, "seed"):
        env_cfg.seed = 0

    print(f"[repro] building {args.task} num_envs={args.num_envs} loops={_loops}", flush=True)
    env = gym.make(args.task, cfg=env_cfg)
    env.reset()
    env.close()

    print(f"COMPLETED_NO_CRASH task={args.task} num_envs={args.num_envs} loops={_loops}", flush=True)
    try:
        simulation_app.close()
    finally:
        os._exit(0)  # Isaac Sim teardown deadlocks; hard-exit so the no-crash path terminates


if __name__ == "__main__":
    sys.exit(main())
