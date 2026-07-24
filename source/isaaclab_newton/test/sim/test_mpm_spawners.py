# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import subprocess
import sys
import textwrap


def test_mpm_config_imports_do_not_load_pxr():
    code = textwrap.dedent(
        """
        import sys

        from isaaclab_newton.assets.mpm_object import MPMObjectCfg
        from isaaclab_newton.sim.spawners.mpm import MPMGridCfg, MPMParticleMaterialCfg, MPMPointsCfg

        MPMObjectCfg(
            prim_path="/World/envs/env_.*/Sand",
            spawn=MPMGridCfg(
                lower=(0.0, 0.0, 0.0),
                upper=(0.1, 0.1, 0.1),
                voxel_size=0.1,
                material=MPMParticleMaterialCfg(),
            ),
        )

        loaded_pxr_modules = [module for module in sys.modules if module == "pxr" or module.startswith("pxr.")]
        if loaded_pxr_modules:
            raise SystemExit("pxr loaded before SimulationApp: " + ", ".join(loaded_pxr_modules[:20]))
        """
    )

    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stdout + result.stderr


def test_granular_demo_does_not_load_pxr_before_app_launcher():
    code = textwrap.dedent(
        """
        import sys
        import types
        import importlib.machinery
        import importlib.util

        sys.argv = ["newton_mpm_granular.py", "--max-steps", "1", "--visualizer", "kit"]

        isaacsim = types.ModuleType("isaacsim")
        isaacsim.SimulationApp = type("SimulationApp", (), {})
        sys.modules["isaacsim"] = isaacsim

        original_find_spec = importlib.util.find_spec
        importlib.util.find_spec = lambda name, *args: (
            importlib.machinery.ModuleSpec(name, loader=None)
            if name == "omni.kit"
            else original_find_spec(name, *args)
        )

        import scripts.demos.mpm.newton_mpm_granular as demo
        from isaaclab.app.app_launcher import AppLauncher

        def stop_before_simulation_app(self, *args, **kwargs):
            loaded_pxr_modules = [module for module in sys.modules if module == "pxr" or module.startswith("pxr.")]
            if loaded_pxr_modules:
                raise SystemExit("pxr loaded before SimulationApp: " + ", ".join(loaded_pxr_modules[:20]))
            raise RuntimeError("stop before SimulationApp")

        AppLauncher.__init__ = stop_before_simulation_app

        try:
            demo.main()
        except RuntimeError as exc:
            if str(exc) != "stop before SimulationApp":
                raise
        """
    )

    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stdout + result.stderr


def test_particle_pour_demo_does_not_load_pxr_before_app_launcher():
    code = textwrap.dedent(
        """
        import sys
        import types
        import importlib.machinery
        import importlib.util

        sys.argv = ["particle_pour.py", "--max-steps", "1", "--visualizer", "kit"]

        isaacsim = types.ModuleType("isaacsim")
        isaacsim.SimulationApp = type("SimulationApp", (), {})
        sys.modules["isaacsim"] = isaacsim

        original_find_spec = importlib.util.find_spec
        importlib.util.find_spec = lambda name, *args: (
            importlib.machinery.ModuleSpec(name, loader=None)
            if name == "omni.kit"
            else original_find_spec(name, *args)
        )

        import scripts.demos.mpm.particle_pour as demo
        from isaaclab.app.app_launcher import AppLauncher

        def stop_before_simulation_app(self, *args, **kwargs):
            loaded_pxr_modules = [module for module in sys.modules if module == "pxr" or module.startswith("pxr.")]
            if loaded_pxr_modules:
                raise SystemExit("pxr loaded before SimulationApp: " + ", ".join(loaded_pxr_modules[:20]))
            raise RuntimeError("stop before SimulationApp")

        AppLauncher.__init__ = stop_before_simulation_app

        try:
            demo.main()
        except RuntimeError as exc:
            if str(exc) != "stop before SimulationApp":
                raise
        """
    )

    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stdout + result.stderr
