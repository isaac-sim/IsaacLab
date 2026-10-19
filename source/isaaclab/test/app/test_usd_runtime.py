# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OpenUSD provider selection occurs before native imports and preserves kitless launches."""

import importlib.metadata
import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.unit


@pytest.fixture
def subprocess_env():
    """Use the installed providers without a manually selected USD runtime."""
    try:
        importlib.metadata.distribution("isaacsim-extscache-kit")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("Isaac Sim's bundled USD provider is not installed")
    env = os.environ.copy()
    for name in ("PYTHONPATH", "LD_LIBRARY_PATH", "ISAAC_PATH", "CARB_APP_PATH", "EXP_PATH"):
        env.pop(name, None)
    return env


@pytest.mark.parametrize("args", [("--viz", "kit"), ("--visualizer=kit,newton_gl",)])
def test_explicit_kit_uses_one_matching_usd_provider(subprocess_env, args):
    """Early Newton imports must use the same native and Python USD build as Kit."""
    code = """
import importlib.util
from pathlib import Path
import isaaclab
assert 'isaacsim' not in __import__('sys').modules
import newton
from pxr import Usd, UsdGeom
root = Path(importlib.util.find_spec('isaacsim').origin).parent
assert Path(Usd.__file__).is_relative_to(root), Usd.__file__
stage = Usd.Stage.CreateInMemory()
assert UsdGeom.Xform.Define(stage, '/World').GetPrim().IsValid()
print(Usd.GetVersion())
"""
    result = subprocess.run(
        [sys.executable, "-c", code, *args], env=subprocess_env, capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("args", [(), ("--visualizer", "newton_rtx"), ("--viz=kit", "--help")])
def test_kitless_and_help_preserve_standalone_usd(subprocess_env, args):
    """Core import and help do not bootstrap Kit or change the standalone provider."""
    code = """
import sys
from pathlib import Path
import isaaclab
assert 'pxr' not in sys.modules
assert 'isaacsim' not in sys.modules
from pxr import Usd
root = Path(__import__('importlib.metadata').metadata.distribution('usd-exchange').locate_file(''))
assert Path(Usd.__file__).is_relative_to(root / 'pxr'), Usd.__file__
"""
    result = subprocess.run(
        [sys.executable, "-c", code, *args], env=subprocess_env, capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("preloaded", ["fresh", "namespace", "native"])
def test_config_selected_kit_selects_or_rejects_loaded_usd(subprocess_env, preloaded):
    """Config-only callers select Kit before imports or fail before starting its native runtime."""
    code = """
import importlib.util
from pathlib import Path
import sys
if sys.argv[1] == 'namespace':
    import pxr
elif sys.argv[1] == 'native':
    from pxr import Usd
from isaaclab.app import SimulationLauncher, launch_simulation
from isaaclab.physics import PhysicsCfg
from isaaclab.sim import SimulationCfg
import isaaclab_physx.app

class CheckProvider(SimulationLauncher):
    def __init__(self, args):
        assert sys.argv[1] != 'native', 'Incompatible USD reached Kit startup'
        from pxr import Usd
        root = Path(importlib.util.find_spec('isaacsim').origin).parent
        assert Path(Usd.__file__).is_relative_to(root), Usd.__file__
        assert Usd.Stage.CreateInMemory()

isaaclab_physx.app.KitLauncher = CheckProvider
try:
    with launch_simulation(SimulationCfg(physics=PhysicsCfg(), device='cpu'), {'require_kit': True}):
        assert sys.argv[1] != 'native'
except RuntimeError as error:
    assert sys.argv[1] == 'native'
    assert 'OpenUSD' in str(error) and '--visualizer kit' in str(error), str(error)
else:
    assert sys.argv[1] != 'native', 'Incompatible OpenUSD was not rejected'
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(preloaded)],
        env=subprocess_env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
