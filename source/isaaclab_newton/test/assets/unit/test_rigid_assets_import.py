# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Runtime guard for the kitless Newton asset suites."""

import os
import subprocess
import sys
from pathlib import Path

_ASSET_TEST_DIR = Path(__file__).resolve().parents[1]
_TARGETS = (
    ("test_rigid_object.py", "test_rigid_object_real_newton_seams[cpu]"),
    ("test_rigid_object_collection.py", "test_rigid_object_collection_real_newton_seams"),
    ("test_articulation.py", "test_articulation_real_newton_seams"),
    ("test_newton_actuators_newton.py", "test_newton_actuator_real_equivalence"),
    ("../controllers/test_newton_task_space_controllers.py", "test_differential_ik_tracks_local_newton_chain"),
    (
        "../controllers/test_newton_task_space_controllers.py",
        "test_operational_space_consumes_newton_jacobian_mass_and_gravity",
    ),
    (
        "../controllers/test_newton_task_space_controllers.py",
        "test_operational_space_gravity_compensation_holds_static_chain",
    ),
)

_TARGET_FILENAMES = tuple(Path(target).name for target, _ in _TARGETS) + ("articulation_test_utils.py",)


def _run_monitored_targets(nodes: tuple[str, ...], tmp_path: Path) -> subprocess.CompletedProcess[str]:
    """Run target nodes together under import and Nucleus sentinels."""
    sitecustomize = tmp_path / "sitecustomize.py"
    sitecustomize.write_text(
        """
import importlib.abc
import sys
import isaaclab.utils.assets as assets


class _ForbiddenFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        frame = sys._getframe(1)
        while frame and (
            frame.f_code.co_filename.startswith("<frozen importlib")
            or frame.f_code.co_filename == __file__
        ):
            frame = frame.f_back
        imported_by_target = frame is not None and frame.f_code.co_filename.endswith(__TARGET_FILENAMES__)
        if fullname.startswith("isaacsim") or (
            fullname.startswith("isaaclab.app") and (imported_by_target or fullname == "isaaclab.app.app_launcher")
        ):
            raise ImportError(f"forbidden kit dependency imported: {fullname}")
        return None


class _ForbiddenNucleusPath(str):
    def _guard(self):
        caller = sys._getframe(2).f_code.co_filename
        if caller.endswith(__TARGET_FILENAMES__):
            raise RuntimeError("forbidden Nucleus asset used by Newton asset test")

    def __format__(self, format_spec):
        self._guard()
        return super().__format__(format_spec)

    def __add__(self, other):
        self._guard()
        return super().__add__(other)


sys.meta_path.insert(0, _ForbiddenFinder())
assets.ISAAC_NUCLEUS_DIR = _ForbiddenNucleusPath(assets.ISAAC_NUCLEUS_DIR)
assets.ISAACLAB_NUCLEUS_DIR = _ForbiddenNucleusPath(assets.ISAACLAB_NUCLEUS_DIR)
""".replace("__TARGET_FILENAMES__", repr(_TARGET_FILENAMES)),
        encoding="utf-8",
    )
    env = os.environ | {"PYTHONPATH": str(tmp_path)}
    return subprocess.run(
        [sys.executable, "-m", "pytest", *nodes, "-q"],
        cwd=_ASSET_TEST_DIR,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_newton_asset_cpu_seams_run_without_kit_isaacsim_or_nucleus(tmp_path: Path) -> None:
    """Run all real CPU seams while rejecting Kit, IsaacSim, and Nucleus access."""
    nodes = tuple(f"{_ASSET_TEST_DIR / target}::{node}" for target, node in _TARGETS)
    result = _run_monitored_targets(nodes, tmp_path)

    assert result.returncode == 0, result.stdout + result.stderr


def test_runtime_nucleus_access_inside_fixture_is_rejected(tmp_path: Path) -> None:
    """The guard must execute fixture helpers instead of stopping after collection."""
    source_target = _ASSET_TEST_DIR / "test_rigid_object.py"
    mutated_target = tmp_path / source_target.name
    mutated_target.write_text(
        source_target.read_text(encoding="utf-8").replace(
            '    """Author two local dynamic cuboids and return their Newton asset."""',
            '    """Author two local dynamic cuboids and return their Newton asset."""\n'
            "    from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR\n"
            '    _ = ISAAC_NUCLEUS_DIR + "/forbidden.usd"',
        ),
        encoding="utf-8",
    )

    result = _run_monitored_targets((f"{mutated_target}::test_rigid_object_real_newton_seams[cpu]",), tmp_path)

    assert result.returncode != 0
    assert "forbidden Nucleus asset used by Newton asset test" in result.stdout + result.stderr


def test_runtime_app_launcher_import_inside_fixture_is_rejected(tmp_path: Path) -> None:
    """The guard must reject AppLauncher imports reached only while a fixture helper executes."""
    source_target = _ASSET_TEST_DIR / "test_rigid_object.py"
    mutated_target = tmp_path / source_target.name
    mutated_target.write_text(
        source_target.read_text(encoding="utf-8").replace(
            '    """Author two local dynamic cuboids and return their Newton asset."""',
            '    """Author two local dynamic cuboids and return their Newton asset."""\n'
            "    from isaaclab.app import AppLauncher",
        ),
        encoding="utf-8",
    )

    result = _run_monitored_targets((f"{mutated_target}::test_rigid_object_real_newton_seams[cpu]",), tmp_path)

    assert result.returncode != 0
    assert "forbidden kit dependency imported: isaaclab.app.app_launcher" in result.stdout + result.stderr
