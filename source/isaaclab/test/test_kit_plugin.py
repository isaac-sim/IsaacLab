# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the marker-driven Kit launch plugin.

The claim the whole arrangement rests on is that :func:`isaaclab.test.kit.pytest_collectstart`
runs *before* pytest imports the test module. If it ever ran after, every ``kit`` file would
fail at its first ``import omni``, so the ordering is asserted here directly rather than left
to CI to discover.

The launch itself is stubbed out. Booting Kit is what these tests exist to schedule correctly,
not something they need to do.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import isaaclab.test.kit as kit

pytestmark = pytest.mark.unit

_LOG_ENV_VAR = "ISAACLAB_KIT_PLUGIN_PROBE_LOG"

_CONFTEST = """\
    import os

    import isaaclab.test.kit as kit


    def _record(entry):
        with open(os.environ["{log_env_var}"], "a", encoding="utf-8") as handle:
            handle.write(entry + "\\n")


    def _fake_launch(*, cameras):
        _record(f"launch:{{cameras}}")
        return object()


    kit._launch = _fake_launch
"""

_PYTEST_INI = """\
    [pytest]
    markers =
        kit: needs a headless Kit app
        kit_cameras: needs a Kit app with cameras enabled
"""

_MODULE = """\
    import os

    import pytest

    with open(os.environ["{log_env_var}"], "a", encoding="utf-8") as handle:
        handle.write("import:{name}\\n")

    {pytestmark}


    def test_placeholder({args}):
        pass
"""


def _write_module(directory: Path, name: str, *, marker: str | None, args: str = "") -> None:
    """Write a scratch test module that logs its own import."""
    (directory / f"test_{name}.py").write_text(
        textwrap.dedent(_MODULE).format(
            log_env_var=_LOG_ENV_VAR,
            name=name,
            pytestmark=f"pytestmark = pytest.mark.{marker}" if marker else "",
            args=args,
        ),
        encoding="utf-8",
    )


def _run_scratch_pytest(directory: Path) -> tuple[subprocess.CompletedProcess, list[str]]:
    """Run pytest over ``directory`` with the plugin loaded, and return its output and log.

    The scratch directory carries its own ``pytest.ini`` so it becomes the rootdir, which keeps
    the repo's own conftest -- and therefore the real, unstubbed plugin -- out of the run.
    """
    (directory / "conftest.py").write_text(
        textwrap.dedent(_CONFTEST).format(log_env_var=_LOG_ENV_VAR), encoding="utf-8"
    )
    (directory / "pytest.ini").write_text(textwrap.dedent(_PYTEST_INI), encoding="utf-8")
    log = directory / "probe.log"

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "isaaclab.test.kit", "-q", "-p", "no:cacheprovider"],
        cwd=directory,
        env={**os.environ, _LOG_ENV_VAR: str(log)},
        capture_output=True,
        text=True,
        timeout=300,
    )
    entries = log.read_text(encoding="utf-8").splitlines() if log.exists() else []
    return result, entries


def test_kit_is_launched_before_the_module_is_imported(tmp_path: Path):
    """The marker only works if the app is up before the module's own imports run."""
    _write_module(tmp_path, "marked", marker="kit")
    result, entries = _run_scratch_pytest(tmp_path)

    assert result.returncode == 0, result.stdout + result.stderr
    assert entries == ["launch:False", "import:marked"], (
        f"expected the launch to precede the import, got {entries}\n{result.stdout}"
    )


def test_the_marker_selects_the_camera_setting(tmp_path: Path):
    """`kit_cameras` must reach AppLauncher as ``enable_cameras=True``, and `kit` as False."""
    _write_module(tmp_path, "plain", marker="kit")
    _write_module(tmp_path, "with_cameras", marker="kit_cameras")
    result, entries = _run_scratch_pytest(tmp_path)

    assert result.returncode == 0, result.stdout + result.stderr
    launches = {entry for entry in entries if entry.startswith("launch:")}
    assert launches == {"launch:False", "launch:True"}


def test_an_unmarked_module_does_not_launch_anything(tmp_path: Path):
    """Most of the suite is unmarked, and collecting it must stay Kit-free."""
    _write_module(tmp_path, "unmarked", marker=None)
    result, entries = _run_scratch_pytest(tmp_path)

    assert result.returncode == 0, result.stdout + result.stderr
    assert entries == ["import:unmarked"]


def test_requesting_kit_app_without_a_marker_says_what_is_missing(tmp_path: Path):
    """The fixture is only meaningful in a module that declared a launch marker."""
    _write_module(tmp_path, "unmarked", marker=None, args="kit_app")
    result, _ = _run_scratch_pytest(tmp_path)

    assert result.returncode != 0
    assert "no Kit app is running" in result.stdout


def test_a_second_configuration_in_one_process_is_refused(monkeypatch: pytest.MonkeyPatch):
    """`kit` and `kit_cameras` files in one process must fail loudly, not silently share."""
    monkeypatch.setattr(kit, "_app", object())
    monkeypatch.setattr(kit, "_cameras", False)

    with pytest.raises(RuntimeError, match="cannot be changed after startup"):
        kit._launch(cameras=True)


def test_an_app_started_by_something_else_is_refused(monkeypatch: pytest.MonkeyPatch):
    """An app of unknown configuration cannot be handed to a file that asked for a known one."""
    monkeypatch.setattr(kit, "_app", None)
    monkeypatch.setattr("isaaclab.utils.has_kit", lambda: True)

    with pytest.raises(RuntimeError, match="not started by this plugin"):
        kit._launch(cameras=False)
