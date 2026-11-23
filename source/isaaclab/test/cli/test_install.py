# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for CLI utility functions used by the uv installation path."""

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest

from isaaclab.cli.utils import (
    extract_isaacsim_path,
    run_python_command,
)

pytestmark = pytest.mark.unit


@pytest.mark.skipif(os.name == "nt", reason="exercises the POSIX source-build launcher")
def test_source_build_cli_child_loads_kit_once(tmp_path: Path) -> None:
    """The CLI child loads native paths, keeps uv's Python, and does not recurse."""
    local_sim = tmp_path / "_isaac_sim"
    bindings = local_sim / "kit" / "plugins" / "bindings-python"
    bindings.mkdir(parents=True)
    (bindings / "kit_bootstrap_probe.py").write_text("VALUE = 'native binding loaded'\n")
    (local_sim / ".isaaclab_source_build").touch()
    launches = tmp_path / "launches"
    launcher = local_sim / "python.sh"
    launcher.write_text(
        '#!/bin/sh\necho launch >> "$BOOTSTRAP_LAUNCHES"\nexport ISAAC_PATH="$BOOTSTRAP_SIM"\nexec "$PYTHONEXE" "$@"\n'
    )
    launcher.chmod(0o755)
    # Configure discovery in both interpreters without changing the developer's local link.
    (tmp_path / "sitecustomize.py").write_text(
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "import isaaclab.cli as cli\n"
        "from isaaclab.cli import utils\n"
        "cli.DEFAULT_ISAAC_SIM_PATH = utils.DEFAULT_ISAAC_SIM_PATH = Path(os.environ['BOOTSTRAP_SIM'])\n"
        "if os.environ.get('ISAAC_PATH') == os.environ['BOOTSTRAP_SIM']:\n"
        "    import kit_bootstrap_probe\n"
        "    print(json.dumps({'python': sys.executable, 'binding': kit_bootstrap_probe.VALUE}))\n"
    )
    env = dict(os.environ)
    env.pop("ISAAC_PATH", None)
    env["BOOTSTRAP_SIM"] = str(local_sim)
    env["BOOTSTRAP_LAUNCHES"] = str(launches)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(tmp_path), env.get("PYTHONPATH")]))
    result = subprocess.run(
        [sys.executable, "-m", "isaaclab", "demo", "--help"],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    probe = next(json.loads(line) for line in result.stdout.splitlines() if line.startswith('{"python":'))
    assert Path(probe["python"]).resolve() == Path(sys.executable).resolve()
    assert probe["binding"] == "native binding loaded"
    assert "usage:" in result.stdout
    assert launches.read_text().splitlines() == ["launch"]


def test_run_python_command_preloads_system_libgomp_path_on_aarch64(tmp_path):
    """isaacsim starts on aarch64 only when the system libgomp path is listed verbatim in LD_PRELOAD."""
    with (
        mock.patch("isaaclab.cli.utils.DEFAULT_ISAAC_SIM_PATH", tmp_path / "_isaac_sim"),
        mock.patch("isaaclab.cli.utils.extract_python_exe", return_value=sys.executable),
        mock.patch("isaaclab.cli.utils.platform.system", return_value="Linux"),
        mock.patch("isaaclab.cli.utils.platform.machine", return_value="aarch64"),
        mock.patch("isaaclab.cli.utils.glob.glob", return_value=["/lib/aarch64-linux-gnu/libgomp.so.1"]),
        mock.patch("isaaclab.cli.utils.run_command") as run,
        mock.patch.dict(os.environ, {"LD_PRELOAD": "libcarb.env.shim.so"}, clear=True),
    ):
        run_python_command("train.py", [])

    assert run.call_args.kwargs["env"]["LD_PRELOAD"] == "/lib/aarch64-linux-gnu/libgomp.so.1:libcarb.env.shim.so"


def test_run_python_command_accepts_virtual_environment_on_bundled_python(tmp_path):
    """A virtual environment created on a downloaded package's Python runs that interpreter."""
    local_sim = tmp_path / "_isaac_sim"
    local_sim.mkdir()
    (local_sim / "python.sh").touch()
    bundled_python = local_sim / "kit" / "python" / "bin" / "python3"
    bundled_python.parent.mkdir(parents=True)
    bundled_python.touch()
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "python").symlink_to(bundled_python)
    (venv / "pyvenv.cfg").write_text(f"home = {bundled_python.parent}\n")

    with (
        mock.patch("isaaclab.cli.utils.DEFAULT_ISAAC_SIM_PATH", local_sim),
        mock.patch("isaaclab.cli.utils.extract_python_exe", return_value=str(venv / "bin" / "python")),
        mock.patch("isaaclab.cli.utils.run_command") as run,
        mock.patch.dict(os.environ, {"VIRTUAL_ENV": str(venv)}, clear=True),
    ):
        run_python_command("script.py", [])

    assert run.call_args is not None


@pytest.mark.parametrize("python_home", [None, "/usr/bin"])
def test_run_python_command_rejects_virtual_environment_on_foreign_python(tmp_path, python_home):
    """Reject unrelated interpreters with missing or foreign pyvenv.cfg metadata."""
    local_sim = tmp_path / "_isaac_sim"
    local_sim.mkdir()
    (local_sim / "python.sh").touch()
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    if python_home is not None:
        (venv / "pyvenv.cfg").write_text(f"home = {python_home}\n")

    with (
        mock.patch("isaaclab.cli.utils.DEFAULT_ISAAC_SIM_PATH", local_sim),
        mock.patch("isaaclab.cli.utils.extract_python_exe", return_value=str(venv / "bin" / "python")),
        mock.patch("isaaclab.cli.utils.run_command") as run,
        mock.patch.dict(os.environ, {"VIRTUAL_ENV": str(venv)}, clear=True),
        pytest.raises(SystemExit, match="1"),
    ):
        run_python_command("script.py", [])
    run.assert_not_called()


def test_run_python_command_does_not_wrap_an_active_isaac_sim_environment(tmp_path):
    """Do not source an active Isaac Sim environment twice."""
    local_sim = tmp_path / "_isaac_sim"
    local_sim.mkdir()
    (local_sim / "python.sh").touch()
    (local_sim / ".isaaclab_source_build").touch()
    active_python = str(tmp_path / ".venv" / "bin" / "python")

    with (
        mock.patch("isaaclab.cli.utils.DEFAULT_ISAAC_SIM_PATH", local_sim),
        mock.patch("isaaclab.cli.utils.extract_python_exe", return_value=active_python),
        mock.patch("isaaclab.cli.utils.run_command") as run,
        mock.patch.dict(os.environ, {"ISAAC_PATH": str(local_sim)}, clear=True),
    ):
        run_python_command("script.py", [])

    assert run.call_args.args[0] == [active_python, "script.py"]


class TestExtractIsaacsimPath:
    """Tests for :func:`extract_isaacsim_path`."""

    def test_returns_none_when_not_required(self):
        """When required=False and Isaac Sim is not found, return None."""
        with (
            mock.patch("isaaclab.cli.utils.DEFAULT_ISAAC_SIM_PATH", Path("/nonexistent/path")),
            mock.patch(
                "isaaclab.cli.utils.subprocess.run",
                return_value=subprocess.CompletedProcess(args=[], returncode=1),
            ),
        ):
            result = extract_isaacsim_path(required=False)
            assert result is None

    def test_exits_when_required(self):
        """When required=True and Isaac Sim is not found, sys.exit."""
        with (
            mock.patch("isaaclab.cli.utils.DEFAULT_ISAAC_SIM_PATH", Path("/nonexistent/path")),
            mock.patch(
                "isaaclab.cli.utils.subprocess.run",
                return_value=subprocess.CompletedProcess(args=[], returncode=1),
            ),
            pytest.raises(SystemExit),
        ):
            extract_isaacsim_path(required=True)

    def test_returns_path_when_symlink_exists(self, tmp_path):
        """When the default path exists, return it."""
        fake_sim = tmp_path / "_isaac_sim"
        fake_sim.mkdir()

        with mock.patch("isaaclab.cli.utils.DEFAULT_ISAAC_SIM_PATH", fake_sim):
            result = extract_isaacsim_path(required=True)
            assert result == fake_sim


@pytest.mark.parametrize("option", ["--install", "--conda", "--uv"])
def test_removed_installer_options_fail(option: str) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "isaaclab", option],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "unrecognized arguments" in result.stderr


@pytest.mark.parametrize(
    ("installation", "configured", "delegated"),
    [("source", False, True), ("source", True, False), ("container", False, True), ("stale", False, False)],
)
def test_runtime_dispatch_loads_local_kit_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, installation: str, configured: bool, delegated: bool
) -> None:
    import isaaclab.cli as cli

    local_sim = tmp_path / "_isaac_sim"
    local_sim.mkdir()
    (local_sim / ("python.bat" if os.name == "nt" else "python.sh")).touch()
    if installation == "source":
        (local_sim / ".isaaclab_source_build").touch()
    elif installation == "container":
        monkeypatch.setattr(sys, "executable", str(local_sim / "kit/python/bin/python3"))
    monkeypatch.setattr(cli, "DEFAULT_ISAAC_SIM_PATH", local_sim)
    monkeypatch.setattr(sys, "argv", ["isaaclab", "demo", "--help"])
    if configured:
        monkeypatch.setenv("ISAAC_PATH", str(local_sim))
    else:
        monkeypatch.delenv("ISAAC_PATH", raising=False)
    with mock.patch.object(cli, "run_python_command") as launch, mock.patch.object(cli, "demo") as demo:
        cli.cli()
    if not delegated:
        launch.assert_not_called()
        demo.assert_called_once_with(["--help"])
    else:
        launch.assert_called_once_with("-m", ["isaaclab", "demo", "--help"], check=True)
        demo.assert_not_called()
