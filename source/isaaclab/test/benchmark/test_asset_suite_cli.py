# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for retained asset micro-benchmark script dispatch."""

from pathlib import Path
from types import SimpleNamespace

import isaaclab.benchmark.asset_suites.cli as cli


def test_script_cli_builds_one_combined_request(monkeypatch, tmp_path) -> None:
    """A retained script should dispatch one method-and-data request."""
    adapter = SimpleNamespace(default_num_bodies=4, default_num_joints=0)
    captured = {}
    monkeypatch.setattr(cli, "get_asset_benchmark_adapter", lambda physics, component: adapter)
    monkeypatch.setattr(
        cli, "run_asset_benchmark", lambda request, selected: captured.update(request=request, adapter=selected) or ()
    )

    result = cli.run_asset_benchmark_cli(
        "physx",
        "rigid_object_collection",
        [
            "--num_iterations",
            "3",
            "--warmup_steps",
            "1",
            "--num_instances",
            "8",
            "--output_dir",
            str(tmp_path),
            "--backend",
            "json",
            "--device",
            "cpu",
        ],
        include_app_launcher_args=False,
    )

    assert result == ()
    assert captured["adapter"] is adapter
    assert captured["request"].physics_variant == "physx"
    assert captured["request"].config.num_bodies == 4
    assert captured["request"].config.num_joints == 0
    assert captured["request"].output_path == Path(tmp_path)
    assert captured["request"].launcher_args is None


def test_script_cli_uses_explicit_exact_physics_variant(monkeypatch, tmp_path) -> None:
    """The factory-selected exact variant should override the wrapper default."""
    adapter = SimpleNamespace(default_num_bodies=1, default_num_joints=2)
    captured = {}
    adapter_calls = []
    monkeypatch.setattr(
        cli,
        "get_asset_benchmark_adapter",
        lambda physics, component: adapter_calls.append((physics, component)) or adapter,
    )
    monkeypatch.setattr(cli, "run_asset_benchmark", lambda request, selected: captured.update(request=request) or ())

    cli.run_asset_benchmark_cli(
        "newton_mjwarp",
        "articulation",
        ["--physics_variant", "newton_kamino", "--output_dir", str(tmp_path), "--device", "cpu"],
        include_app_launcher_args=False,
    )

    assert adapter_calls == [("newton_kamino", "articulation")]
    assert captured["request"].physics_variant == "newton_kamino"


def test_script_cli_accepts_app_launcher_device_argument(monkeypatch, tmp_path) -> None:
    """Simulator-backed scripts should share AppLauncher device parsing without conflicts."""
    adapter = SimpleNamespace(default_num_bodies=1, default_num_joints=0)
    captured = {}
    monkeypatch.setattr(cli, "get_asset_benchmark_adapter", lambda physics, component: adapter)
    monkeypatch.setattr(cli, "run_asset_benchmark", lambda request, selected: captured.update(request=request) or ())

    cli.run_asset_benchmark_cli(
        "physx",
        "rigid_object",
        ["--device", "cpu", "--output_dir", str(tmp_path)],
    )

    assert captured["request"].config.device == "cpu"
    assert captured["request"].launcher_args.device == "cpu"
