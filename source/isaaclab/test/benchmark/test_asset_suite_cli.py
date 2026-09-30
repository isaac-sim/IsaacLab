# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for asset micro-benchmark script dispatch."""

from pathlib import Path
from types import SimpleNamespace

import pytest

import isaaclab.benchmark.asset_suites.cli as cli


@pytest.mark.parametrize(
    ("script_variant", "component", "variant_argv", "expected_variant"),
    [
        ("physx", "rigid_object_collection", [], "physx"),
        # The factory-selected exact variant should override the wrapper default.
        ("newton_mjwarp", "articulation", ["--physics_variant", "newton_kamino"], "newton_kamino"),
    ],
)
def test_script_cli_builds_one_combined_request(
    monkeypatch, tmp_path, script_variant: str, component: str, variant_argv: list[str], expected_variant: str
) -> None:
    """An asset script should dispatch one method-and-data request."""
    adapter = SimpleNamespace(
        default_num_bodies=4,
        default_num_joints=0,
        capabilities=frozenset(),
        generator_overrides={},
    )
    captured = {}
    adapter_calls = []
    monkeypatch.setattr(
        cli,
        "get_asset_benchmark_adapter",
        lambda physics, component: adapter_calls.append((physics, component)) or adapter,
    )
    monkeypatch.setattr(
        cli, "run_asset_benchmark", lambda request, selected: captured.update(request=request, adapter=selected) or ()
    )

    result = cli.run_asset_benchmark_cli(
        script_variant,
        component,
        [
            *variant_argv,
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
    assert adapter_calls == [(expected_variant, component)]
    assert captured["adapter"] is adapter
    assert captured["request"].physics_variant == expected_variant
    assert captured["request"].config.num_bodies == 4
    assert captured["request"].config.num_joints == 0
    assert captured["request"].output_path == Path(tmp_path)
    assert captured["request"].launcher_args is None


@pytest.mark.parametrize(
    ("option", "value", "message"),
    [
        ("--num_iterations", "0", "must be greater than zero"),
        ("--warmup_steps", "-1", "must be non-negative"),
        ("--num_instances", "0", "must be greater than zero"),
        ("--num_bodies", "0", "must be greater than zero"),
        ("--num_joints", "-1", "must be non-negative"),
        ("--mode", "unknown", "invalid choice"),
    ],
)
def test_script_cli_reports_invalid_arguments(monkeypatch, capsys, option: str, value: str, message: str) -> None:
    """Invalid benchmark arguments should exit through argparse without a traceback."""
    adapter = SimpleNamespace(
        default_num_bodies=1,
        default_num_joints=0,
        capabilities=frozenset(),
        generator_overrides={},
    )
    monkeypatch.setattr(cli, "get_asset_benchmark_adapter", lambda physics, component: adapter)

    with pytest.raises(SystemExit) as exc_info:
        cli.run_asset_benchmark_cli(
            "physx",
            "rigid_object",
            [option, value, "--device", "cpu"],
            include_app_launcher_args=False,
        )

    assert exc_info.value.code == 2
    assert message in capsys.readouterr().err
