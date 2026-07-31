# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the discoverable micro-benchmark command."""

import sys
from pathlib import Path
from unittest import mock

import pytest

import isaaclab.cli as cli
from isaaclab.benchmark.microbenchmark import MicrobenchmarkFactory, run_microbenchmark_cli


@pytest.mark.parametrize(
    ("physics", "component", "relative_script", "prefix_args"),
    [
        (
            "physx",
            "articulation",
            "source/isaaclab_physx/benchmark/assets/benchmark_articulation.py",
            ["--physics_variant", "physx"],
        ),
        (
            "ovphysx",
            "ray_caster",
            "source/isaaclab_ovphysx/benchmark/sensors/benchmark_ray_caster.py",
            ["--physics_variant", "ovphysx"],
        ),
        (
            "newton_mjwarp",
            "imu",
            "source/isaaclab_newton/benchmark/sensors/benchmark_imu_pva.py",
            ["--physics_variant", "newton_mjwarp", "--sensor", "imu"],
        ),
        (
            "newton_kamino",
            "pva",
            "source/isaaclab_newton/benchmark/sensors/benchmark_imu_pva.py",
            ["--physics_variant", "newton_kamino", "--sensor", "pva"],
        ),
    ],
)
def test_factory_resolves_exact_physics_variant(
    physics: str,
    component: str,
    relative_script: str,
    prefix_args: list[str],
) -> None:
    """Each exact physics selector should resolve to a concrete retained command."""
    command = MicrobenchmarkFactory().build_command(physics, component, ["--num_steps", "3"])

    assert command.script == MicrobenchmarkFactory.repository_root() / relative_script
    assert command.args == [*prefix_args, "--num_steps", "3"]
    assert command.physics == physics
    assert command.component == component


def test_factory_rejects_unknown_variant_without_substitution() -> None:
    """Unknown variants should fail instead of silently selecting a related solver."""
    with pytest.raises(ValueError, match="newton_unknown"):
        MicrobenchmarkFactory().build_command("newton_unknown", "articulation", [])


def test_factory_rejects_unknown_component_with_available_choices() -> None:
    """An invalid component should report the discoverable component names."""
    with pytest.raises(ValueError, match="rigid_object"):
        MicrobenchmarkFactory().build_command("physx", "unknown", [])


@pytest.mark.parametrize("override", [["--physics_variant", "newton_mjwarp"], ["--physics_variant=newton_mjwarp"]])
def test_factory_rejects_passthrough_physics_variant_override(override: list[str]) -> None:
    """Passthrough arguments must not replace the exact selected physics variant."""
    with pytest.raises(ValueError, match="physics_variant"):
        MicrobenchmarkFactory().build_command("newton_kamino", "articulation", override)


def test_microbenchmark_cli_launches_selected_command() -> None:
    """The public CLI should launch the selected workload in a clean child process."""
    with mock.patch("isaaclab.benchmark.microbenchmark.run_python_command") as run_python:
        status = run_microbenchmark_cli(["--component", "joint_wrench", "physics=newton_kamino", "--num_steps", "2"])

    assert status == 0
    script, args = run_python.call_args.args
    assert isinstance(script, Path)
    assert script.name == "benchmark_joint_wrench.py"
    assert args == ["--physics_variant", "newton_kamino", "--num_steps", "2"]
    assert run_python.call_args.kwargs == {"check": True}


def test_top_level_cli_dispatches_microbenchmark() -> None:
    """``isaaclab microbenchmark`` should forward arguments to its dispatcher."""
    args = ["--component", "articulation", "physics=physx", "--num_iterations", "2"]

    with (
        mock.patch.object(sys, "argv", ["isaaclab", "microbenchmark", *args]),
        mock.patch("isaaclab.benchmark.run_microbenchmark_cli", return_value=0) as run_microbenchmark,
    ):
        cli.cli()

    run_microbenchmark.assert_called_once_with(args)
