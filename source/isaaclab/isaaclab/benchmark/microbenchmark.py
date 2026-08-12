# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Discoverable command factory for component micro-benchmarks."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from isaaclab.cli.utils import run_python_command


@dataclass(frozen=True)
class MicrobenchmarkCommand:
    """Resolved micro-benchmark child command.

    Attributes:
        physics: Exact physics variant selected for the workload.
        component: Component workload to benchmark.
        script: Benchmark entrypoint to execute.
        args: Arguments passed to the benchmark entrypoint.
    """

    physics: str
    component: str
    script: Path
    args: list[str]


@dataclass(frozen=True)
class _PhysicsDescriptor:
    """Lightweight workload descriptor for one exact physics variant."""

    package: str
    variant: str


class MicrobenchmarkFactory:
    """Resolve exact physics variants and components to benchmark entrypoints."""

    _PHYSICS = MappingProxyType(
        {
            "physx": _PhysicsDescriptor("isaaclab_physx", "physx"),
            "ovphysx": _PhysicsDescriptor("isaaclab_ovphysx", "ovphysx"),
            "newton_mjwarp": _PhysicsDescriptor("isaaclab_newton", "newton_mjwarp"),
            "newton_kamino": _PhysicsDescriptor("isaaclab_newton", "newton_kamino"),
        }
    )
    _ASSET_COMPONENTS = frozenset({"articulation", "rigid_object", "rigid_object_collection"})
    _SENSOR_COMPONENTS = frozenset({"contact_sensor", "frame_transformer", "imu", "pva", "joint_wrench", "ray_caster"})

    @classmethod
    def repository_root(cls) -> Path:
        """Return the Isaac Lab repository root.

        Returns:
            Repository root containing the backend benchmark entrypoints.
        """
        return Path(__file__).parents[4]

    @classmethod
    def physics_variants(cls) -> tuple[str, ...]:
        """Return discoverable exact physics variants.

        Returns:
            Exact physics selectors accepted by :meth:`build_command`.
        """
        return tuple(cls._PHYSICS)

    @classmethod
    def components(cls) -> tuple[str, ...]:
        """Return discoverable component workloads.

        Returns:
            Sorted component names accepted by :meth:`build_command`.
        """
        return tuple(sorted(cls._ASSET_COMPONENTS | cls._SENSOR_COMPONENTS))

    def build_command(self, physics: str, component: str, passthrough_args: list[str]) -> MicrobenchmarkCommand:
        """Resolve one exact physics/component selection.

        Args:
            physics: Exact physics variant.
            component: Asset or sensor component workload.
            passthrough_args: Additional arguments for the selected entrypoint.

        Returns:
            Child command for the selected workload.

        Raises:
            ValueError: If the physics variant or component is unknown, or if
                :paramref:`passthrough_args` overrides the selected variant.
        """
        if any(arg == "--physics_variant" or arg.startswith("--physics_variant=") for arg in passthrough_args):
            raise ValueError("--physics_variant is reserved; select the variant with physics=<variant>.")

        try:
            descriptor = self._PHYSICS[physics]
        except KeyError as exc:
            available = ", ".join(self.physics_variants())
            raise ValueError(f"Unknown physics variant '{physics}'. Available variants: {available}.") from exc

        if component not in self._ASSET_COMPONENTS | self._SENSOR_COMPONENTS:
            available = ", ".join(self.components())
            raise ValueError(f"Unknown component '{component}'. Available components: {available}.")

        group = "assets" if component in self._ASSET_COMPONENTS else "sensors"
        script_component = "imu_pva" if component in {"imu", "pva"} else component
        script = (
            self.repository_root()
            / "source"
            / descriptor.package
            / "benchmark"
            / group
            / f"benchmark_{script_component}.py"
        )
        child_args = ["--physics_variant", descriptor.variant]
        if component in {"imu", "pva"}:
            child_args.extend(["--sensor", component])
        child_args.extend(passthrough_args)
        return MicrobenchmarkCommand(descriptor.variant, component, script, child_args)


def _create_parser(factory: MicrobenchmarkFactory, *, add_help: bool = True) -> argparse.ArgumentParser:
    """Create the top-level micro-benchmark parser."""
    variants = ", ".join(factory.physics_variants())
    components = ", ".join(factory.components())
    return argparse.ArgumentParser(
        prog="isaaclab microbenchmark",
        description="Run one component micro-benchmark with an exact physics variant.",
        epilog=f"physics variants: {variants}\ncomponents: {components}",
        formatter_class=argparse.RawTextHelpFormatter,
        add_help=add_help,
    )


def run_microbenchmark_cli(args: list[str] | None = None) -> int:
    """Parse and run ``isaaclab microbenchmark``.

    Args:
        args: Command arguments. Uses :data:`sys.argv` when ``None``.

    Returns:
        Zero after the child benchmark completes successfully.

    Raises:
        SystemExit: If command arguments are invalid.
        subprocess.CalledProcessError: If the child benchmark fails.
    """
    factory = MicrobenchmarkFactory()
    raw_args = list(args) if args is not None else sys.argv[1:]
    forward_help = any(arg in ("-h", "--help") for arg in raw_args)
    if forward_help:
        has_component = any(arg == "--component" or arg.startswith("--component=") for arg in raw_args)
        has_physics = any(arg.startswith("physics=") for arg in raw_args)
        forward_help = has_component and has_physics
    parser = _create_parser(factory, add_help=not forward_help)
    parser.add_argument("--component", required=True, choices=factory.components())
    parser.add_argument("physics", help="Exact selector in the form physics=<variant>.")
    parsed, passthrough = parser.parse_known_args(raw_args)
    physics = parsed.physics.removeprefix("physics=")
    if physics == parsed.physics or not physics:
        parser.error("physics must use the form physics=<variant>")
    try:
        command = factory.build_command(physics, parsed.component, passthrough)
    except ValueError as exc:
        parser.error(str(exc))
    run_python_command(command.script, command.args, check=True)
    return 0
