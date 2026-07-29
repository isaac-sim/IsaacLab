# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared contact-sensor benchmark scene and workload."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from ..measurements import SingleMeasurement
from ..micro import LatencyBenchmarkRunner, LatencySample, measure_latency
from .timing import SensorLatencySamples, add_sensor_latency_measurements

if TYPE_CHECKING:
    from isaaclab.scene import InteractiveSceneCfg


def create_contact_sensor_scene_cfg(
    *, history_length: int, num_envs: int, env_spacing: float = 2.0
) -> InteractiveSceneCfg:
    """Create the common contact-sensor benchmark scene.

    Args:
        history_length: Number of contact history frames.
        num_envs: Number of cloned environments.
        env_spacing: Distance between adjacent environment origins [m].

    Returns:
        Scene configuration with one contact-sensed cube per environment.
    """
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObjectCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sensors import ContactSensorCfg
    from isaaclab.terrains import TerrainImporterCfg
    from isaaclab.utils.configclass import configclass

    @configclass
    class ContactSensorBenchmarkSceneCfg(InteractiveSceneCfg):
        """Scene with one cube and one contact sensor per environment."""

        terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane")
        cube = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cube",
            spawn=sim_utils.CuboidCfg(
                size=(0.5, 0.5, 0.5),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                activate_contact_sensors=True,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.2)),
        )
        contact_sensor = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Cube",
            track_air_time=True,
            update_period=0.0,
            history_length=0,
        )

    scene_cfg = ContactSensorBenchmarkSceneCfg(
        num_envs=num_envs,
        env_spacing=env_spacing,
        lazy_sensor_update=True,
    )
    scene_cfg.contact_sensor.history_length = history_length
    return scene_cfg


def run_contact_sensor_workload(
    *,
    benchmark_name: str,
    formatter_type: str,
    output_path: str,
    metadata: dict[str, str | int | float | dict],
    num_steps: int,
    warmup_steps: int,
    decimation: int,
    expected_contacts: int,
    step: Callable[[], None],
    update: Callable[[], None],
    read: Callable[[], None],
    count_contacts: Callable[[], int],
    synchronize: Callable[[], None],
) -> tuple[Path, ...]:
    """Run and report the common contact-sensor cadence workload.

    Args:
        benchmark_name: Name used in output metadata and filenames.
        formatter_type: Formatter used to report results.
        output_path: Directory for result files.
        metadata: Workload metadata stored with the result.
        num_steps: Number of measured sensor cadences.
        warmup_steps: Number of unmeasured sensor cadences.
        decimation: Physics and sensor updates per cadence.
        expected_contacts: Expected number of sensors reporting contact.
        step: One unmeasured physics step.
        update: One measured sensor update.
        read: One measured sensor data read.
        count_contacts: Return the number of sensors reporting contact.
        synchronize: Block until pending device work completes.

    Returns:
        Paths written by the benchmark formatters.

    Raises:
        RuntimeError: If the measured contact count differs from
            :paramref:`expected_contacts`.
    """
    for _ in range(warmup_steps):
        for _ in range(decimation):
            step()
            update()
        read()
    synchronize()

    cadence_samples: list[LatencySample] = []
    for _ in range(num_steps):
        synchronized_s = 0.0
        submission_s = 0.0
        for _ in range(decimation):
            step()
            sample = measure_latency(update, synchronize)
            synchronized_s += sample.synchronized_s
            submission_s += sample.submission_s
        sample = measure_latency(read, synchronize)
        cadence_samples.append(
            LatencySample(
                submission_s=submission_s + sample.submission_s,
                synchronized_s=synchronized_s + sample.synchronized_s,
            )
        )

    observer_s = [
        sum(measure_latency(lambda: None, synchronize).synchronized_s for _ in range(decimation + 1))
        for _ in range(num_steps)
    ]
    num_in_contact = count_contacts()
    if num_in_contact != expected_contacts:
        raise RuntimeError(f"Expected {expected_contacts} contacting sensors, received {num_in_contact}.")

    benchmark = LatencyBenchmarkRunner(
        benchmark_name=benchmark_name,
        formatter_type=formatter_type,
        output_path=output_path,
        metadata=metadata,
    )
    add_sensor_latency_measurements(
        benchmark,
        samples=SensorLatencySamples(cadence_samples, observer_s, []),
        validation=[
            SingleMeasurement(name="Sensors in Contact", value=num_in_contact, unit="count"),
            SingleMeasurement(name="Expected Sensors", value=expected_contacts, unit="count"),
        ],
        update_phase="sensor_cadence",
        observer_phase="observer",
        validation_phase="validation",
    )
    return benchmark.finalize()
