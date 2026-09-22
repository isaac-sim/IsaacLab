# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the lazy-update, reset, and caching contract of :class:`SensorBase`."""

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True).app

from collections.abc import Sequence

import pytest
import torch
import warp as wp

from pxr import UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.sensors import SensorBase, SensorBaseCfg
from isaaclab.utils import configclass

pytestmark = pytest.mark.integration

NUM_ENVS = 5
DT = 0.01


@wp.kernel
def increment_count_kernel(env_mask: wp.array(dtype=wp.bool), count: wp.array(dtype=wp.int32)):
    env_id = wp.tid()
    if env_mask[env_id]:
        count[env_id] += 1


class DummySensor(SensorBase):
    """Counts how often each environment's buffers were refreshed by the backend."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.count: torch.Tensor | None = None
        self.backend_update_count = 0

    def _initialize_impl(self):
        super()._initialize_impl()
        self.count = torch.zeros(self._num_envs, dtype=torch.int32, device=self.device)

    @property
    def data(self) -> torch.Tensor:
        self._update_outdated_buffers()
        return self.count

    def _update_buffers_impl(self, env_mask: wp.array):
        self.backend_update_count += 1
        wp.launch(
            increment_count_kernel, dim=self._num_envs, inputs=[env_mask, wp.from_torch(self.count)], device=self.device
        )

    def reset(self, env_ids: Sequence[int] | None = None, env_mask: wp.array | None = None):
        super().reset(env_ids=env_ids, env_mask=env_mask)
        if env_ids is None and env_mask is not None:
            env_ids = wp.to_torch(env_mask).nonzero(as_tuple=False).squeeze(-1)
        self.count[slice(None) if env_ids is None else env_ids] = 0


@configclass
class DummySensorCfg(SensorBaseCfg):
    class_type = DummySensor

    prim_path = "{ENV_REGEX_NS}/Cube/dummy_sensor"


@pytest.fixture
def sim(device):
    sim_utils.create_new_stage()
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device=device, dt=DT))
    for i in range(NUM_ENVS):
        sim_utils.create_prim(f"/World/envs/env_{i:02d}/Cube", "Cube", translation=(i * 1.0, 0.0, 0.0))
    sim_utils.update_stage()
    yield sim
    sim.stop()
    sim.clear_instance()


def _expect_count(sensor: DummySensor, value: int, env_ids=slice(None)):
    torch.testing.assert_close(sensor.data[env_ids], torch.full_like(sensor.data[env_ids], value))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_sensor_update_and_data_cache(sim, device):
    """Buffers refresh once per update only when read (or forced), and repeated reads reuse the cached data."""
    sensor = DummySensor(DummySensorCfg())
    sim.reset()
    assert sensor.is_initialized
    assert sensor.num_instances == NUM_ENVS

    for step in range(3):
        sim.step()
        sensor.update(dt=DT, force_recompute=True)
        _expect_count(sensor, step + 1)

    # lazy updates only refresh the buffers when the data is accessed
    for _ in range(2):
        sim.step()
        sensor.update(dt=DT)
    assert sensor.count.max() == 3
    _expect_count(sensor, 4)

    # repeated reads within one generation do not refresh again; a forced refresh bypasses the cache
    backend_updates = sensor.backend_update_count
    _ = sensor.data
    assert sensor.backend_update_count == backend_updates
    sensor._update_outdated_buffers(force_recompute=True)
    assert sensor.backend_update_count == backend_updates + 1


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_sensor_update_period(sim, device):
    """With an update period of two steps the buffers only refresh every other update."""
    sensor = DummySensor(DummySensorCfg(update_period=2 * DT))
    sim.reset()

    expected = 1
    for step in range(6):
        sim.step()
        sensor.update(dt=DT, force_recompute=True)
        _expect_count(sensor, expected)
        expected += step % 2


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_sensor_reset(sim, device):
    """Full and partial resets clear the selected environments and invalidate the cached data exactly once."""
    sensor = DummySensor(DummySensorCfg())
    sim.reset()
    for step in range(3):
        sim.step()
        sensor.update(dt=DT)
        _expect_count(sensor, step + 1)

    sensor.reset()
    backend_updates = sensor.backend_update_count
    _expect_count(sensor, 1)
    _ = sensor.data
    assert sensor.backend_update_count == backend_updates + 1

    reset_ids, continued_ids = [2, 4], [0, 1, 3]
    sensor.reset(env_ids=reset_ids)
    _expect_count(sensor, 1, reset_ids)
    _expect_count(sensor, 1, continued_ids)
    for step in range(2):
        sim.step()
        sensor.update(dt=DT)
        _expect_count(sensor, step + 2, reset_ids)
        _expect_count(sensor, step + 2, continued_ids)

    sensor.reset(env_mask=wp.array([True, False, False, False, False], dtype=wp.bool, device=sensor.device))
    _expect_count(sensor, 1, [0])
    _expect_count(sensor, 3, [1, 2, 3, 4])


@pytest.mark.parametrize("device", ["cuda"])
def test_repeated_data_reads_are_graph_safe(sim, device):
    """CUDA graph capture records exactly one backend refresh for repeated reads."""
    sensor = DummySensor(DummySensorCfg())
    sim.reset()
    # warm up the kernels before capture
    sensor.update(dt=DT)
    _ = sensor.data
    backend_updates = sensor.backend_update_count

    with wp.ScopedCapture(device=device) as capture:
        sensor.update(dt=DT)
        _ = sensor.data
        _ = sensor.data

    assert sensor.backend_update_count == backend_updates + 1
    wp.capture_launch(capture.graph)


@pytest.mark.parametrize("device", ["cpu"])
def test_rigid_body_ancestor_expr_trims_only_terminal_suffix(sim, device):
    """Ancestor expression trimming keeps repeated path segments above the sensor."""
    parent_path = "/World/envs/env_00/Robot/link"
    sim_utils.create_prim(parent_path, "Xform")
    sim_utils.create_prim(parent_path + "/link", "Xform")
    UsdPhysics.RigidBodyAPI.Apply(sim_utils.get_current_stage().GetPrimAtPath(parent_path))
    sim_utils.update_stage()

    sensor = DummySensor(DummySensorCfg(prim_path="{ENV_REGEX_NS}/Robot/link/link"))
    rigid_parent_expr, fixed_pos_b, fixed_quat_b = sensor._resolve_rigid_body_ancestor_expr()

    assert rigid_parent_expr == "/World/envs/env_[^/]+/Robot/link"
    assert fixed_pos_b is not None and fixed_quat_b is not None
