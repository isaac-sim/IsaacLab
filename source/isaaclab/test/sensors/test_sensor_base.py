# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app


"""Rest everything follows."""

import torch
from collections.abc import Sequence
from dataclasses import dataclass

import isaacsim.core.utils.prims as prim_utils
import isaacsim.core.utils.stage as stage_utils
import pytest

import isaaclab.sim as sim_utils
from isaaclab.sensors import SensorBase, SensorBaseCfg
from isaaclab.utils import configclass


@dataclass
class DummyData:
    count: torch.Tensor = None


class DummySensor(SensorBase):

    def __init__(self, cfg):
        super().__init__(cfg)
        self._data = DummyData()

    def _initialize_impl(self):
        super()._initialize_impl()
        self._data.count = torch.zeros((self._num_envs), dtype=torch.int, device=self.device)

    @property
    def data(self):
        # update sensors if needed
        self._update_outdated_buffers()
        # return the data (where `_data` is the data for the sensor)
        return self._data

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        self._data.count[env_ids] += 1

    # def _set_debug_vis_impl(self, debug_vis: bool):

    # def _debug_vis_callback(self, event):


@configclass
class DummySensorCfg(SensorBaseCfg):
    class_type = DummySensor

    prim_path = "/World/envs/env_.*/Cube/dummy_sensor"


def _populate_scene():
    """"""

    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.SphereLightCfg()
    cfg.func("/World/Light/GreySphere", cfg, translation=(4.5, 3.5, 10.0))
    cfg.func("/World/Light/WhiteSphere", cfg, translation=(-4.5, 3.5, 10.0))

    # create prims
    for i in range(5):
        _ = prim_utils.create_prim(
            f"/World/envs/env_{i:02d}/Cube",
            "Cube",
            translation=(i * 1.0, 0.0, 0.0),
            scale=(0.25, 0.25, 0.25),
        )


@pytest.fixture
def create_dummy_sensor(request, device):

    # Create a new stage
    stage_utils.create_new_stage()

    # Simulation time-step
    dt = 0.05
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=dt, device=device)
    sim = sim_utils.SimulationContext(sim_cfg)

    # create sensor
    _populate_scene()

    sensor_cfg = DummySensorCfg()

    stage_utils.update_stage()

    yield sensor_cfg, sim, dt

    # stop simulation
    # note: cannot use self.sim.stop() since it does one render step after stopping!! This doesn't make sense :(
    sim._timeline.stop()
    # clear the stage
    sim.clear_all_callbacks()
    sim.clear_instance()


@pytest.mark.parametrize("device", ("cpu", "cuda"))
def test_sensor_init(create_dummy_sensor, device):

    sensor_cfg, sim, dt = create_dummy_sensor
    sensor = DummySensor(cfg=sensor_cfg)

    # Play sim
    sim.step()

    sim.reset()

    assert sensor.is_initialized
    assert int(sensor.num_instances) == 5

    for _ in range(10):
        sim.step()
        sensor.update(dt=dt, force_recompute=True)

    assert sensor.data.count.shape[0] == 5
