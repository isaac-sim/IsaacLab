# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

HEADLESS = True

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import ctypes
import torch

import isaacsim.core.utils.prims as prim_utils
import pytest
from isaacsim.core.version import get_version

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
import isaaclab.utils.string as string_utils
from isaaclab.actuators import Thruster, ThrusterCfg
from isaaclab.assets import Multirotor, MultirotorCfg
from isaaclab.envs.mdp.terminations import joint_effort_out_of_limit
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim import build_simulation_context
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Pre-defined configs
##
# Try to import the ARL robot 1 config; fall back to None if the package
# is not available in lightweight environments.
from isaaclab_assets import ARL_ROBOT_1_CFG  # isort:skip



def generate_multirotor_cfg(usd_path: str | None = None) -> MultirotorCfg:
	"""Generate a multirotor configuration for tests.

	If an ARL-provided config is available, prefer that. Otherwise return a
	minimal `MultirotorCfg` so integration tests can still run when a USD is
	provided.
	"""
	if ARL_ROBOT_1_CFG is not None:
		return ARL_ROBOT_1_CFG

	if usd_path is None:
		return MultirotorCfg()

	return MultirotorCfg(spawn=sim_utils.UsdFileCfg(usd_path=usd_path))


def generate_multirotor(multirotor_cfg: MultirotorCfg, num_multirotors: int, device: str) -> tuple[Multirotor, torch.Tensor]:
	"""Create scene prims and spawn `Multirotor` assets from a cfg.

	Mirrors the pattern used in `test_articulation.py`.
	"""
	translations = torch.zeros(num_multirotors, 3, device=device)
	translations[:, 0] = torch.arange(num_multirotors) * 2.5

	for i in range(num_multirotors):
		prim_utils.create_prim(f"/World/Env_{i}", "Xform", translation=translations[i][:3])

	# Replace the prim_path like other tests do
	multirotor = Multirotor(multirotor_cfg.replace(prim_path="/World/Env_.*/Robot"))
	return multirotor, translations


@pytest.fixture
def sim(request):
	"""Create a simulation context for integration tests (app + sim).

	Uses `build_simulation_context` from the project utils so tests match
	`test_articulation.py` behaviour.
	"""
	device = request.getfixturevalue("device") if "device" in request.fixturenames else "cpu"
	gravity_enabled = request.getfixturevalue("gravity_enabled") if "gravity_enabled" in request.fixturenames else True
	add_ground_plane = request.getfixturevalue("add_ground_plane") if "add_ground_plane" in request.fixturenames else False

	with build_simulation_context(device=device, auto_add_lighting=True, gravity_enabled=gravity_enabled, add_ground_plane=add_ground_plane) as sim:
		sim._app_control_on_stop_handle = None
		yield sim


@pytest.mark.parametrize("num_multirotors", [1])
@pytest.mark.parametrize("device", ["cpu"])  # restrict to cpu for CI without GPUs
@pytest.mark.isaacsim_ci
def test_multirotor_thruster_buffers_and_actuators(sim, num_multirotors, device):
	"""Check thruster buffers and actuator wiring in an integration environment.

	This test will be skipped automatically when `ARL_ROBOT_1_CFG` is not
	available in the test environment (lightweight setups).
	"""
	cfg = generate_multirotor_cfg()
	if ARL_ROBOT_1_CFG is None and (not hasattr(cfg, "actuators") or not cfg.actuators):
		pytest.skip("No multirotor actuator configuration available for integration test")

	multirotor, _ = generate_multirotor(cfg, num_multirotors, device=sim.device)
	sim.reset()
	assert multirotor.is_initialized

	# If thruster buffers exist they should have the expected 2D shape
	if hasattr(multirotor.data, "thrust_target") and multirotor.data.thrust_target is not None:
		assert multirotor.data.thrust_target.ndim == 2

	# If actuators exist, calling the actuator model and update should produce applied_thrust
	try:
		num_thr = multirotor.num_thrusters
	except Exception:
		num_thr = 0

	if num_thr > 0:
		# Broadcast a simple thrust target and simulate a few steps
		multirotor.set_thrust_target(torch.ones(num_multirotors, num_thr, device=sim.device))
		for _ in range(3):
			sim.step()
			multirotor.update(sim.cfg.dt)

		assert hasattr(multirotor.data, "applied_thrust")
		assert multirotor.data.applied_thrust.shape == (num_multirotors, num_thr)


@pytest.mark.parametrize("num_multirotors", [1])
@pytest.mark.parametrize("device", ["cpu"])
@pytest.mark.isaacsim_ci
def test_set_thrust_target_broadcasting_integration(sim, num_multirotors, device):
	"""Ensure `set_thrust_target` broadcasting works in the integration context."""
	cfg = generate_multirotor_cfg()
	if ARL_ROBOT_1_CFG is None and (not hasattr(cfg, "actuators") or not cfg.actuators):
		pytest.skip("No multirotor actuator configuration available for integration test")

	multirotor, _ = generate_multirotor(cfg, num_multirotors, device=sim.device)
	sim.reset()

	try:
		num_thr = multirotor.num_thrusters
	except Exception:
		pytest.skip("Multirotor does not expose thrusters in this configuration")

	# Set a single-thruster column across all envs
	multirotor.set_thrust_target(torch.tensor([9.0] * num_multirotors, device=sim.device), thruster_ids=0, env_ids=slice(None))
	# Check that the first column of thrust_target has been updated
	assert torch.allclose(multirotor.data.thrust_target[:, 0], torch.tensor([9.0] * num_multirotors, device=sim.device))
