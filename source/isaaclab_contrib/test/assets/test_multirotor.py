# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
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

import types
import warnings

import pytest
import torch
from isaaclab_physx.physics import PhysxCfg

import isaaclab.sim.utils.prims as prim_utils
from isaaclab.sim import SimulationCfg, build_simulation_context

from isaaclab_contrib.assets import Multirotor, MultirotorCfg

# Best-effort: suppress unraisable destructor warnings emitted during
# teardown of partially-constructed assets in CI/dev environments. We still
# perform explicit cleanup where possible, but filter the remaining noisy
# warnings to keep test output clean.
warnings.filterwarnings("ignore", category=pytest.PytestUnraisableExceptionWarning)

##
# Pre-defined configs
##
from isaaclab_assets.robots.arl_robot_1 import ARL_ROBOT_1_CFG

# -----------------------
# Unit tests (simulator-free)
# -----------------------


def make_multirotor_stub(num_instances: int, num_thrusters: int, device=torch.device("cpu")):
    """Create a lightweight Multirotor instance suitable for unit tests that
    don't require IsaacSim. We construct via __new__ and inject minimal
    attributes the class methods expect.
    """
    # Use a plain object (not a Multirotor instance) to avoid assigning to
    # properties that only exist on the real class. We'll bind the
    # Multirotor methods we need onto this fake object.
    m = types.SimpleNamespace()
    # runtime attributes the methods expect
    m.device = device
    m.num_instances = num_instances
    m.num_bodies = 1

    # allocation matrix as a plain Python list (the Multirotor property will
    # convert it to a tensor using `self.cfg.allocation_matrix`), so provide
    # it on `m.cfg` like the real object expects.
    alloc_list = [[1.0 if r < 2 and c == r else 0.0 for c in range(num_thrusters)] for r in range(6)]
    m.cfg = types.SimpleNamespace(allocation_matrix=alloc_list)
    # Also provide allocation_matrix directly on the fake object so bound methods
    # that access `self.allocation_matrix` succeed (properties won't dispatch
    # because `m` is not a real Multirotor instance).
    m.allocation_matrix = torch.tensor(alloc_list, device=device)

    # lightweight data container
    data = types.SimpleNamespace()
    data.default_thruster_rps = torch.zeros(num_instances, num_thrusters, device=device)
    data.thrust_target = torch.zeros(num_instances, num_thrusters, device=device)
    data.computed_thrust = torch.zeros(num_instances, num_thrusters, device=device)
    data.applied_thrust = torch.zeros(num_instances, num_thrusters, device=device)
    data.thruster_names = [f"thr_{i}" for i in range(num_thrusters)]
    m._data = data

    # combined-wrench buffers
    m._thrust_target_sim = torch.zeros_like(m._data.thrust_target)
    m._internal_wrench_target_sim = torch.zeros(num_instances, 6, device=device)
    m._internal_force_target_sim = torch.zeros(num_instances, m.num_bodies, 3, device=device)
    m._internal_torque_target_sim = torch.zeros(num_instances, m.num_bodies, 3, device=device)

    # bind class methods we want to test onto the fake object
    m._combine_thrusts = types.MethodType(Multirotor._combine_thrusts, m)
    m.set_thrust_target = types.MethodType(Multirotor.set_thrust_target, m)

    return m


def test_multirotor_combine_thrusts_unit():
    # Allocation matmul and the force/torque split select no branch on instance/thruster counts or device.
    num_instances, num_thrusters = 2, 4
    m = make_multirotor_stub(num_instances=num_instances, num_thrusters=num_thrusters)

    # Create thrust target with predictable values
    thrust_values = torch.arange(1.0, num_instances * num_thrusters + 1.0).reshape(num_instances, num_thrusters)
    m._thrust_target_sim = thrust_values

    # Signed entries so a sign-dropping allocation bug changes the wrench.
    alloc = [
        [float((r + 1) * (c + 1)) * (-1.0 if (r + c) % 2 else 1.0) for c in range(num_thrusters)] for r in range(6)
    ]
    m.cfg.allocation_matrix = alloc
    m.allocation_matrix = torch.tensor(alloc)

    m._combine_thrusts()

    # Expected wrench: thrust @ allocation.T
    alloc_t = torch.tensor(alloc)
    expected = torch.matmul(thrust_values, alloc_t.T)

    assert torch.allclose(m._internal_wrench_target_sim, expected)
    assert torch.allclose(m._internal_force_target_sim[:, 0, :], expected[:, :3])
    assert torch.allclose(m._internal_torque_target_sim[:, 0, :], expected[:, 3:])


def test_set_thrust_target_broadcasting_unit():
    num_instances, num_thrusters = 4, 4
    m = make_multirotor_stub(num_instances=num_instances, num_thrusters=num_thrusters)

    # Set full-row targets for env 0
    targets = torch.arange(1.0, num_thrusters + 1.0).unsqueeze(0)
    m.set_thrust_target(targets, thruster_ids=slice(None), env_ids=slice(0, 1))
    assert torch.allclose(m._data.thrust_target[0], targets[0])

    # Set a column across all envs (use integer thruster id so broadcasting works)
    # Use the last thruster to avoid index out of bounds
    thruster_id = num_thrusters - 1
    column_values = torch.full((num_instances,), 9.0)
    m.set_thrust_target(column_values, thruster_ids=thruster_id, env_ids=slice(None))
    assert torch.allclose(m._data.thrust_target[:, thruster_id], column_values)

    # Tensor env ids with a thruster id list broadcast to a (envs x thrusters) sub-block.
    before = m._data.thrust_target.clone()
    env_ids = torch.tensor([1, 3])
    thruster_ids = [0, 2]
    block = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    m.set_thrust_target(block, thruster_ids=thruster_ids, env_ids=env_ids)
    expected = before.clone()
    expected[1, 0], expected[1, 2], expected[3, 0], expected[3, 2] = 5.0, 6.0, 7.0, 8.0
    assert torch.equal(m._data.thrust_target, expected)


def test_set_thrust_target_env_slice_unit():
    """Setting targets for an env slice updates only those envs."""
    m = make_multirotor_stub(num_instances=4, num_thrusters=3)

    original = m._data.thrust_target.clone()
    targets = torch.tensor([[1.0, 2.0, 3.0]], device=m.device)
    # Update envs 1 and 2
    m.set_thrust_target(targets, thruster_ids=slice(None), env_ids=slice(1, 3))

    assert torch.allclose(m._data.thrust_target[1:3], targets.repeat(2, 1))
    # other envs remain unchanged
    assert torch.allclose(m._data.thrust_target[0], original[0])
    assert torch.allclose(m._data.thrust_target[3], original[3])


def generate_multirotor(
    multirotor_cfg: MultirotorCfg, num_multirotors: int, device: str
) -> tuple[Multirotor, torch.Tensor]:
    """Create scene prims and spawn `Multirotor` assets from a cfg.

    Mirrors the pattern used in `test_articulation.py`.
    """
    translations = torch.zeros(num_multirotors, 3, device=device)
    translations[:, 0] = torch.arange(num_multirotors) * 2.5

    for i in range(num_multirotors):
        prim_utils.create_prim(f"/World/Env_{i}", "Xform", translation=translations[i][:3])

    multirotor = Multirotor(multirotor_cfg.replace(prim_path="/World/Env_[^/]*/Robot"))
    return multirotor, translations


@pytest.fixture
def sim(request):
    """Create a PhysX simulation context without gravity, matching the ARL drone tasks' actuator setup."""
    device = request.getfixturevalue("device") if "device" in request.fixturenames else "cpu"
    # Thruster actuators run in Isaac Lab, not as Newton-native actuators (as in the ARL drone tasks).
    sim_cfg = SimulationCfg(dt=0.01, gravity=(0.0, 0.0, 0.0), physics=PhysxCfg(), use_newton_actuators=False)

    with build_simulation_context(device=device, sim_cfg=sim_cfg, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None
        yield sim


@pytest.mark.parametrize("num_multirotors", [1])
@pytest.mark.parametrize("device", ["cpu"])  # restrict to cpu for CI without GPUs
@pytest.mark.isaacsim_ci
def test_multirotor_thruster_buffers_and_actuators(sim, num_multirotors, device):
    """Thrust targets pass through the thruster actuators and lift the real multirotor."""
    cfg = ARL_ROBOT_1_CFG.replace(
        actuators={"thrusters": ARL_ROBOT_1_CFG.actuators["thrusters"].replace(dt=sim.cfg.dt)}
    )
    multirotor, _ = generate_multirotor(cfg, num_multirotors, device=sim.device)
    sim.reset()
    assert multirotor.is_initialized
    assert multirotor.data.thrust_target.ndim == 2

    num_thr = multirotor.num_thrusters
    thrust_range = ARL_ROBOT_1_CFG.actuators["thrusters"].thrust_range
    multirotor.set_thrust_target(torch.full((num_multirotors, num_thr), 5.0, device=sim.device))
    for _ in range(3):
        multirotor.write_data_to_sim()
        sim.step()
        multirotor.update(sim.cfg.dt)

    applied = multirotor.data.applied_thrust
    assert applied.shape == (num_multirotors, num_thr)
    assert torch.all(applied > 0.0)
    assert torch.all(applied >= thrust_range[0]) and torch.all(applied <= thrust_range[1])
    # With gravity off, the upward thrust must accelerate the body along +z.
    assert torch.all(multirotor.data.root_lin_vel_w.torch[:, 2] > 0.0)
