# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Integration tests comparing WrenchComposer output against raw PhysX ``apply_forces_and_torques_at_position``.

Two identical groups of cubes are placed in the same scene. One group receives its wrench through the composer
(``set_forces_and_torques_index`` -> ``write_data_to_sim`` -> compose -> PhysX apply in the body frame), the other
through the raw PhysX API with the matching ``is_global`` flag. After stepping, both groups must move identically.
"""

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import math
from dataclasses import dataclass

import pytest
import torch
import warp as wp

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sim import build_simulation_context
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

pytestmark = pytest.mark.integration

N_STEPS = 50
FORCE_MAGNITUDE = 10.0
TORQUE_MAGNITUDE = 1.0
ROT_45_Z = (0.0, 0.0, math.sin(math.pi / 8), math.cos(math.pi / 8))  # 45 degrees about Z in (x, y, z, w)
SPACING = 20.0


def spawn_cube_groups(num_cubes: int, device: str, height: float = 1.0) -> tuple[RigidObject, RigidObject]:
    """Spawn a composer group and a raw-PhysX group of ``num_cubes`` cubes, offset in Y so they never touch."""
    for i in range(num_cubes):
        sim_utils.create_prim(f"/World/Composer_{i}", "Xform", translation=(i * SPACING, 0.0, height))
        sim_utils.create_prim(f"/World/Raw_{i}", "Xform", translation=(i * SPACING, SPACING, height))
    spawn_cfg = sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
        rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
    )
    composer_cube = RigidObject(
        RigidObjectCfg(
            prim_path="/World/Composer_[^/]*/Object",
            spawn=spawn_cfg,
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, height), rot=ROT_45_Z),
        )
    )
    raw_cube = RigidObject(
        RigidObjectCfg(
            prim_path="/World/Raw_[^/]*/Object",
            spawn=spawn_cfg,
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, SPACING, height), rot=ROT_45_Z),
        )
    )
    return composer_cube, raw_cube


@dataclass
class Scenario:
    """One environment's wrench: ``offset`` is relative to the CoM (world axes if global, body axes otherwise)."""

    name: str
    force: tuple[float, float, float] = (0.0, 0.0, 0.0)
    torque: tuple[float, float, float] = (0.0, 0.0, 0.0)
    offset: tuple[float, float, float] | None = None
    is_global: bool = False


SCENARIOS = [
    Scenario("local_force", force=(FORCE_MAGNITUDE, 0.0, 0.0)),
    Scenario("global_force", force=(FORCE_MAGNITUDE, 0.0, 0.0), is_global=True),
    Scenario("local_force_at_position", force=(FORCE_MAGNITUDE, 0.0, 0.0), offset=(0.0, 0.5, 0.0)),
    Scenario("global_force_at_position", force=(FORCE_MAGNITUDE, 0.0, 0.0), offset=(0.0, 1.0, 0.0), is_global=True),
    Scenario("local_torque", torque=(0.0, 0.0, TORQUE_MAGNITUDE)),
    Scenario("global_torque", torque=(0.0, 0.0, TORQUE_MAGNITUDE), is_global=True),
    Scenario("global_force_lever_z", force=(0.0, 0.0, FORCE_MAGNITUDE), offset=(0.0, 1.0, 0.0), is_global=True),
]


class RawWrench:
    """Per-step raw PhysX application of a group of scenarios sharing ``is_global`` and position usage.

    The PhysX view API takes full ``(count, 3)`` arrays and applies only the rows named by ``indices``.
    """

    def __init__(self, cube: RigidObject, env_ids: list[int], scenarios: list[Scenario]):
        device = cube.device
        self.view = cube.root_view
        self.is_global = scenarios[0].is_global
        self.indices = wp.array(env_ids, dtype=wp.int32, device=device)
        forces = torch.zeros(cube.num_instances, 3, device=device)
        torques = torch.zeros(cube.num_instances, 3, device=device)
        forces[env_ids] = torch.tensor([s.force for s in scenarios], device=device)
        torques[env_ids] = torch.tensor([s.torque for s in scenarios], device=device)
        self.forces = wp.from_torch(forces, dtype=wp.float32)
        self.torques = wp.from_torch(torques, dtype=wp.float32)
        self.positions = None
        if scenarios[0].offset is not None:
            positions = torch.zeros(cube.num_instances, 3, device=device)
            positions[env_ids] = torch.tensor([s.offset for s in scenarios], device=device)
            if self.is_global:
                positions[env_ids] += cube.data.body_com_pos_w.torch[env_ids, 0, :3]
            self.positions = wp.from_torch(positions, dtype=wp.float32)

    def apply(self) -> None:
        self.view.apply_forces_and_torques_at_position(
            force_data=self.forces,
            torque_data=self.torques,
            position_data=self.positions,
            indices=self.indices,
            is_global=self.is_global,
        )


def set_wrenches(composer_cube: RigidObject, raw_cube: RigidObject, scenarios: list[Scenario]) -> list[RawWrench]:
    """Set each environment's scenario on the composer and build the matching raw PhysX appliers."""
    device = composer_cube.device
    groups: dict[tuple[bool, bool], list[int]] = {}
    for env_id, scenario in enumerate(scenarios):
        groups.setdefault((scenario.is_global, scenario.offset is not None), []).append(env_id)

    raw_wrenches = []
    for (is_global, has_offset), env_ids in groups.items():
        group = [scenarios[i] for i in env_ids]
        forces = torch.tensor([s.force for s in group], device=device).unsqueeze(1)
        torques = torch.tensor([s.torque for s in group], device=device).unsqueeze(1)
        positions = None
        if has_offset:
            positions = torch.tensor([s.offset for s in group], device=device).unsqueeze(1)
            if is_global:
                positions = positions + composer_cube.data.body_com_pos_w.torch[env_ids, :, :3]
        composer_cube.permanent_wrench_composer.set_forces_and_torques_index(
            forces=forces, torques=torques, positions=positions, body_ids=[0], env_ids=env_ids, is_global=is_global
        )
        raw_wrenches.append(RawWrench(raw_cube, env_ids, group))
    return raw_wrenches


def step(sim, composer_cube: RigidObject, raw_cube: RigidObject, raw_wrenches: list[RawWrench], num_steps: int):
    for _ in range(num_steps):
        composer_cube.write_data_to_sim()
        raw_cube.write_data_to_sim()  # no-op: the raw group's composer is inactive
        for raw_wrench in raw_wrenches:
            raw_wrench.apply()
        sim.step()
        composer_cube.update(sim.cfg.dt)
        raw_cube.update(sim.cfg.dt)


def assert_same_motion(composer_cube: RigidObject, raw_cube: RigidObject, tol: float = 1e-4, env_ids=slice(None)):
    torch.testing.assert_close(
        composer_cube.data.root_lin_vel_w.torch[env_ids],
        raw_cube.data.root_lin_vel_w.torch[env_ids],
        rtol=tol,
        atol=tol,
    )
    torch.testing.assert_close(
        composer_cube.data.root_ang_vel_w.torch[env_ids],
        raw_cube.data.root_ang_vel_w.torch[env_ids],
        rtol=tol,
        atol=tol,
    )


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_composer_matches_physx_for_all_wrench_types(device):
    """Every scenario runs in its own environment of one scene; the composer and raw groups must agree."""
    with build_simulation_context(device=device, gravity_enabled=False, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None
        composer_cube, raw_cube = spawn_cube_groups(len(SCENARIOS), device)
        sim.reset()
        composer_cube.update(sim.cfg.dt)
        raw_cube.update(sim.cfg.dt)

        raw_wrenches = set_wrenches(composer_cube, raw_cube, SCENARIOS)
        step(sim, composer_cube, raw_cube, raw_wrenches, N_STEPS)
        # the fast-spinning lever scenario accumulates more float error than the others
        assert_same_motion(composer_cube, raw_cube, env_ids=slice(0, 6))
        assert_same_motion(composer_cube, raw_cube, tol=1e-3, env_ids=slice(6, 7))

        # sanity: forces at the CoM and pure torques leave the other velocity untouched, offsets spin the body
        ang_vel = composer_cube.data.root_ang_vel_w.torch
        lin_vel = composer_cube.data.root_lin_vel_w.torch
        names = [s.name for s in SCENARIOS]
        assert ang_vel[[names.index("local_force"), names.index("global_force")]].abs().max().item() < 1e-4
        assert lin_vel[[names.index("local_torque"), names.index("global_torque")]].abs().max().item() < 1e-4
        for name in ("local_force_at_position", "global_force_at_position", "global_force_lever_z"):
            assert ang_vel[names.index(name)].abs().max().item() > 0.1

        # a longer run catches drift between the stored positional torque and the moving body
        step(sim, composer_cube, raw_cube, raw_wrenches, N_STEPS)
        assert_same_motion(composer_cube, raw_cube, tol=1e-3)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_composer_matches_physx_with_gravity_and_ground_contact(device):
    """Mirrors the payload MDP term: a permanent world-frame downward force while falling onto the ground."""
    with build_simulation_context(device=device, gravity_enabled=True, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None
        composer_cube, raw_cube = spawn_cube_groups(1, device, height=0.5)
        sim.reset()
        composer_cube.update(sim.cfg.dt)
        raw_cube.update(sim.cfg.dt)
        initial_pos = composer_cube.data.root_pos_w.torch.clone(), raw_cube.data.root_pos_w.torch.clone()

        payload = Scenario("payload", force=(0.0, 0.0, -2.0 * 9.81), is_global=True)
        raw_wrenches = set_wrenches(composer_cube, raw_cube, [payload])
        step(sim, composer_cube, raw_cube, raw_wrenches, N_STEPS)

        torch.testing.assert_close(
            composer_cube.data.root_pos_w.torch - initial_pos[0],
            raw_cube.data.root_pos_w.torch - initial_pos[1],
            rtol=1e-4,
            atol=1e-4,
        )
        assert_same_motion(composer_cube, raw_cube)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_composer_matches_physx_after_partial_reset(device):
    """Resetting half of the environments clears their permanent wrench; re-setting it must match raw PhysX."""
    num_cubes = 4
    with build_simulation_context(device=device, gravity_enabled=False, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None
        composer_cube, raw_cube = spawn_cube_groups(num_cubes, device)
        sim.reset()
        composer_cube.update(sim.cfg.dt)
        raw_cube.update(sim.cfg.dt)
        # world-frame initial states (they include the per-environment origins)
        initial_states = [
            torch.cat(
                [cube.data.root_link_pos_w.torch, cube.data.root_link_quat_w.torch, cube.data.root_com_vel_w.torch],
                dim=-1,
            ).clone()
            for cube in (composer_cube, raw_cube)
        ]

        scenarios = [Scenario("global_force", force=(FORCE_MAGNITUDE, 0.0, 0.0), is_global=True)] * num_cubes
        raw_wrenches = set_wrenches(composer_cube, raw_cube, scenarios)
        step(sim, composer_cube, raw_cube, raw_wrenches, N_STEPS // 2)

        reset_ids = torch.arange(num_cubes // 2, device=device)
        for cube, state in zip((composer_cube, raw_cube), initial_states):
            cube.write_root_link_pose_to_sim_index(root_pose=state[reset_ids, :7], env_ids=reset_ids)
            cube.write_root_com_velocity_to_sim_index(root_velocity=state[reset_ids, 7:], env_ids=reset_ids)
            cube.reset(reset_ids.tolist())
        # the reset cleared the permanent wrench of those environments
        set_wrenches(composer_cube, raw_cube, scenarios)
        step(sim, composer_cube, raw_cube, raw_wrenches, N_STEPS // 2)

        assert_same_motion(composer_cube, raw_cube)
        assert composer_cube.data.root_ang_vel_w.torch.abs().max().item() < 1e-4
