# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

"""Integration tests for :class:`NewtonKaminoManager` reset / forward-kinematics correctness.

These tests exercise the path used by RL reset events:

1. ``articulation.write_*_to_sim_index(..., env_ids=subset)`` (plus ``write_root_*`` for
   floating bases) writes new joint / root state into the simulation and dirties only the
   selected environments via :meth:`NewtonManager.invalidate_fk`.
2. Reading a body-pose property (``body_link_pose_w``) lazily triggers
   :meth:`ArticulationData._ensure_fk_fresh` -> :meth:`NewtonManager.forward` -> the Kamino
   masked ``solver.reset()`` reconcile.

The reconcile must (a) reconcile only the dirtied worlds, leaving others untouched until the
next step, (b) clear the reset masks afterwards, and (c) resolve the closed four-bar loop and
floating-base anchoring from the written coordinates.

The assets are spawned from USD (``FOURBAR_POLE_CFG`` and its floating sibling) following the
manual multi-env spawn pattern used in ``test_newton_actuators_newton.py``.
"""

from isaaclab.app import AppLauncher

# Kamino reset/FK reconcile needs USD spawning, which requires the Kit app.
simulation_app = AppLauncher(headless=True).app

import math

import pytest
import torch
import warp as wp
from isaaclab_newton.assets import Articulation, RigidObject
from isaaclab_newton.physics import KaminoSolverCfg, NewtonCfg
from isaaclab_newton.physics import NewtonManager as SimulationManager

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.sim import SimulationCfg, build_simulation_context
from isaaclab.utils.math import quat_apply

from isaaclab_assets.robots.fourbar_pole import FOURBAR_POLE_CFG, FOURBAR_POLE_FLOATING_CFG

# Kamino runs on CUDA; skip the whole module on CPU-only machines.
pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="Kamino requires a CUDA device"),
    pytest.mark.parametrize("use_cuda_graph", [False, True], ids=["no_cuda_graph", "cuda_graph"]),
]

DEVICE = "cuda:0"
NUM_ENVS = 4
DIRTY_ENVS = [0, 1, 2]
PROTECTED_ENV = 3
ENV_SPACING = 4.0

# Four-bar joints: ``ground_to_crank`` and ``coupler_to_pole`` are the two actuated (independent)
# DoFs; ``crank_to_coupler`` / ``coupler_to_rocker`` are passive and resolved by the loop closure.
ACTUATED_JOINTS = ["ground_to_crank", "coupler_to_pole"]
PASSIVE_JOINTS = ["crank_to_coupler", "coupler_to_rocker"]

# Loop-closure anchors of the excluded ``rocker_to_ground`` joint (USD ``localPos0`` / ``localPos1``).
_ROCKER_ANCHOR = (0.0, 0.0, 0.0)
_GROUND_ANCHOR = (0.0, 0.2, 0.0)
_LOOP_CLOSURE_TOL = 1e-3


def _kamino_sim_cfg(use_cuda_graph: bool) -> SimulationCfg:
    # ``use_cuda_graph=True`` matches production Kamino. Double-buffered substeps fold into a
    # stable ``state_0`` via ``assign`` (see :meth:`NewtonManager._run_solver_substeps`) so
    # articulation bindings and Kamino FK stay on the same buffer.
    return SimulationCfg(
        dt=1.0 / 120.0,
        device=DEVICE,
        gravity=(0.0, 0.0, -9.81),
        physics=NewtonCfg(
            solver_cfg=KaminoSolverCfg(use_fk_solver=True, use_collision_detector=True),
            use_cuda_graph=use_cuda_graph,
        ),
    )


def _spawn_articulation(cfg) -> Articulation:
    """Spawn ``cfg`` into ``NUM_ENVS`` separate Newton worlds at ``/World/Env_*/Robot``."""
    for i in range(NUM_ENVS):
        sim_utils.create_prim(f"/World/Env_{i}", "Xform", translation=(i * ENV_SPACING, 0.0, 0.0))
    return Articulation(cfg.replace(prim_path="/World/Env_.*/Robot"))


def _spawn_rigid_object() -> RigidObject:
    """Spawn a free rigid cube into ``NUM_ENVS`` separate Newton worlds."""
    for i in range(NUM_ENVS):
        sim_utils.create_prim(f"/World/Env_{i}", "Xform", translation=(i * ENV_SPACING, 0.0, 0.0))
    cfg = RigidObjectCfg(
        prim_path="/World/Env_.*/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.3, 0.3, 0.3),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
    )
    return RigidObject(cfg)


def _steady_state(sim, asset) -> None:
    """Reset, step once, and assert the seeded reset masks have been consumed."""
    sim.reset()
    sim.step(render=False)
    asset.update(sim.get_physics_dt())
    _assert_masks_clean()


def _world_reset_mask() -> torch.Tensor:
    return wp.to_torch(SimulationManager._world_reset_mask)


def _fk_reset_mask() -> torch.Tensor:
    return wp.to_torch(SimulationManager._fk_reset_mask)


def _assert_masks_clean() -> None:
    assert int(_world_reset_mask().abs().sum().item()) == 0, "world reset mask must be clean"
    assert not bool(_fk_reset_mask().any().item()), "fk reset mask must be clean"


def _body_index(asset, name: str) -> int:
    """Per-articulation body index for ``name`` in the order used by ``body_link_pose_w``.

    ``body_link_pose_w`` is indexed by the Newton model's physical body order
    (``model.body_label``), not the view's joint-child ``body_names`` (which contains a
    phantom duplicate for the closed-loop joint).
    """
    labels = [str(label).split("/")[-1] for label in asset.root_view.model.body_label]
    return labels.index(name)


def _raw_link_transforms(asset) -> torch.Tensor:
    """Writable torch view (N, 1, B, 7) over ``state_0.body_q`` for the asset's links.

    Writing here perturbs the body cache *without* invalidating FK, so it can plant
    sentinels that a subsequent reconcile must (dirty) or must not (protected) overwrite.
    """
    return wp.to_torch(asset.root_view.get_link_transforms(SimulationManager.get_state_0()))


def _plant_body_sentinel(asset, env: int, body_id: int) -> torch.Tensor:
    """Write a far-away sentinel pose into ``env``'s ``body_id`` link; return the sentinel."""
    sentinel = torch.tensor([100.0 + env, 200.0, 300.0, 0.0, 0.0, 0.0, 1.0], device=DEVICE)
    _raw_link_transforms(asset)[env, 0, body_id, :] = sentinel
    return sentinel


def _loop_closure_error(asset) -> torch.Tensor:
    """Per-env distance between the two anchors of the excluded loop joint (triggers FK)."""
    pos_w = asset.data.body_link_pos_w.torch
    quat_w = asset.data.body_link_quat_w.torch
    rocker = _body_index(asset, "rocker")
    ground = _body_index(asset, "ground_link")
    rocker_anchor = torch.tensor(_ROCKER_ANCHOR, device=DEVICE).expand(NUM_ENVS, 3)
    ground_anchor = torch.tensor(_GROUND_ANCHOR, device=DEVICE).expand(NUM_ENVS, 3)
    rocker_world = pos_w[:, rocker] + quat_apply(quat_w[:, rocker], rocker_anchor)
    ground_world = pos_w[:, ground] + quat_apply(quat_w[:, ground], ground_anchor)
    return torch.linalg.norm(rocker_world - ground_world, dim=-1)


def test_fixed_fourbar_partial_env_reset_via_write_and_read(use_cuda_graph: bool):
    """Writing only the actuated joints on a subset of envs reconciles those envs' bodies.

    The two passive joints are not written; the Kamino FK solve must move the passive-driven
    links (coupler / rocker) to satisfy the closed loop, while the actuated coordinates are
    preserved. Non-reset envs keep their (sentinel) body cache untouched.
    """
    with build_simulation_context(sim_cfg=_kamino_sim_cfg(use_cuda_graph)) as sim:
        assert sim.physics_manager.__name__ == "NewtonKaminoManager"

        articulation = _spawn_articulation(FOURBAR_POLE_CFG)
        _steady_state(sim, articulation)

        act_ids = articulation.find_joints(ACTUATED_JOINTS, preserve_order=True)[0]
        crank_id, pole_joint_id = act_ids
        passive_ids = articulation.find_joints(PASSIVE_JOINTS, preserve_order=True)[0]
        pole_body = _body_index(articulation, "pole")
        coupler_body = _body_index(articulation, "coupler")

        # Snapshot the settled state before perturbing anything.
        joint_pos_before = articulation.data.joint_pos.torch.clone()
        coupler_pos_before = articulation.data.body_link_pose_w.torch[:, coupler_body, :3].clone()

        # Plant a far-away sentinel on the pole link of a dirty and the protected env.
        _plant_body_sentinel(articulation, DIRTY_ENVS[0], pole_body)
        protected_sentinel = _plant_body_sentinel(articulation, PROTECTED_ENV, pole_body)

        # Distinct per-env actuated targets (within the +-60 deg crank range).
        crank_targets = torch.tensor([0.2, 0.4, 0.6], device=DEVICE)
        pole_targets = torch.tensor([-0.3, -0.6, -0.9], device=DEVICE)
        position = torch.stack([crank_targets, pole_targets], dim=1)
        velocity = torch.zeros_like(position)

        # Production-like reset: write only the actuated joints, only on the dirty envs.
        articulation.write_joint_position_to_sim_index(position=position, joint_ids=act_ids, env_ids=DIRTY_ENVS)
        articulation.write_joint_velocity_to_sim_index(velocity=velocity, joint_ids=act_ids, env_ids=DIRTY_ENVS)

        # Reading body poses triggers the lazy, masked FK reconcile.
        body_pose_after = articulation.data.body_link_pose_w.torch
        joint_pos_after = articulation.data.joint_pos.torch
        loop_err = _loop_closure_error(articulation)

        for i, env in enumerate(DIRTY_ENVS):
            # Actuated coordinates are preserved exactly.
            torch.testing.assert_close(joint_pos_after[env, crank_id], crank_targets[i], rtol=0.0, atol=1e-5)
            torch.testing.assert_close(joint_pos_after[env, pole_joint_id], pole_targets[i], rtol=0.0, atol=1e-5)
            # Passive joint coordinates were resolved (and written back) by the FK solve.
            passive_shift = torch.linalg.norm(joint_pos_after[env, passive_ids] - joint_pos_before[env, passive_ids])
            assert passive_shift > 1e-3, f"env {env}: passive joints not updated by FK"
            # Passive-driven coupler link moved to satisfy the new configuration.
            coupler_shift = torch.linalg.norm(body_pose_after[env, coupler_body, :3] - coupler_pos_before[env])
            assert coupler_shift > 1e-3, f"env {env}: coupler did not move (passive FK not applied)"
            # Loop closure holds after the reconcile.
            assert loop_err[env] < _LOOP_CLOSURE_TOL, f"env {env}: loop closure error {loop_err[env].item()}"

        # The dirty env's pole sentinel was overwritten by the FK solve.
        dirty_pole = body_pose_after[DIRTY_ENVS[0], pole_body]
        assert torch.linalg.norm(dirty_pole[:3] - torch.tensor([100.0, 200.0, 300.0], device=DEVICE)) > 1.0

        # Protected env: pole sentinel survives and joints are unchanged.
        torch.testing.assert_close(body_pose_after[PROTECTED_ENV, pole_body], protected_sentinel, rtol=0.0, atol=1e-6)
        torch.testing.assert_close(joint_pos_after[PROTECTED_ENV], joint_pos_before[PROTECTED_ENV], rtol=0.0, atol=1e-6)

        _assert_masks_clean()


def test_floating_fourbar_root_and_joint_reset_partial_env(use_cuda_graph: bool):
    """Floating-base reset drives per-env base pose/twist anchoring through ``write_root_*``.

    A distinct root pose and velocity is written per dirty env (landing in the free joint's
    ``joint_q[0:7]`` / ``joint_qd[0:6]`` head) along with the actuated joints. The reconcile
    must anchor each dirty env's base body to its target, preserve the written actuated
    coordinates, resolve passive joint coordinates from the loop closure, and leave the
    protected env alone.
    """
    with build_simulation_context(sim_cfg=_kamino_sim_cfg(use_cuda_graph)) as sim:
        assert sim.physics_manager.__name__ == "NewtonKaminoManager"

        articulation = _spawn_articulation(FOURBAR_POLE_FLOATING_CFG)
        _steady_state(sim, articulation)

        act_ids = articulation.find_joints(ACTUATED_JOINTS, preserve_order=True)[0]
        crank_id, pole_joint_id = act_ids
        passive_ids = articulation.find_joints(PASSIVE_JOINTS, preserve_order=True)[0]
        ground_body = _body_index(articulation, "ground_link")
        pole_body = _body_index(articulation, "pole")

        # Snapshot settled joint state before perturbing anything.
        joint_pos_before = articulation.data.joint_pos.torch.clone()

        # Snapshot the protected env's base pose; plant a sentinel on its pole link.
        ground_pose_before = articulation.data.body_link_pose_w.torch[:, ground_body].clone()
        protected_sentinel = _plant_body_sentinel(articulation, PROTECTED_ENV, pole_body)

        # Distinct per-env base pose (position + yaw) and twist (linear + angular).
        root_pose = torch.zeros((len(DIRTY_ENVS), 7), device=DEVICE)
        root_vel = torch.zeros((len(DIRTY_ENVS), 6), device=DEVICE)
        for i, env in enumerate(DIRTY_ENVS):
            yaw = 0.1 * (env + 1)
            root_pose[i] = torch.tensor(
                [10.0 + env, -5.0 - env, 1.0 + 0.5 * env, 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)],
                device=DEVICE,
            )
            root_vel[i] = torch.tensor(
                [0.1 * (env + 1), -0.2 * (env + 1), 0.05 * (env + 1), 0.3 * (env + 1), 0.0, -0.15 * (env + 1)],
                device=DEVICE,
            )
        crank_targets = torch.tensor([0.2, 0.3, 0.4], device=DEVICE)
        pole_targets = torch.tensor([-0.2, -0.3, -0.4], device=DEVICE)
        joint_position = torch.stack([crank_targets, pole_targets], dim=1)

        # Reset root state + actuated joints on the dirty envs only.
        articulation.write_root_link_pose_to_sim_index(root_pose=root_pose, env_ids=DIRTY_ENVS)
        articulation.write_root_link_velocity_to_sim_index(root_velocity=root_vel, env_ids=DIRTY_ENVS)
        articulation.write_joint_position_to_sim_index(position=joint_position, joint_ids=act_ids, env_ids=DIRTY_ENVS)

        # Trigger the masked reconcile.
        body_pose_after = articulation.data.body_link_pose_w.torch
        body_vel_after = articulation.data.body_link_vel_w.torch
        joint_pos_after = articulation.data.joint_pos.torch
        loop_err = _loop_closure_error(articulation)

        for i, env in enumerate(DIRTY_ENVS):
            torch.testing.assert_close(body_pose_after[env, ground_body], root_pose[i], rtol=0.0, atol=1e-4)
            torch.testing.assert_close(body_vel_after[env, ground_body], root_vel[i], rtol=0.0, atol=1e-4)
            # Actuated coordinates are preserved exactly.
            torch.testing.assert_close(joint_pos_after[env, crank_id], crank_targets[i], rtol=0.0, atol=1e-5)
            torch.testing.assert_close(joint_pos_after[env, pole_joint_id], pole_targets[i], rtol=0.0, atol=1e-5)
            # Passive joint coordinates were resolved (and written back) by the FK solve.
            passive_shift = torch.linalg.norm(joint_pos_after[env, passive_ids] - joint_pos_before[env, passive_ids])
            assert passive_shift > 1e-3, f"env {env}: passive joints not updated by FK"
            assert loop_err[env] < _LOOP_CLOSURE_TOL, f"env {env}: loop closure error {loop_err[env].item()}"

        # Protected env: base pose unchanged, pole sentinel survives, joints are unchanged.
        torch.testing.assert_close(
            body_pose_after[PROTECTED_ENV, ground_body], ground_pose_before[PROTECTED_ENV], rtol=0.0, atol=1e-6
        )
        torch.testing.assert_close(body_pose_after[PROTECTED_ENV, pole_body], protected_sentinel, rtol=0.0, atol=1e-6)
        torch.testing.assert_close(joint_pos_after[PROTECTED_ENV], joint_pos_before[PROTECTED_ENV], rtol=0.0, atol=1e-6)

        _assert_masks_clean()


def test_rigid_object_partial_env_root_reset(use_cuda_graph: bool):
    """A free rigid body (no articulation joints) reconciles only the reset envs' root pose."""
    with build_simulation_context(sim_cfg=_kamino_sim_cfg(use_cuda_graph)) as sim:
        assert sim.physics_manager.__name__ == "NewtonKaminoManager"

        cube = _spawn_rigid_object()
        _steady_state(sim, cube)

        # Snapshot protected env pose; plant a sentinel on every env's single body.
        pose_before = cube.data.body_link_pose_w.torch[:, 0].clone()
        _plant_body_sentinel(cube, DIRTY_ENVS[0], 0)
        protected_sentinel = _plant_body_sentinel(cube, PROTECTED_ENV, 0)

        root_pose = torch.zeros((len(DIRTY_ENVS), 7), device=DEVICE)
        for i, env in enumerate(DIRTY_ENVS):
            yaw = 0.2 * (env + 1)
            root_pose[i] = torch.tensor(
                [3.0 + env, 2.0 + env, 1.5 + 0.25 * env, 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)],
                device=DEVICE,
            )
        cube.write_root_link_pose_to_sim_index(root_pose=root_pose, env_ids=DIRTY_ENVS)

        body_pose_after = cube.data.body_link_pose_w.torch

        for i, env in enumerate(DIRTY_ENVS):
            torch.testing.assert_close(body_pose_after[env, 0], root_pose[i], rtol=0.0, atol=1e-4)

        torch.testing.assert_close(body_pose_after[PROTECTED_ENV, 0], protected_sentinel, rtol=0.0, atol=1e-6)
        # Sanity: the protected env's pose is the planted sentinel, not the pre-write pose.
        assert not torch.allclose(body_pose_after[PROTECTED_ENV, 0], pose_before[PROTECTED_ENV], atol=1e-3)

        _assert_masks_clean()


def test_lazy_fk_masks_until_body_pose_read(use_cuda_graph: bool):
    """The reset masks persist until a body-pose read, then are consumed by the reconcile."""
    with build_simulation_context(sim_cfg=_kamino_sim_cfg(use_cuda_graph)) as sim:
        assert sim.physics_manager.__name__ == "NewtonKaminoManager"

        articulation = _spawn_articulation(FOURBAR_POLE_CFG)
        _steady_state(sim, articulation)

        act_ids = articulation.find_joints(ACTUATED_JOINTS, preserve_order=True)[0]
        pole_body = _body_index(articulation, "pole")
        protected_sentinel = _plant_body_sentinel(articulation, PROTECTED_ENV, pole_body)

        position = torch.tensor([[0.2, -0.2], [0.3, -0.3], [0.4, -0.4]], device=DEVICE)
        articulation.write_joint_position_to_sim_index(position=position, joint_ids=act_ids, env_ids=DIRTY_ENVS)

        # Before any body-pose read: the dirty worlds are flagged and the protected sentinel
        # is still present in the raw body cache (no reconcile has run yet).
        world_mask = _world_reset_mask()
        assert int(world_mask.sum().item()) == len(DIRTY_ENVS)
        for env in DIRTY_ENVS:
            assert int(world_mask[env].item()) == 1
        assert int(world_mask[PROTECTED_ENV].item()) == 0
        raw_protected = _raw_link_transforms(articulation)[PROTECTED_ENV, 0, pole_body].clone()
        torch.testing.assert_close(raw_protected, protected_sentinel, rtol=0.0, atol=1e-6)

        # Reading body poses runs the reconcile and consumes the masks.
        _ = articulation.data.body_link_pose_w.torch
        _assert_masks_clean()
