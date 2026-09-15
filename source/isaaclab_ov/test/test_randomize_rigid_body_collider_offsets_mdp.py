# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Real-backend test for the OVPhysX branch of the ``randomize_rigid_body_collider_offsets`` MDP term.

Drives the public :class:`isaaclab.envs.mdp.events.randomize_rigid_body_collider_offsets` term against
a real OVPhysX :class:`~isaaclab_ov.assets.RigidObject` and :class:`~isaaclab_ov.assets.Articulation`, so
the backend dispatch itself is exercised (it previously fell through to the PhysX implementation, whose
``root_view`` accessors do not exist on :class:`~isaaclab_ov.sim.views.OvPhysxView`). The ``cfg`` /
``env`` / ``asset_cfg`` inputs are stubbed: the term only reads ``cfg.params["asset_cfg"]``,
``env.scene[...]``, ``env.scene.num_envs`` and ``env.sim.physics_manager``.

Kitless; run once per device (``-k cpu`` / ``-k 'cuda:0'``) -- the ovphysx runtime binds the
device mode process-globally (see the asset tests' module docstring).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import warp as wp

pytest.importorskip("ovphysx.types", reason="ovphysx wheel not installed")

from isaaclab_ov import tensor_types as TT  # noqa: E402
from isaaclab_ov.assets import Articulation, RigidObject  # noqa: E402
from isaaclab_ov.physics import OvPhysxCfg  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import RigidObjectCfg  # noqa: E402
from isaaclab.envs.mdp.events import randomize_rigid_body_collider_offsets  # noqa: E402
from isaaclab.sim import SimulationCfg, build_simulation_context  # noqa: E402
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR  # noqa: E402

from isaaclab_assets import CARTPOLE_CFG  # isort:skip  # noqa: E402

wp.init()

pytestmark = pytest.mark.device_split

_LOCKED_DEVICE: list[str | None] = [None]

REST_RANGE = (0.0002, 0.0006)  # below the 1 mm default contact offset: PhysX rejects rest >= contact
CONTACT_RANGE = (0.03, 0.05)
EPS = 1e-6


@pytest.fixture(autouse=True)
def _ovphysx_skip_other_device(request):
    """Skip parametrized tests on the device the session is not pinned to (process-global lock)."""
    callspec = getattr(request.node, "callspec", None)
    device = callspec.params.get("device") if callspec is not None else None
    if device is None:
        return
    locked = _LOCKED_DEVICE[0]
    if locked is None:
        _LOCKED_DEVICE[0] = device
        return
    if device != locked:
        pytest.skip(
            f"ovphysx process-global device lock is held by '{locked}'; cannot run '{device}' "
            "tests in the same session.  Run pytest twice (once per device) for full coverage."
        )


def _ovphysx_sim_context(device: str, **kwargs):
    """Build a simulation context that dispatches to the OVPhysX manager."""
    sim_cfg = SimulationCfg(physics=OvPhysxCfg(), device=device, dt=1.0 / 60.0, gravity=(0.0, 0.0, -9.81))
    return build_simulation_context(device=device, sim_cfg=sim_cfg, **kwargs)


class _SceneStub(dict):
    """Minimal ``InteractiveScene`` stand-in: name lookup plus ``num_envs``."""

    def __init__(self, num_envs: int, **assets):
        super().__init__(**assets)
        self.num_envs = num_envs


def _make_env(sim, num_envs: int, **assets) -> SimpleNamespace:
    return _make_env_from_scene(sim, _SceneStub(num_envs, **assets))


def _make_env_from_scene(sim, scene: _SceneStub) -> SimpleNamespace:
    return SimpleNamespace(sim=sim, scene=scene, num_envs=scene.num_envs, device=sim.device)


def _make_cubes(num_cubes: int) -> RigidObject:
    """Spawn ``num_cubes`` rigid-body cubes as a single RigidObject."""
    for i in range(num_cubes):
        sim_utils.create_prim(f"/World/Env_{i}", "Xform", translation=(i * 1.0, 0.0, 1.0))
    cfg = RigidObjectCfg(
        prim_path="/World/Env_[^/]+/Object",
        spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd"),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
    )
    return RigidObject(cfg=cfg)


def _make_cartpoles(num_envs: int) -> Articulation:
    """Spawn ``num_envs`` cartpoles as a single Articulation."""
    for i in range(num_envs):
        sim_utils.create_prim(f"/World/Env_{i}", "Xform", translation=(i * 2.5, 0.0, 0.0))
    return Articulation(cfg=CARTPOLE_CFG.replace(prim_path="/World/Env_[^/]+/Robot"))


def _read_offsets(asset, rest_type, contact_type) -> tuple[torch.Tensor, torch.Tensor]:
    rest = wp.to_torch(asset.root_view.get_attribute(rest_type)).clone()
    contact = wp.to_torch(asset.root_view.get_attribute(contact_type)).clone()
    return rest, contact


def _assert_randomized_rows(before: torch.Tensor, after: torch.Tensor, rows: list[int], value_range) -> None:
    """Selected rows land within ``value_range``; all other rows are untouched."""
    lo, hi = value_range
    selected = after[rows]
    assert (selected >= lo - EPS).all() and (selected <= hi + EPS).all(), selected
    untouched = [i for i in range(before.shape[0]) if i not in rows]
    torch.testing.assert_close(after[untouched], before[untouched])


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_rigid_object_offsets_randomized_for_selected_envs(device):
    """The term must dispatch to an OVPhysX implementation and write rigid-body rest/contact offsets."""
    num_cubes = 3
    with _ovphysx_sim_context(device=device, auto_add_lighting=True) as sim:
        cube_object = _make_cubes(num_cubes)
        sim.reset()

        asset_cfg = SimpleNamespace(name="cube", body_ids=slice(None))
        env = _make_env(sim, num_cubes, cube=cube_object)
        term = randomize_rigid_body_collider_offsets(SimpleNamespace(params={"asset_cfg": asset_cfg}), env)

        rest_before, contact_before = _read_offsets(
            cube_object, TT.RIGID_BODY_REST_OFFSET, TT.RIGID_BODY_CONTACT_OFFSET
        )
        assert rest_before.shape[0] == num_cubes
        # sanity: the sampling ranges must be disjoint from the authored defaults or the test proves nothing
        assert not ((rest_before >= REST_RANGE[0]) & (rest_before <= REST_RANGE[1])).any()
        assert not ((contact_before >= CONTACT_RANGE[0]) & (contact_before <= CONTACT_RANGE[1])).any()

        env_ids = torch.tensor([0, 2], device=device)
        term(
            env,
            env_ids,
            asset_cfg,
            rest_offset_distribution_params=REST_RANGE,
            contact_offset_distribution_params=CONTACT_RANGE,
        )

        rest_after, contact_after = _read_offsets(cube_object, TT.RIGID_BODY_REST_OFFSET, TT.RIGID_BODY_CONTACT_OFFSET)
        _assert_randomized_rows(rest_before, rest_after, [0, 2], REST_RANGE)
        _assert_randomized_rows(contact_before, contact_after, [0, 2], CONTACT_RANGE)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_articulation_offsets_randomized_for_all_envs(device):
    """The articulation path writes per-shape rest/contact offsets on every env when ``env_ids`` is None."""
    num_envs = 2
    with _ovphysx_sim_context(device=device, auto_add_lighting=True) as sim:
        articulation = _make_cartpoles(num_envs)
        sim.reset()

        asset_cfg = SimpleNamespace(name="robot", body_ids=slice(None))
        env = _make_env(sim, num_envs, robot=articulation)
        term = randomize_rigid_body_collider_offsets(SimpleNamespace(params={"asset_cfg": asset_cfg}), env)

        rest_before, contact_before = _read_offsets(articulation, TT.REST_OFFSET, TT.CONTACT_OFFSET)
        assert rest_before.shape[0] == num_envs
        assert not ((rest_before >= REST_RANGE[0]) & (rest_before <= REST_RANGE[1])).any()
        assert not ((contact_before >= CONTACT_RANGE[0]) & (contact_before <= CONTACT_RANGE[1])).any()

        # only rest offsets requested: contact offsets must stay untouched
        term(env, None, asset_cfg, rest_offset_distribution_params=REST_RANGE)
        rest_after, contact_after = _read_offsets(articulation, TT.REST_OFFSET, TT.CONTACT_OFFSET)
        _assert_randomized_rows(rest_before, rest_after, list(range(num_envs)), REST_RANGE)
        torch.testing.assert_close(contact_after, contact_before)

        term(env, None, asset_cfg, contact_offset_distribution_params=CONTACT_RANGE)
        _, contact_after = _read_offsets(articulation, TT.REST_OFFSET, TT.CONTACT_OFFSET)
        _assert_randomized_rows(contact_before, contact_after, list(range(num_envs)), CONTACT_RANGE)
