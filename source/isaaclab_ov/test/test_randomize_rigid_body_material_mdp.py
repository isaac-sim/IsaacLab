# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Real-backend test for the OVPhysX branch of the ``randomize_rigid_body_material`` MDP term.

Drives the public :class:`isaaclab.envs.mdp.events.randomize_rigid_body_material` term against a real
OVPhysX :class:`~isaaclab_ov.assets.RigidObject`, so the backend dispatch itself is exercised (the
``ovphysxmanager`` name also contains ``physx``), and verifies that it writes per-shape
friction/restitution through the asset's ``OvPhysxView``. The ``cfg`` / ``env`` / ``asset_cfg`` inputs
are stubbed: the term only reads ``cfg.params``, ``env.scene[...]`` and ``env.sim.physics_manager``.

Kitless; run once per device (``-k cpu`` / ``-k 'cuda:0'``) -- the ovphysx runtime binds the
device mode process-globally (see the asset tests' module docstring).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import warp as wp

pytest.importorskip("ovphysx.types", reason="ovphysx wheel not installed")

from isaaclab_ov import tensor_types as TT  # noqa: E402
from isaaclab_ov.assets import RigidObject  # noqa: E402
from isaaclab_ov.physics import OvPhysxCfg  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import RigidObjectCfg  # noqa: E402
from isaaclab.envs.mdp.events import randomize_rigid_body_material  # noqa: E402
from isaaclab.sim import SimulationCfg, build_simulation_context  # noqa: E402
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR  # noqa: E402

wp.init()

pytestmark = pytest.mark.device_split

_LOCKED_DEVICE: list[str | None] = [None]


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


def _make_cubes(num_cubes: int, device: str) -> RigidObject:
    """Spawn ``num_cubes`` rigid-body cubes as a single RigidObject."""
    for i in range(num_cubes):
        sim_utils.create_prim(f"/World/Table_{i}", "Xform", translation=(i * 1.0, 0.0, 1.0))
    cfg = RigidObjectCfg(
        prim_path="/World/Table_[^/]+/Object",
        spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd"),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
    )
    return RigidObject(cfg=cfg)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_randomize_material_writes_friction_within_range(device):
    """The term should dispatch to OVPhysX and write per-shape friction/restitution within the given ranges.

    A per-body selection on a standalone rigid object must fail loud (no per-body shape counts).
    """
    num_cubes = 2
    with _ovphysx_sim_context(device=device, auto_add_lighting=True) as sim:
        cube_object = _make_cubes(num_cubes, device)
        sim.reset()

        # The ranges exclude the asset's default material, so values inside them prove the write happened.
        static_range, dynamic_range, restitution_range = (0.9, 1.2), (0.7, 0.9), (0.4, 0.6)
        materials_before = wp.to_torch(
            cube_object.root_view.get_attribute(TT.RIGID_BODY_SHAPE_FRICTION_AND_RESTITUTION)
        ).clone()
        assert (materials_before[..., 0] < static_range[0]).all(), f"default material overlaps: {materials_before}"
        params = {
            "static_friction_range": static_range,
            "dynamic_friction_range": dynamic_range,
            "restitution_range": restitution_range,
            "num_buckets": 16,
        }
        asset_cfg = SimpleNamespace(name="cube", body_ids=[0])
        env = SimpleNamespace(sim=sim, scene={"cube": cube_object})

        term = randomize_rigid_body_material(SimpleNamespace(params={**params, "asset_cfg": asset_cfg}), env)
        term(env, None, static_range, dynamic_range, restitution_range, 16, asset_cfg)

        materials = wp.to_torch(cube_object.root_view.get_attribute(TT.RIGID_BODY_SHAPE_FRICTION_AND_RESTITUTION))
        assert materials.shape[0] == num_cubes and materials.shape[-1] == 3
        eps = 1e-5
        for component, (lo, hi) in enumerate((static_range, dynamic_range, restitution_range)):
            values = materials[..., component]
            assert (values >= lo - eps).all() and (values <= hi + eps).all()

        subset_cfg = SimpleNamespace(name="cube", body_ids=[])  # proper subset of the rigid object's single body
        with pytest.raises(NotImplementedError, match="per-body"):
            randomize_rigid_body_material(SimpleNamespace(params={**params, "asset_cfg": subset_cfg}), env)
