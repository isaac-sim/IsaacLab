# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Real-backend test for the OVPhysX branch of the ``randomize_rigid_body_material`` MDP term.

Exercises :class:`isaaclab.envs.mdp.events._RandomizeRigidBodyMaterialOvPhysx` against a real
OVPhysX :class:`~isaaclab_ovphysx.assets.RigidObject`, verifying that it writes per-shape
friction/restitution through the asset's ``OvPhysxView``. The ``cfg`` / ``env`` / ``asset_cfg``
inputs are stubbed (the OVPhysX implementation only reads ``cfg.params`` and
``asset_cfg.body_ids`` and operates on ``asset.root_view``).

Kitless; run once per device (``-k cpu`` / ``-k 'cuda:0'``) -- the ovphysx runtime binds the
device mode process-globally (see the asset tests' module docstring).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import warp as wp

pytest.importorskip("ovphysx.types", reason="ovphysx wheel not installed")

from isaaclab_ovphysx import tensor_types as TT  # noqa: E402
from isaaclab_ovphysx.assets import RigidObject  # noqa: E402
from isaaclab_ovphysx.physics import OvPhysxCfg  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import RigidObjectCfg  # noqa: E402
from isaaclab.envs.mdp.events import _RandomizeRigidBodyMaterialOvPhysx  # noqa: E402
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
        prim_path="/World/Table_*/Object",
        spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd"),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
    )
    return RigidObject(cfg=cfg)


@pytest.mark.parametrize("num_cubes", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_randomize_material_writes_friction_within_range(num_cubes, device):
    """The OVPhysX impl should write per-shape friction/restitution sampled within the given ranges."""
    with _ovphysx_sim_context(device=device, auto_add_lighting=True) as sim:
        cube_object = _make_cubes(num_cubes, device)
        sim.reset()

        static_range, dynamic_range, restitution_range = (0.4, 0.8), (0.2, 0.6), (0.0, 0.3)
        cfg = SimpleNamespace(
            params={
                "static_friction_range": static_range,
                "dynamic_friction_range": dynamic_range,
                "restitution_range": restitution_range,
                "num_buckets": 16,
            }
        )
        asset_cfg = SimpleNamespace(body_ids=[0])
        env = SimpleNamespace()  # unused by the OVPhysX implementation

        impl = _RandomizeRigidBodyMaterialOvPhysx(cfg, env, cube_object, asset_cfg)
        impl(env, None, static_range, dynamic_range, restitution_range, 16, asset_cfg)

        materials = wp.to_torch(cube_object.root_view.get_attribute(TT.RIGID_BODY_SHAPE_FRICTION_AND_RESTITUTION))
        assert materials.shape[0] == num_cubes and materials.shape[-1] == 3
        eps = 1e-5
        for component, (lo, hi) in enumerate((static_range, dynamic_range, restitution_range)):
            values = materials[..., component]
            assert (values >= lo - eps).all() and (values <= hi + eps).all()


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_randomize_material_body_subset_unsupported(device):
    """A per-body selection must fail loud on OVPhysX (no per-body shape counts)."""
    with _ovphysx_sim_context(device=device, auto_add_lighting=True) as sim:
        cube_object = _make_cubes(1, device)
        sim.reset()

        cfg = SimpleNamespace(params={"static_friction_range": (0.4, 0.8), "num_buckets": 4})
        asset_cfg = SimpleNamespace(body_ids=[])  # proper subset of the rigid object's single body
        with pytest.raises(NotImplementedError, match="per-body"):
            _RandomizeRigidBodyMaterialOvPhysx(cfg, SimpleNamespace(), cube_object, asset_cfg)
