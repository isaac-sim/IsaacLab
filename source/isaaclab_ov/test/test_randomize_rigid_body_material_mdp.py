# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Real-backend test for the OVPhysX branch of the ``randomize_rigid_body_material`` MDP term.

Constructs the public term with an ``EventTermCfg`` and verifies per-shape friction/restitution
writes through a real OVPhysX rigid object's ``OvPhysxView``.

Kitless; the CPU and CUDA cases run in one process.
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
from isaaclab.managers import EventTermCfg, SceneEntityCfg  # noqa: E402
from isaaclab.sim import SimulationCfg, build_simulation_context  # noqa: E402
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR  # noqa: E402

wp.init()


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
            "asset_cfg": SceneEntityCfg("cube", body_ids=[0]),
        }
        env = SimpleNamespace(sim=sim, scene={"cube": cube_object})

        cfg = EventTermCfg(func=randomize_rigid_body_material, mode="reset", params=params)
        term = randomize_rigid_body_material(cfg, env)
        term(env, None, **cfg.params)

        materials = wp.to_torch(cube_object.root_view.get_attribute(TT.RIGID_BODY_SHAPE_FRICTION_AND_RESTITUTION))
        assert materials.shape[0] == num_cubes and materials.shape[-1] == 3
        eps = 1e-5
        for component, (lo, hi) in enumerate((static_range, dynamic_range, restitution_range)):
            values = materials[..., component]
            assert (values >= lo - eps).all() and (values <= hi + eps).all()

        cfg.params["asset_cfg"] = SceneEntityCfg("cube", body_ids=[])
        with pytest.raises(NotImplementedError, match="per-body"):
            randomize_rigid_body_material(cfg, env)
