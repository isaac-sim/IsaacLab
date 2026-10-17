# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the OVPhysX multi-mesh ray caster."""

import pytest

# OVRTX-only CI jobs collect the consolidated isaaclab_ov test suite without
# the optional ovphysx wheel. Skip the OVPhysX tests gracefully in that case.
pytest.importorskip("ovphysx.types", reason="ovphysx wheel not installed")

import torch  # noqa: E402
from isaaclab_ov.physics import OvPhysxCfg  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import RigidObjectCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sensors.ray_caster import MultiMeshRayCasterCfg, patterns  # noqa: E402
from isaaclab.sim import SimulationCfg, build_simulation_context  # noqa: E402
from isaaclab.test.utils import DeviceScope, test_devices  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402


@configclass
class _StaticTargetSceneCfg(InteractiveSceneCfg):
    """A crate in each environment with a ray caster above it; the crate is the ray caster's static target."""

    env_spacing = 2.0
    crate = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Crate",
        spawn=sim_utils.CuboidCfg(
            size=(0.2, 0.2, 0.2),
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.1)),
    )
    # one ray straight down onto the top face of the crate
    scanner = MultiMeshRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Crate",
        offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 1.0)),
        mesh_prim_paths=[
            MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr="{ENV_REGEX_NS}/Crate", track_mesh_transforms=False)
        ],
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(0.0, 0.0)),
        ray_alignment="world",
    )


@pytest.mark.parametrize("device", test_devices(DeviceScope.DEFAULT_CUDA))
def test_multi_mesh_ray_caster_hits_static_target_of_each_env(device):
    """Each environment's ray hits its own copy of a static environment-local target."""
    sim_cfg = SimulationCfg(physics=OvPhysxCfg(), device=device)
    with build_simulation_context(device=device, sim_cfg=sim_cfg) as sim:
        scene = InteractiveScene(_StaticTargetSceneCfg(num_envs=4))
        sim.reset()

        hits = scene["scanner"].data.ray_hits_w.torch[:, 0]
        crate_top = scene["crate"].data.root_pos_w.torch + torch.tensor([0.0, 0.0, 0.1], device=device)
        torch.testing.assert_close(hits, crate_top)
