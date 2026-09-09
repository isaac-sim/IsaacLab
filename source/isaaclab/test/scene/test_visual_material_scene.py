# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scene-level regression test for cloned visual materials and bindings."""

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True).app

"""Everything else follows."""

import pytest
import torch

from pxr import Sdf, UsdShade

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, VisualMaterial, VisualMaterialCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import build_simulation_context
from isaaclab.utils.configclass import configclass

pytestmark = pytest.mark.integration


@configclass
class _VisualMaterialSceneCfg(InteractiveSceneCfg):
    # Deliberately declared first: nested materials must wait for the heterogeneous Robot prototypes.
    warm = VisualMaterialCfg(
        prim_path="{ENV_REGEX_NS}/Robot/warm",
        spawn=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.1, 0.1)),
    )
    cool = VisualMaterialCfg(
        prim_path="{ENV_REGEX_NS}/Robot/cool",
        spawn=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.1, 0.8)),
    )
    shared = VisualMaterialCfg(
        prim_path="/World/shared",
        spawn=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.8, 0.1)),
    )
    robot = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=[
                sim_utils.UsdFileCfg(
                    usd_path="",
                    scale=(0.8, 0.8, 0.8),
                    visual_material_bindings={"body": "warm"},
                ),
                sim_utils.UsdFileCfg(
                    usd_path="",
                    scale=(1.2, 1.2, 1.2),
                    visual_material_bindings={"body": "./cool"},
                ),
                sim_utils.UsdFileCfg(
                    usd_path="",
                    scale=(1.4, 1.4, 1.4),
                    visual_material_bindings={"body": "/World/shared"},
                ),
            ],
            random_choice=False,
        ),
        cloning_contexts=("isaaclab.cloner:UsdReplicateContext",),
    )


def test_nested_materials_clone_with_parent_rows_and_write_selected_gpu_rows(tmp_path) -> None:
    asset_path = tmp_path / "visual_material_robot.usda"
    asset_path.write_text(
        """#usda 1.0
(
    defaultPrim = "Robot"
)
def Xform "Robot"
{
    def Xform "body"
    {
        def Cube "mesh"
        {
            double size = 0.2
        }
    }
}
"""
    )

    with build_simulation_context(
        device="cuda:0", gravity_enabled=False, add_ground_plane=False, auto_add_lighting=False
    ) as sim:
        cfg = _VisualMaterialSceneCfg(num_envs=12, env_spacing=1.0, replicate_physics=False, filter_collisions=False)
        for variant in cfg.robot.spawn.assets_cfg:
            variant.usd_path = str(asset_path)

        scene = InteractiveScene(cfg)
        sim.reset()

        assert scene.num_envs == 12
        assert len(scene.clone_plan.sources) == 3
        assert scene.clone_plan.destinations == ("/World/envs/env_{}/Robot",) * 3
        assert scene.clone_plan.cfg_rows[id(cfg.robot)] == (0, 1, 2)
        assert id(cfg.warm) not in scene.clone_plan.cfg_rows
        assert id(cfg.cool) not in scene.clone_plan.cfg_rows
        assert scene["warm"].num_instances == scene["cool"].num_instances == 12
        assert scene["shared"].num_instances == 1
        for env_id in range(scene.num_envs):
            for cloned_material in ("warm", "cool"):
                assert scene.stage.GetPrimAtPath(f"/World/envs/env_{env_id}/Robot/{cloned_material}").IsValid()
            material_name = ("warm", "cool", None)[env_id % 3]
            material_path = (
                "/World/shared" if material_name is None else f"/World/envs/env_{env_id}/Robot/{material_name}"
            )
            binding = UsdShade.MaterialBindingAPI(
                scene.stage.GetPrimAtPath(f"/World/envs/env_{env_id}/Robot/body")
            ).GetDirectBindingRel()
            assert binding.GetTargets() == [Sdf.Path(material_path)]

        material = scene["warm"]
        before = material.data["color"].clone()
        env_ids = torch.tensor([8, 2], dtype=torch.int32, device=scene.device)
        colors = torch.tensor([[[0.2, 0.7, 0.3], [0.9, 0.4, 0.1]]], device=scene.device)
        VisualMaterial.write_channels([material], {"color": colors}, env_ids)
        torch.cuda.synchronize()

        expected = before.clone()
        expected[env_ids] = colors[0]
        torch.testing.assert_close(material.data["color"], expected)
        shader = UsdShade.Shader(scene.stage.GetPrimAtPath("/World/envs/env_2/Robot/warm/Shader"))
        assert tuple(shader.GetInput("diffuseColor").Get()) == pytest.approx((0.8, 0.1, 0.1))
