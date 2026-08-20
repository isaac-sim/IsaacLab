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

_USD_CONTEXT = ("isaaclab.cloner:UsdReplicateContext",)


@configclass
class _VisualMaterialSceneCfg(InteractiveSceneCfg):
    # Deliberately declared first: the scene must still spawn its material dependency first.
    cube = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(size=(0.2, 0.2, 0.2), visual_material_path="{ENV_REGEX_NS}/Materials/warm"),
        cloning_contexts=_USD_CONTEXT,
    )
    warm = VisualMaterialCfg(
        prim_path="{ENV_REGEX_NS}/Materials/warm",
        spawn=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.1, 0.1)),
        cloning_contexts=_USD_CONTEXT,
    )
    cool = VisualMaterialCfg(
        prim_path="{ENV_REGEX_NS}/Materials/cool",
        spawn=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.1, 0.8)),
        cloning_contexts=_USD_CONTEXT,
    )
    robot = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=[
                sim_utils.UsdFileCfg(
                    usd_path="",
                    scale=(0.8, 0.8, 0.8),
                    visual_material_bindings={"body": "{ENV_REGEX_NS}/Materials/warm"},
                ),
                sim_utils.UsdFileCfg(
                    usd_path="",
                    scale=(1.2, 1.2, 1.2),
                    visual_material_bindings={"body": "{ENV_REGEX_NS}/Materials/cool"},
                ),
            ],
            random_choice=False,
        ),
        cloning_contexts=_USD_CONTEXT,
    )


def test_scene_clones_material_bindings_round_robin_and_writes_selected_gpu_rows(tmp_path) -> None:
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
        cfg = _VisualMaterialSceneCfg(num_envs=10, env_spacing=1.0, replicate_physics=False, filter_collisions=False)
        for variant in cfg.robot.spawn.assets_cfg:
            variant.usd_path = str(asset_path)

        scene = InteractiveScene(cfg)
        sim.reset()

        assert scene.num_envs == 10
        assert scene["warm"].num_instances == scene["cool"].num_instances == 10
        for env_id in range(scene.num_envs):
            for cloned_material in ("warm", "cool"):
                assert scene.stage.GetPrimAtPath(f"/World/envs/env_{env_id}/Materials/{cloned_material}").IsValid()
            cube_binding = UsdShade.MaterialBindingAPI(
                scene.stage.GetPrimAtPath(f"/World/envs/env_{env_id}/Cube/geometry/mesh")
            ).GetDirectBindingRel()
            assert cube_binding.GetTargets() == [Sdf.Path(f"/World/envs/env_{env_id}/Materials/warm")]
            material_name = "warm" if env_id % 2 == 0 else "cool"
            material_path = f"/World/envs/env_{env_id}/Materials/{material_name}"
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
        shader = UsdShade.Shader(scene.stage.GetPrimAtPath("/World/envs/env_2/Materials/warm/Shader"))
        assert tuple(shader.GetInput("diffuseColor").Get()) == pytest.approx((0.8, 0.1, 0.1))
