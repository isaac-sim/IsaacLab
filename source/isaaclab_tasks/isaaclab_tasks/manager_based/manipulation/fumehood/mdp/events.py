from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING, Literal

import carb
import omni.physics.tensors.impl.api as physx
import omni.usd
from isaacsim.core.utils.extensions import enable_extension
from pxr import Gf, Sdf, UsdGeom, Vt

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.actuators import ImplicitActuator
from isaaclab.assets import Articulation, DeformableObject, RigidObject
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv



def randomize_object(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    
):
    """Randomise the object to use?
    .. tip::
        This function uses CPU tensors to assign the body masses. It is recommended to use this function
        only during the initialization of the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")
    
    object: RigidObject = env.scene[asset_cfg.name]
    test_obj = env.scene["object_collection"].cfg
    print(f"Asset  : {asset_cfg.name}")
   # print(f"found flask object :  {test_obj.prim_path}")
    print(f"Asset spawn : {object.cfg.spawn}")
    for rigid_object in env.scene.rigid_objects.values():
        print(f"objects found  : {rigid_object.body_names}")

    #try to replace the asset 
   # env.scene.object = 
    # n = rd.randint(0,len(objects)-1)
    #     #some objects import upside down
    #     if n<2:
    #         rot = [0, 0, 1, 0]
    #     else :
    #         rot=[1, 0, 0, 0]
    #     print(f"Random spawn object : {objects[n]}")
    #     object = RigidObjectCfg(
    #         prim_path="{ENV_REGEX_NS}/Object",
    #         init_state=RigidObjectCfg.InitialStateCfg(pos=[0.4, 0.0, 0.0203], rot=rot),
    #         spawn=UsdFileCfg(
    #             usd_path=f"/workspace/isaaclab/source/isaaclab_assets/data/Props/glassware/{objects[n]}.usd",
    #             scale=(1.0, 1.0, 1.0),
    #             rigid_props=RigidBodyPropertiesCfg(
    #                 solver_position_iteration_count=16,
    #                 solver_velocity_iteration_count=1,
    #                 max_angular_velocity=1000.0,
    #                 max_linear_velocity=1000.0,
    #                 max_depenetration_velocity=5.0,
    #                 disable_gravity=False,
    #             ),
    #             semantic_tags=[("class", "object")],
    #         ),
    #     )
        
