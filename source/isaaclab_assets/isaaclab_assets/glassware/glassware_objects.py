from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.spawners.wrappers import MultiUsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

@configclass
class Chem_Assets:
        cube_properties = RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
        )

        def flask(self,pos=[0.65, 0.4, 0.05],rot=[1, 0, 0, 0], name="Flask")-> RigidObjectCfg:
           return RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + name,
                init_state=RigidObjectCfg.InitialStateCfg(pos=pos,rot=rot),
                spawn=UsdFileCfg(
                    usd_path="/workspace/isaaclab/source/isaaclab_assets/data/Props/glassware/conical_flask.usd",
                    scale=(1, 1, 1),
                    rigid_props=self.cube_properties,
                    visible=True,
                    copy_from_source = False,
                    semantic_tags=[("class", "Flask")],
                ),
            ) 

        def vial (self,pos=[0.65, 0.3, 0.05],rot=[0, 0, 1, 0], name="Vial")-> RigidObjectCfg:
            return RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + name,
                init_state=RigidObjectCfg.InitialStateCfg(pos=pos,rot=rot),
                spawn=UsdFileCfg(
                    usd_path="/workspace/isaaclab/source/isaaclab_assets/data/Props/glassware/sample_vial_20ml.usd",
                    scale=(1, 1, 1),
                    rigid_props=self.cube_properties,
                    visible=True,
                    copy_from_source = False,
                    semantic_tags=[("class", name)],
                ),
            ) 

        def beaker(self,pos=[0.4, 0.35, 0.0203],rot=[0, 0, 1, 0], name="Beaker")-> RigidObjectCfg:
            return RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + name,
                init_state=RigidObjectCfg.InitialStateCfg(pos = pos,rot=rot ),
                spawn=UsdFileCfg(
                    usd_path="/workspace/isaaclab/source/isaaclab_assets/data/Props/glassware/beaker.usd",
                    scale=(0.5, 0.5, 0.5),
                    rigid_props=self.cube_properties,
                    semantic_tags=[("class", name)],
                ),
            )

        def stirplate(self,pos=[0.50, -0.3, 0.05],rot=[0.707, 0, 0, -0.707], name="Stirplate")-> RigidObjectCfg:
            return RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + name ,
                init_state=RigidObjectCfg.InitialStateCfg(pos=pos,rot=rot),
                spawn=UsdFileCfg(
                    usd_path="/workspace/isaaclab/source/isaaclab_assets/data/Props/lab_equipment/mag_hotplate.usd",
                    scale=(0.8, 0.8, 0.8),
                    rigid_props=self.cube_properties,
                    semantic_tags=[("class", name)],
                ),
            ) 
        
        def random_object(self,pos=[0.65, 0.3, 0.05],rot=[0, 0, 1, 0], name="random")-> RigidObjectCfg:
            return RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + name ,
                init_state=RigidObjectCfg.InitialStateCfg(pos=pos,rot=rot),
                spawn=MultiUsdFileCfg(
                    usd_path=["/workspace/isaaclab/source/isaaclab_assets/data/Props/glassware/beaker.usd", "/workspace/isaaclab/source/isaaclab_assets/data/Props/glassware/sample_vial_20ml.usd"],
                    scale=(0.8, 0.8, 0.8),
                    rigid_props=self.cube_properties,
                    semantic_tags=[("class", name)],
                    random_choice = True
                ),
            )
                
            
