import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from robot.a10_single_cfg import A10_SINGLE_CFG

ASSET_DIR = "A10_Single/assets"


@configclass
class A10SceneCfg(InteractiveSceneCfg):
    """Standalone A10 scene for pi policy evaluation."""

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=(0.55, 0.77, 0.10),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.05, 0.05), roughness=0.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.95, 0.0, 0.0),
        ),
    )

    apple = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Apple",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ASSET_DIR}/apple.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.90, 0.25, 0.105)),
    )

    lemon = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Lemon",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ASSET_DIR}/lemon.usd",
            scale=(3.5, 3.5, 3.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.90, 0, 0.105)),
    )

    strawberry = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Strawberry",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ASSET_DIR}/strawberry.usd",
            scale=(1.6, 1.6, 1.6),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.00, 0.21, 0.105)),
    )

    robot = A10_SINGLE_CFG.replace(  # type: ignore
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=A10_SINGLE_CFG.init_state.replace(
            pos=(0.15, 0.0, 0.0),
            rot=(0.7071068, 0.0, 0.0, -0.7071068),
        ),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    # Single wrist camera for policy observation.
    # Camera prim is already authored inside robot USD, so we bind to it directly.
    wrist_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/link6/wrist_camera",
        update_period=0.0,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=None,
    )

    # Fixed camera on robot base (spawned under base_link; tune offset for your layout).
    base_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/base_camera",
        update_period=0.0,
        height=1080,
        width=1920,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=0.5,
            focus_distance=400.0,
            horizontal_aperture=1.0,
            vertical_aperture=0.8,
            clipping_range=(0.1, 1.0e5),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.48, 0.48, 0.508),
            rot=(0.3091, 0.08182, -0.4775, -0.81839),
            convention="opengl",
        ),
    )
