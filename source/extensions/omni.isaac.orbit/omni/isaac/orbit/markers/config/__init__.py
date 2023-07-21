from omni.isaac.orbit.utils.assets import ISAAC_NUCLEUS_DIR

from ..visualization_markers import VisualizationMarkersCfg

FRAME_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "frame": VisualizationMarkersCfg.FileMarkerCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
            scale=(0.5, 0.5, 0.5),
        )
    }
)
"""Configuration for the frame marker."""

POSITION_GOAL_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "target_far": VisualizationMarkersCfg.MarkerCfg(
            prim_type="Sphere",
            color=(1.0, 0.0, 0.0),
            attributes={"radius": 0.01},
        ),
        "target_near": VisualizationMarkersCfg.MarkerCfg(
            prim_type="Sphere",
            color=(0.0, 1.0, 0.0),
            attributes={"radius": 0.01},
        ),
        "target_invisible": VisualizationMarkersCfg.MarkerCfg(
            prim_type="Sphere",
            color=(0.0, 0.0, 1.0),
            attributes={"radius": 0.01},
            visible=False,
        ),
    }
)
"""Configuration for the end-effector tracking marker."""

RAY_CASTER_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "hit": VisualizationMarkersCfg.MarkerCfg(
            prim_type="Sphere",
            color=(1.0, 0.0, 0.0),
            attributes={"radius": 0.02},
        ),
    },
)
"""Configuration for the ray-caster marker."""


CONTACT_SENSOR_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "contact": VisualizationMarkersCfg.MarkerCfg(
            prim_type="Sphere",
            color=(1.0, 0.0, 0.0),
            attributes={"radius": 0.02},
        ),
        "no_contact": VisualizationMarkersCfg.MarkerCfg(
            visible=False,
            prim_type="Sphere",
            color=(0.0, 1.0, 0.0),
            attributes={"radius": 0.02},
        ),
    },
)
"""Configuration for the contact sensor marker."""


ARROW_X_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "arrow": VisualizationMarkersCfg.FileMarkerCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
            scale=[1.0, 0.1, 0.1],
        )
    }
)
"""Configuration for the arrow marker (along x-direction)."""

CUBOID_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "cuboid": VisualizationMarkersCfg.MarkerCfg(
            prim_type="Cube",
        )
    }
)
"""Configuration for the cuboid marker."""
