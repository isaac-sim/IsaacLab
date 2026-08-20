# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Demonstrate per-environment visual-material randomization.

Five ANYmal-C appearance styles are assigned round-robin by heterogeneous cloning. Each styled
robot binds scene-declared :class:`~isaaclab.assets.VisualMaterial` assets for its body, legs, and
feet, so the three part groups and every environment randomize independently on partial resets.

The surface, glass, and solid styles exercise their numeric shader channels. Newton currently
mirrors material color only, so its demo run randomizes tint and per-shape colors while leaving the
other optical channels to RTX renderers.

.. code-block:: bash

    # PhysX physics and Kit visualizer.
    uv run --extra isaacsim python scripts/demos/visual_color_randomization.py

    # Newton physics and Newton GL visualizer.
    uv run python scripts/demos/visual_color_randomization.py \
        --physics newton_mjwarp --visualizer newton_gl

"""

from __future__ import annotations

import argparse

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description=__doc__, conflict_handler="resolve")
parser.add_argument("--num_envs", type=int, default=512, help="Number of environments to spawn.")
parser.add_argument(
    "--physics", default="isaacsim_physx", choices=["isaacsim_physx", "newton_mjwarp"], help="Physics backend."
)
add_launcher_args(parser)
parser.set_defaults(visualizer=["kit"])
args_cli = parser.parse_args()

import torch

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, VisualMaterialCfg
from isaaclab.envs import ManagerBasedEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.physics import PhysicsCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.timer import Timer

from isaaclab_assets.robots.anymal import ANYMAL_C_CFG  # isort: skip


_LEGS = ("LF", "LH", "RF", "RH")


def _bindings(body: str, legs: str, feet: str) -> dict[str, str]:
    """Bind the three material groups to the corresponding robot visuals."""
    return {
        "base/visuals": body,
        **{f"{leg}_{link}/visuals": legs for leg in _LEGS for link in ("HIP", "THIGH", "SHANK")},
        **{f"{leg}_FOOT/visuals": feet for leg in _LEGS},
    }


@configclass
class VisualMaterialSceneCfg(InteractiveSceneCfg):
    """One of five styled ANYmal variants per environment and three materials per style."""

    surface_body = VisualMaterialCfg(
        prim_path="{ENV_REGEX_NS}/Robot/surface_body",
        spawn=sim_utils.PreviewSurfaceCfg(),
        channels=("color", "roughness", "metallic", "emissive_color", "opacity"),
    )
    surface_leg = VisualMaterialCfg(
        prim_path="{ENV_REGEX_NS}/Robot/surface_leg",
        spawn=sim_utils.PreviewSurfaceCfg(),
        channels=("color", "roughness", "metallic", "emissive_color", "opacity"),
    )
    surface_foot = VisualMaterialCfg(
        prim_path="{ENV_REGEX_NS}/Robot/surface_foot",
        spawn=sim_utils.PreviewSurfaceCfg(),
        channels=("color", "roughness", "metallic", "emissive_color", "opacity"),
    )
    glass_body = VisualMaterialCfg(
        prim_path="{ENV_REGEX_NS}/Robot/glass_body",
        spawn=sim_utils.GlassMdlCfg(glass_color=(0.8, 0.9, 1.0), glass_ior=1.5),
        channels=("color", "roughness", "ior"),
    )
    glass_leg = VisualMaterialCfg(
        prim_path="{ENV_REGEX_NS}/Robot/glass_leg",
        spawn=sim_utils.GlassMdlCfg(glass_color=(0.8, 0.9, 1.0), glass_ior=1.5),
        channels=("color", "roughness", "ior"),
    )
    glass_foot = VisualMaterialCfg(
        prim_path="{ENV_REGEX_NS}/Robot/glass_foot",
        spawn=sim_utils.GlassMdlCfg(glass_color=(0.8, 0.9, 1.0), glass_ior=1.5),
        channels=("color", "roughness", "ior"),
    )
    solid_body = VisualMaterialCfg(
        prim_path="{ENV_REGEX_NS}/Robot/solid_body",
        spawn=sim_utils.PbrMdlCfg(diffuse_color_constant=(0.8, 0.3, 0.1)),
    )
    solid_leg = VisualMaterialCfg(
        prim_path="{ENV_REGEX_NS}/Robot/solid_leg",
        spawn=sim_utils.PbrMdlCfg(diffuse_color_constant=(0.2, 0.2, 0.7)),
    )
    solid_foot = VisualMaterialCfg(
        prim_path="{ENV_REGEX_NS}/Robot/solid_foot",
        spawn=sim_utils.PbrMdlCfg(diffuse_color_constant=(0.1, 0.6, 0.2), reflection_roughness_constant=0.9),
    )

    robot: ArticulationCfg = ANYMAL_C_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=[
                ANYMAL_C_CFG.spawn,
                ANYMAL_C_CFG.spawn.replace(
                    visual_material_bindings=_bindings(
                        "./surface_body",
                        "./surface_leg",
                        "./surface_foot",
                    )
                ),
                ANYMAL_C_CFG.spawn.replace(
                    visual_material_bindings=_bindings(
                        "./glass_body",
                        "./glass_leg",
                        "./glass_foot",
                    )
                ),
                ANYMAL_C_CFG.spawn.replace(
                    visual_material_bindings=_bindings(
                        "./solid_body",
                        "./solid_leg",
                        "./solid_foot",
                    )
                ),
                ANYMAL_C_CFG.spawn.replace(
                    visual_material_bindings=_bindings(
                        "./surface_body",
                        "./solid_leg",
                        "./glass_foot",
                    )
                ),
            ],
            random_choice=False,
        ),
    )

    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0)
    )


@configclass
class ActionsCfg:
    """Hold the robot at its default pose."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], use_default_offset=True)


@configclass
class ObservationsCfg:
    """Minimal policy observation group."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)

        def __post_init__(self):
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Randomize each style's three material assets independently."""

    randomize_surface_style = EventTerm(
        func=mdp.randomize_visual_material,
        mode="reset",
        params={
            "materials": [
                SceneEntityCfg("surface_body"),
                SceneEntityCfg("surface_leg"),
                SceneEntityCfg("surface_foot"),
            ],
            "channels": {
                "color": {"r": (0.05, 1.0), "g": (0.05, 1.0), "b": (0.05, 1.0)},
                "roughness": (0.0, 1.0),
                "metallic": (0.0, 1.0),
                "emissive_color": ((0.0, 0.0, 0.0), (0.25, 0.25, 0.25)),
                "opacity": (0.5, 1.0),
            },
        },
    )
    randomize_glass_style = EventTerm(
        func=mdp.randomize_visual_material,
        mode="reset",
        params={
            "materials": [SceneEntityCfg("glass_body"), SceneEntityCfg("glass_leg"), SceneEntityCfg("glass_foot")],
            "channels": {
                "color": {"r": (0.05, 1.0), "g": (0.05, 1.0), "b": (0.05, 1.0)},
                "roughness": (0.0, 0.8),
                "ior": (1.1, 1.8),
            },
        },
    )
    randomize_solid_style = EventTerm(
        func=mdp.randomize_visual_material,
        mode="reset",
        params={
            "materials": [SceneEntityCfg("solid_body"), SceneEntityCfg("solid_leg"), SceneEntityCfg("solid_foot")],
            "channels": {"color": {"r": (0.05, 1.0), "g": (0.05, 1.0), "b": (0.05, 1.0)}},
        },
    )
    randomize_parts_per_env = EventTerm(
        func=mdp.randomize_visual_shape,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=[f"{leg}_{link}" for leg in _LEGS for link in ("SHANK", "FOOT", "HIP", "THIGH")]
            ),
            "channels": {"color": ((0.05, 0.05, 0.05), (1.0, 1.0, 1.0))},
        },
    )


@configclass
class VisualMaterialEnvCfg(ManagerBasedEnvCfg):
    """Manager-based environment for the visual-material demo."""

    scene: VisualMaterialSceneCfg = VisualMaterialSceneCfg(num_envs=512, env_spacing=1.5)
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.decimation = 4
        self.sim.dt = 0.005


def main() -> None:
    """Launch the selected backends and run the randomization scene."""
    with launch_simulation(cfg=PhysicsCfg(), launcher_args=args_cli) as physics_cfg:
        from isaaclab.envs import ManagerBasedEnv  # noqa: PLC0415

        env_cfg = VisualMaterialEnvCfg()
        env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.sim.device = args_cli.device
        env_cfg.sim.physics = physics_cfg
        visualizers = set(args_cli.visualizer or ())
        newton_only = bool(visualizers) and visualizers <= {"newton_gl", "newton_rtx"}
        if newton_only:
            for name in (
                "surface_body",
                "surface_leg",
                "surface_foot",
                "glass_body",
                "glass_leg",
                "glass_foot",
            ):
                getattr(env_cfg.scene, name).channels = ("color",)
            env_cfg.events.randomize_surface_style.params["channels"] = {
                "color": {"r": (0.05, 1.0), "g": (0.05, 1.0), "b": (0.05, 1.0)}
            }
            env_cfg.events.randomize_glass_style.params["channels"] = {
                "color": {"r": (0.05, 1.0), "g": (0.05, 1.0), "b": (0.05, 1.0)}
            }
        if not newton_only:
            env_cfg.events.randomize_parts_per_env = None

        env = ManagerBasedEnv(cfg=env_cfg)
        actions = torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device)
        count = 0
        env.reset()
        print("[INFO]: Setup complete.")
        while env.sim.is_headless_or_exist_active_visualizer():
            if count > 0 and count % 50 == 0:
                num_reset = int(torch.randint(1, env.num_envs + 1, ()).item())
                env_ids = torch.randperm(env.num_envs, dtype=torch.int32, device=env.device)[:num_reset]
                with Timer(name="visual material reset", time_unit="ms") as timer:
                    env.reset(env_ids=env_ids)
                mean = Timer.timing_info["visual material reset"]["mean"] * 1000.0
                print(
                    f"[INFO]: Randomized {num_reset} environments in {timer.total_run_time * 1000.0:.3f} ms "
                    f"(mean {mean:.3f} ms)."
                )
            env.step(actions)
            count += 1
        env.close()


if __name__ == "__main__":
    main()
