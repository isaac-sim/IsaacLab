# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared scene + assertions for the per-renderer visual-color contract tests.

Not named ``test_*`` so pytest does not collect it. The per-renderer files
(``test_color_randomization_contract_{rtx,newton,ovrtx}.py``) import :func:`assert_color_contract`
and call it with their renderer/physics cfgs; renderers cannot share a process, so each backend lives
in its own file (mirroring ``test_camera_ppisp_gaussian_{newton,ovrtx}``).

The scene loads a small bundled-OmniPBR asset (``_data/bundled_omnipbr_cube.usda``) via ``UsdFileCfg``:
two cubes per env (``cube_a`` right, ``cube_b`` left) + a dome light + a per-env camera. Each cube
ships its own OmniPBR material with a direct ``material:binding``, mirroring stock assets (cartpole,
dexsuite); this exercises the OVRTX pre-bake unbind path that a fresh-PreviewSurface scene misses.

The per-cube ``randomize_visual_color`` writers are driven with a fixed *saturated* palette and
the rendered pixels are read back and classified (nearest-palette, robust to lighting /
tonemapping). The isaac_rtx file launches Kit (``AppLauncher``) before importing this module; the
kit-less renderers (newton_warp / ovrtx) need no launch.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils.configclass import configclass

# Bundled-OmniPBR USDA asset that ships with a direct material:binding on the Cube prim. The path
# is resolved relative to this module so the test data lives next to the contract code.
_BUNDLED_CUBE_USD = str(Path(__file__).parent / "_data" / "bundled_omnipbr_cube.usda")

# Saturated palette (sRGB 0-1), one entry per (env, cube). Saturated primaries survive lighting +
# tonemapping so the rendered pixel still classifies to the right entry.
_PALETTE = {
    "red": (1.0, 0.0, 0.0),
    "green": (0.0, 1.0, 0.0),
    "blue": (0.0, 0.0, 1.0),
    "yellow": (1.0, 1.0, 0.0),
    "magenta": (1.0, 0.0, 1.0),
}
# cube_a is sampled at the right of the view, cube_b at the left (see _SceneCfg layout).
_RIGHT, _LEFT = 0.75, 0.25


def _cube(prim_suffix: str, y: float) -> RigidObjectCfg:
    """Cube spawned from the bundled-OmniPBR test asset (ships a direct ``material:binding`` like stock
    assets). Rooted with rigid-body + collision APIs so ``RigidObjectCfg`` spawns it cleanly.
    """
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/" + prim_suffix,
        spawn=sim_utils.UsdFileCfg(
            usd_path=_BUNDLED_CUBE_USD,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, y, 0.5)),
    )


def _color_term(asset: str, event_name: str) -> EventTerm:
    return EventTerm(
        func=mdp.randomize_visual_color,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(asset),
            # Target the renderable Mesh prim under the bundled asset's ``visuals`` scope. The
            # bundled USDA authors the Mesh at ``<asset_prim_path>/visuals/mesh`` (one per env).
            "mesh_name": "/visuals/mesh",
            "colors": {"r": (0.0, 1.0), "g": (0.0, 1.0), "b": (0.0, 1.0)},
            "event_name": event_name,
        },
    )


@configclass
class _SceneCfg(InteractiveSceneCfg):
    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=600.0))
    cube_a = _cube("cube_a", y=-0.8)  # projects to the right of the camera view
    cube_b = _cube("cube_b", y=+0.8)  # projects to the left
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        offset=CameraCfg.OffsetCfg(pos=(-2.5, 0.0, 0.5), rot=(0.0, 0.0, 0.0, 1.0), convention="world"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, horizontal_aperture=20.955, clipping_range=(0.1, 100.0)),
        width=256,
        height=256,
    )


@configclass
class _ActionsCfg:
    pass  # no articulation -> no action terms


@configclass
class _ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        cube_pos = ObsTerm(func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("cube_a")})

        def __post_init__(self):
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class _EventCfg:
    cube_a_color = _color_term("cube_a", "cube_a_color")
    cube_b_color = _color_term("cube_b", "cube_b_color")


@configclass
class _ContractEnvCfg(ManagerBasedEnvCfg):
    scene = _SceneCfg(num_envs=2, env_spacing=8.0, replicate_physics=False)
    actions = _ActionsCfg()
    observations = _ObservationsCfg()
    events = _EventCfg()

    def __post_init__(self):
        self.decimation = 1
        self.sim.dt = 0.01


def _classify(pixel: np.ndarray) -> str:
    """Classify a pixel to the nearest palette entry by hue (cosine similarity).

    Direction-based, not Euclidean: brightness-invariant, so it is robust to the large exposure
    differences between renderers (kit-less Newton renders these saturated colors far dimmer than RTX,
    e.g. a dark ``[31, 31, 0]`` is still unmistakably *yellow* by direction).
    """
    vec = pixel.astype(np.float32)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-6:
        return "black"
    vec = vec / norm
    return max(_PALETTE, key=lambda k: float(np.dot(vec, np.asarray(_PALETTE[k]) / np.linalg.norm(_PALETTE[k]))))


def _palette_tensor(keys, device) -> torch.Tensor:
    return torch.tensor([_PALETTE[k] for k in keys], dtype=torch.float32, device=device)


def _render_rgb(env: ManagerBasedEnv) -> np.ndarray:
    for _ in range(3):
        env.sim.step()
    camera = env.scene["camera"]
    camera.update(env.sim.cfg.dt)
    rgb = camera.data.output["rgb"]
    if not isinstance(rgb, torch.Tensor):
        rgb = rgb.torch
    return rgb.detach().cpu().numpy()


def _sample(rgb: np.ndarray, env_id: int, frac_x: float) -> np.ndarray:
    h, w = rgb.shape[1], rgb.shape[2]
    cx, cy, r = int(w * frac_x), h // 2, 3
    return rgb[env_id, cy - r : cy + r + 1, cx - r : cx + r + 1, :3].reshape(-1, 3).mean(0)


def assert_color_contract(renderer_cfg, physics_cfg=None) -> None:
    """Render the 2-cube scene on the given backend and assert the visual-color contract.

    Args:
        renderer_cfg: The camera renderer cfg for the backend under test (e.g. ``IsaacRtxRendererCfg``,
            ``NewtonWarpRendererCfg``, ``OVRTXRendererCfg``).
        physics_cfg: The physics cfg for the backend (``NewtonCfg`` for the kit-less renderers); ``None``
            keeps the PhysX default (for ``isaac_rtx``).

    Asserts color-lands, per-prim distinctness, per-env distinctness, and an ``env_ids`` subset write.
    """
    cfg = _ContractEnvCfg()
    cfg.scene.camera.renderer_cfg = renderer_cfg
    if physics_cfg is not None:
        cfg.sim.physics = physics_cfg

    # The caller launches Kit (via AppLauncher) before importing this module for the isaac_rtx renderer;
    # the kit-less renderers (newton_warp / ovrtx) need no launch. See the per-renderer test files.
    env = ManagerBasedEnv(cfg=cfg)
    try:
        env.reset()
        device = env.device
        writer_a = env.event_manager.get_term_cfg("cube_a_color").func._writer
        writer_b = env.event_manager.get_term_cfg("cube_b_color").func._writer
        assert writer_a.num_targets == 2 and writer_b.num_targets == 2, (writer_a.num_targets, writer_b.num_targets)

        # fixed palette: env0 -> {cube_a red, cube_b green}; env1 -> {cube_a blue, cube_b yellow}
        all_envs = torch.arange(2, device=device)
        writer_a.write_colors(all_envs, _palette_tensor(["red", "blue"], device))
        writer_b.write_colors(all_envs, _palette_tensor(["green", "yellow"], device))
        rgb = _render_rgb(env)

        env0_a, env0_b = _classify(_sample(rgb, 0, _RIGHT)), _classify(_sample(rgb, 0, _LEFT))
        env1_a, env1_b = _classify(_sample(rgb, 1, _RIGHT)), _classify(_sample(rgb, 1, _LEFT))

        # color-lands: each cube renders its written color
        assert (env0_a, env0_b) == ("red", "green"), f"env0 got ({env0_a}, {env0_b})"
        assert (env1_a, env1_b) == ("blue", "yellow"), f"env1 got ({env1_a}, {env1_b})"
        # per-prim (cubes within an env differ) and per-env (same cube differs across envs)
        assert env0_a != env0_b and env1_a != env1_b, "per-prim distinctness failed"
        assert env0_a != env1_a and env0_b != env1_b, "per-env distinctness failed"

        # env_ids subset: recolor only env 1's cube_a (-> magenta); env 0 must be untouched
        writer_a.write_colors(torch.tensor([1], device=device), _palette_tensor(["red", "magenta"], device))
        rgb = _render_rgb(env)
        assert _classify(_sample(rgb, 0, _RIGHT)) == "red", "env_ids subset leaked into env 0"
        assert _classify(_sample(rgb, 1, _RIGHT)) == "magenta", "env_ids subset did not recolor env 1"
    finally:
        env.close()
