# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Optional domain-randomization event terms for the OpenArm pick-up/stack tasks.

These terms are not attached to any env cfg's `EventCfg` by default. They are opt-in,
attached only when `--enable_domain_randomization` is passed to generate_dataset.py
(see scripts/imitation_learning/isaaclab_mimic/generate_dataset.py), so normal recording
and generation runs are unaffected unless a caller explicitly asks for randomization.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim.utils.stage import get_current_stage

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


# NOTE: every term below is a plain function, deliberately not a `ManagerTermBase` class.
# Class-based "reset"-mode terms (e.g. the core `isaaclab.envs.mdp.events.randomize_visual_color`)
# rely on the manager swapping `term_cfg.func` from the class to an instance the first time the
# simulation starts playing; in a freshly `gym.make(...)`-created env, that swap is driven by a
# deferred scene-entity-resolution callback that can lose the race against the very first
# `env.reset()` call, e.g. in scripts/imitation_learning/isaaclab_mimic/generate_dataset.py. The
# observed failure is `TypeError: randomize_visual_color.__init__() got an unexpected keyword
# argument 'asset_cfg'` -- the manager calling the raw class instead of an instance. Plain
# functions (matching how `franka_stack_events.randomize_visual_texture_material` is already
# used successfully in "reset" mode elsewhere in this task family) never go through that swap.


def randomize_static_asset_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    base_pos: tuple[float, float, float],
    asset_cfg: SceneEntityCfg,
):
    """Jitter a static (non-rigid-body) `AssetBaseCfg` asset's world pose around a known default
    local position, e.g. the workspace pad.

    `AssetBaseCfg`-backed assets are exposed to the scene as an `XformPrimView`, not a
    `RigidObject`/`Articulation` -- they have no `write_root_pose_to_sim`/`default_root_state`,
    so pose changes go through `XformPrimView.set_world_poses` instead, and the pose to jitter
    around must be passed in explicitly as `base_pos` (the asset's env-local spawn position, see
    e.g. `PAD_CENTER_X`/`PAD_HEIGHT` in stack_joint_pos_env_cfg.py) rather than read back from a
    cached default -- reading it back would need a class instance to cache it in, which is
    exactly the pattern this module avoids (see module docstring above).
    """
    asset = env.scene[asset_cfg.name]
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=env.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=env.device)

    base_pos_t = torch.tensor(base_pos, device=env.device, dtype=torch.float32)
    positions = env.scene.env_origins[env_ids] + base_pos_t.unsqueeze(0) + rand_samples[:, 0:3]
    orientations = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])

    asset.set_world_poses(positions, orientations, indices=env_ids.tolist())


def _resolve_preview_surface_shaders(asset, env_ids: torch.Tensor) -> list:
    """Find the `UsdPreviewSurface` Shader prim(s) created by a `PreviewSurfaceCfg` override,
    one per requested env id.

    `spawn_preview_surface` (`isaaclab.sim.spawners.materials.visual_materials`) creates the
    material at a known, deterministic path and always names the shader child `"Shader"`:
    `UsdShade.Material.Define(stage, prim_path)` then `CreateShaderPrimFromSdrCommand(...,
    parent_path=prim_path, name="Shader")` -> shader lives at `f"{prim_path}/Shader"`.

    The material's own path depends on which spawn function bound it (`visual_material_path`
    defaults to `"material"` in both, but relative to a different base):
    - `spawn_from_usd` (`cube_2`, a Nucleus block prop) binds directly on the asset root:
      `f"{asset.cfg.prim_path}/material"` (`from_files.py:_spawn_from_usd_file`).
    - `spawn_cuboid` (`workspace_pad`, a primitive shape) binds on the nested geometry prim,
      never the root: `f"{asset.prim_paths[env_id]}/geometry/material"`
      (`shapes.py:_spawn_geom_from_prim_type`).

    `asset.cfg.prim_path` (used for the first case) is NOT a concrete per-env path -- it's the
    env-regex TEMPLATE (`InteractiveScene` resolves `{ENV_REGEX_NS}` into the literal string
    `/World/envs/env_.*` once at spawn time via `.format(...)`, and never substitutes it down to
    a concrete `env_0`/`env_1`/... after that). Feeding that `"env_.*"` wildcard straight into
    `stage.GetPrimAtPath(...)` produces an "Ill-formed SdfPath" USD warning (`GetPrimAtPath`
    needs a literal path, not a regex) and always resolves to an invalid prim -- silently a
    no-op every time, confirmed in practice: `cube_2`'s color never actually changed. Substitute
    the real env index for each requested `env_id` instead of using the raw template.

    Returns one Shader `Usd.Prim` per `env_id` (or `None` in that slot if it doesn't exist --
    e.g. the asset was never given a `PreviewSurfaceCfg` visual_material override).
    """
    stage = get_current_stage()
    shaders = []
    for env_id in env_ids.tolist():
        if hasattr(asset, "cfg"):
            concrete_prim_path = asset.cfg.prim_path.replace("env_.*", f"env_{env_id}")
            material_path = f"{concrete_prim_path}/material"
        else:
            material_path = f"{asset.prim_paths[env_id]}/geometry/material"
        shader_prim = stage.GetPrimAtPath(f"{material_path}/Shader")
        shaders.append(shader_prim if shader_prim.IsValid() else None)
    return shaders


def randomize_visual_color(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    colors: list[tuple[float, float, float]] | dict[str, tuple[float, float]],
    roughness_range: tuple[float, float] | None = None,
):
    """Randomize an asset's `PreviewSurfaceCfg` material color (and optionally roughness) by
    writing directly to its Shader prim's USD attributes -- independently per env id.

    Earlier versions of this function (and the equivalent core
    `isaaclab.envs.mdp.events.randomize_visual_color`) went through the Replicator API
    (`rep.get.prims` + `rep.randomizer.color`), which creates a *new* OmniGraph SDGPipeline node
    and a *new* OmniPBR material every call. Called once per "reset" across many episodes, that
    leaves behind a growing pile of orphaned nodes -- observed in practice as repeated
    `UsdExpiredPrimAccessError: Used null prim` errors from `OgnSampleOmniPBR` once an older
    node's cached material/shader prim gets replaced by a newer call. Setting the existing
    Shader prim's `inputs:diffuseColor`/`inputs:roughness` attributes directly (the same
    approach `randomize_scene_lighting` already uses for the dome light, which never hit this
    issue) has none of that: no new nodes, no growing pile, nothing to expire.

    `colors` is either a list of `(r, g, b)` tuples to sample from, or a dict
    `{"r": (low, high), "g": (low, high), "b": (low, high)}` to sample a uniform range from.
    """
    asset = env.scene[asset_cfg.name]
    for shader_prim in _resolve_preview_surface_shaders(asset, env_ids):
        if shader_prim is None:
            continue

        if isinstance(colors, dict):
            r = random.uniform(*colors["r"])
            g = random.uniform(*colors["g"])
            b = random.uniform(*colors["b"])
        else:
            r, g, b = random.choice(list(colors))
        shader_prim.GetAttribute("inputs:diffuseColor").Set((r, g, b))

        if roughness_range is not None:
            shader_prim.GetAttribute("inputs:roughness").Set(random.uniform(*roughness_range))


def randomize_fixed_camera_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pos_range: dict[str, tuple[float, float]],
    look_at_offset: tuple[float, float, float] = (0.45, 0.0, 0.15),
    look_at_range: dict[str, tuple[float, float]] | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("front_cam"),
):
    """Jitter a fixed, env-anchored camera's world position AND, via `look_at_range`, the point
    it's aimed at -- so the viewing angle actually varies, not just the eye position with a
    constant gaze direction.

    Only appropriate for cameras whose prim is anchored directly under the env root (like
    `front_cam`) -- cameras mounted on a moving robot link use `randomize_mounted_camera_pose`
    instead, since their nominal pose is relative to a moving parent link, not the env origin.
    """
    camera = env.scene[asset_cfg.name]
    range_list = [pos_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device=env.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device=env.device)

    base_offset = torch.tensor(camera.cfg.offset.pos, device=env.device, dtype=torch.float32)
    env_origins = env.scene.env_origins[env_ids]
    eyes = env_origins + base_offset.unsqueeze(0) + rand_samples

    look_at_range = look_at_range or {}
    target_range_list = [look_at_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    target_ranges = torch.tensor(target_range_list, device=env.device)
    target_jitter = math_utils.sample_uniform(
        target_ranges[:, 0], target_ranges[:, 1], (len(env_ids), 3), device=env.device
    )
    targets = env_origins + torch.tensor(look_at_offset, device=env.device, dtype=torch.float32).unsqueeze(
        0
    ) + target_jitter

    camera.set_world_poses_from_view(eyes, targets, env_ids=env_ids)


def randomize_mounted_camera_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pos_range: dict[str, tuple[float, float]],
    rot_range: dict[str, tuple[float, float]],
    parent_body_name: str,
    asset_cfg: SceneEntityCfg,
):
    """Jitter a robot-mounted camera's effective mount offset (position + small rotation)
    relative to its parent link, e.g. `wrist_cam`/`right_wrist_cam`/`body_cam`.

    Unlike `front_cam`, these cameras' nominal pose is a LOCAL offset relative to a moving robot
    link (`CameraCfg.offset`), not a world-frame constant -- so we can't just add jitter to
    `env_origins` like `randomize_fixed_camera_pose` does. Instead:

    1. Read the parent link's CURRENT world pose from the robot's `Articulation` data
       (`body_pos_w`/`body_quat_w`). This is written synchronously by the physics reset, unlike
       the camera sensor's own cached `data.pos_w`/`quat_w_world`, which can still be lagging
       behind a just-reset robot pose (see `num_rerenders_on_reset` elsewhere in this task
       family -- it exists precisely because camera data doesn't refresh immediately on reset).
    2. Convert the camera's configured local offset+rotation (`camera.cfg.offset`, which may be
       authored in "ros"/"opengl"/"world" convention) into the "world"-convention quaternion
       `body_quat_w` is already expressed in, via `convert_camera_frame_orientation_convention`.
    3. Add a small random position/rotation delta to that local offset, compose it onto the
       parent's current world pose, and write it with `Camera.set_world_poses(..., convention="world")`.

    `Camera.set_world_poses` -> `XformPrimView.set_world_poses` converts the given world pose into
    a LOCAL xformOp offset relative to the parent's world pose *at call time*, then that offset
    rides along rigidly with the parent for the rest of the episode (see
    `isaaclab.sim.views.xform_prim_view._set_world_poses_usd`) -- it does not fight physics, so
    this is safe to call once per reset.
    """
    camera = env.scene[asset_cfg.name]
    robot = env.scene["robot"]
    body_ids, _ = robot.find_bodies(parent_body_name)
    body_id = body_ids[0]

    link_pos_w = robot.data.body_pos_w[env_ids, body_id]
    link_quat_w = robot.data.body_quat_w[env_ids, body_id]

    offset_pos = torch.tensor(camera.cfg.offset.pos, device=env.device, dtype=torch.float32)
    offset_quat_native = torch.tensor(camera.cfg.offset.rot, device=env.device, dtype=torch.float32).unsqueeze(0)
    offset_quat_world = math_utils.convert_camera_frame_orientation_convention(
        offset_quat_native, origin=camera.cfg.offset.convention, target="world"
    ).squeeze(0)

    pos_ranges = torch.tensor([pos_range.get(k, (0.0, 0.0)) for k in ("x", "y", "z")], device=env.device)
    pos_delta = math_utils.sample_uniform(pos_ranges[:, 0], pos_ranges[:, 1], (len(env_ids), 3), device=env.device)
    rot_ranges = torch.tensor([rot_range.get(k, (0.0, 0.0)) for k in ("roll", "pitch", "yaw")], device=env.device)
    rot_delta = math_utils.sample_uniform(rot_ranges[:, 0], rot_ranges[:, 1], (len(env_ids), 3), device=env.device)
    delta_quat = math_utils.quat_from_euler_xyz(rot_delta[:, 0], rot_delta[:, 1], rot_delta[:, 2])

    local_pos = offset_pos.unsqueeze(0) + pos_delta
    local_quat = math_utils.quat_mul(offset_quat_world.unsqueeze(0).expand(len(env_ids), -1), delta_quat)

    world_pos = link_pos_w + math_utils.quat_apply(link_quat_w, local_pos)
    world_quat = math_utils.quat_mul(link_quat_w, local_quat)

    camera.set_world_poses(world_pos, world_quat, env_ids=env_ids, convention="world")


def randomize_scene_lighting(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    intensity_range: tuple[float, float],
    color_range: tuple[float, float] = (0.6, 1.0),
    skybox_textures: list[str] | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("light"),
):
    """Randomize the scene dome light's intensity, grayscale brightness, and (if
    `skybox_textures` is given) background HDR skybox.

    The dome light prim (`/World/light`) is spawned once outside the per-env namespace and
    shared by every cloned env, so there is exactly one prim to randomize regardless of
    `num_envs` -- this samples a single draw per call, not one per env id.
    """
    light_asset = env.scene[asset_cfg.name]
    light_prim = light_asset.prims[0]

    intensity = random.uniform(*intensity_range)
    light_prim.GetAttribute("inputs:intensity").Set(intensity)

    gray = random.uniform(*color_range)
    light_prim.GetAttribute("inputs:color").Set((gray, gray, gray))

    if skybox_textures:
        texture_file_attr = light_prim.GetAttribute("inputs:texture:file")
        texture_file_attr.Set(random.choice(skybox_textures))
