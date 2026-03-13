# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from .eigenbot_env_cfg import EigenbotEnvCfg

# Path to USDZ file and meshes directory for texture overwriting
_ASSETS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "assets")
_USDZ_PATH = os.path.join(_ASSETS_DIR, "eigenbot_new.usdz")
_MESHES_DIR = os.path.join(_ASSETS_DIR, "eigenbot", "meshes")


class EigenbotEnv(DirectRLEnv):
    """Minimal environment for rendering and basic control of the Eigenbot.

    This is a bare-minimum port that spawns the eigenbot and allows sending
    joint position commands. The reward/termination logic is placeholder and
    should be replaced with a meaningful task.
    """

    cfg: EigenbotEnvCfg

    def __init__(self, cfg: EigenbotEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel
        self.num_joints = self.joint_pos.shape[1]

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # apply USDZ textures before cloning so materials propagate to all envs
        self._apply_usdz_textures()
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _apply_usdz_textures(self):
        """Overwrite robot visual materials using colors extracted from the USDZ asset.

        Opens the USDZ file, reads diffuse colors from its materials, and creates
        solid-color PreviewSurface materials on the sim stage bound to the URDF robot.
        STL meshes lack UV coordinates, so solid colors are used instead of textures.
        Falls back to hardcoded colors if the USDZ cannot be opened.
        """
        from pxr import Usd, UsdShade, UsdGeom, Sdf, Gf

        import omni.usd

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            print("[WARN] No USD stage available for texture overwriting")
            return

        # Try to extract diffuse colors from USDZ materials
        link_colors = self._extract_colors_from_usdz()

        # Create materials scope
        materials_root = Sdf.Path("/World/Materials")
        if not stage.GetPrimAtPath(materials_root).IsValid():
            stage.DefinePrim(materials_root, "Scope")

        # Create a PreviewSurface material for each link type
        created_materials = {}
        for prefix, (r, g, b, roughness, metallic) in link_colors.items():
            mat_prim_path = materials_root.AppendChild(prefix + "_material")
            if not stage.GetPrimAtPath(mat_prim_path).IsValid():
                material = UsdShade.Material.Define(stage, mat_prim_path)
                shader = UsdShade.Shader.Define(
                    stage, mat_prim_path.AppendChild("Shader")
                )
                shader.CreateIdAttr("UsdPreviewSurface")
                shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
                    Gf.Vec3f(r, g, b)
                )
                shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
                shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
                shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
                material.CreateSurfaceOutput().ConnectToSource(
                    shader.GetOutput("surface")
                )
            created_materials[prefix] = mat_prim_path

        # Bind materials to robot link prims in env_0 (propagates via cloning)
        robot_prim = stage.GetPrimAtPath("/World/envs/env_0/Robot")
        if not robot_prim.IsValid():
            print("[WARN] Robot prim not found at /World/envs/env_0/Robot")
            return

        applied_count = 0
        for prim in Usd.PrimRange(robot_prim):
            # Bind at the link level (parent of visuals/collisions), not on
            # instanced children, to avoid the instanced-prim modification error
            prim_name = prim.GetName()
            matched_prefix = None
            for prefix in created_materials:
                if prim_name.startswith(prefix + "_"):
                    matched_prefix = prefix
                    break
            if matched_prefix is None:
                continue

            mat_path = created_materials[matched_prefix]
            binding_api = UsdShade.MaterialBindingAPI.Apply(prim)
            mat = UsdShade.Material(stage.GetPrimAtPath(mat_path))
            binding_api.Bind(mat)
            applied_count += 1

        print(f"[INFO] Applied {applied_count} material bindings to robot links")

    def _extract_colors_from_usdz(self) -> dict:
        """Read diffuse colors from the USDZ materials and map them to link prefixes.

        Returns a dict of link_prefix -> (r, g, b, roughness, metallic).
        Falls back to hardcoded colors if USDZ is unavailable.
        """
        # Hardcoded defaults that approximate the real Eigenbot appearance
        defaults = {
            "eigenbody":    (0.10, 0.10, 0.10, 0.6, 0.0),   # dark grey body
            "bendy_input":  (0.12, 0.12, 0.12, 0.5, 0.0),   # dark module input
            "bendy_output": (0.15, 0.15, 0.15, 0.5, 0.0),   # slightly lighter output
            "static_elbow": (0.55, 0.55, 0.55, 0.3, 0.4),   # metallic silver elbow
            "foot_input":   (0.20, 0.20, 0.20, 0.8, 0.0),   # rubber-like dark foot
        }

        if not os.path.isfile(_USDZ_PATH):
            print("[INFO] USDZ file not found, using default colors")
            return defaults

        try:
            from pxr import Usd, UsdShade, UsdGeom, Sdf, Gf

            usdz_stage = Usd.Stage.Open(_USDZ_PATH)
            if not usdz_stage:
                return defaults

            # Collect mesh -> bound material from USDZ
            mesh_to_color = {}
            for prim in usdz_stage.Traverse():
                if not (prim.IsA(UsdGeom.Mesh) or prim.IsA(UsdGeom.Subset)):
                    continue
                binding_api = UsdShade.MaterialBindingAPI(prim)
                bound_mat, _ = binding_api.ComputeBoundMaterial()
                if not bound_mat:
                    continue

                # Try to read diffuse color from the shader
                color = self._read_shader_diffuse(bound_mat)
                if color is not None:
                    mesh_to_color[prim.GetName().lower()] = color

            if not mesh_to_color:
                print("[INFO] No diffuse colors found in USDZ, using defaults")
                return defaults

            # Map USDZ mesh names to URDF link prefixes
            result = dict(defaults)  # start with defaults
            for mesh_name, (r, g, b) in mesh_to_color.items():
                name = mesh_name.lower()
                if "eigenbody" in name or "eigenbot_base" in name or "body" in name:
                    result["eigenbody"] = (r, g, b, 0.6, 0.0)
                elif "bendyin" in name or "bendy_in" in name or "bendy_input" in name:
                    result["bendy_input"] = (r, g, b, 0.5, 0.0)
                elif "bendyout" in name or "bendy_out" in name or "bendy_output" in name:
                    result["bendy_output"] = (r, g, b, 0.5, 0.0)
                elif "elbow" in name:
                    result["static_elbow"] = (r, g, b, 0.3, 0.4)
                elif "foot" in name:
                    result["foot_input"] = (r, g, b, 0.8, 0.0)

            print(f"[INFO] Extracted {len(mesh_to_color)} colors from USDZ")
            return result

        except Exception as e:
            print(f"[WARN] Failed to extract colors from USDZ: {e}")
            return defaults

    @staticmethod
    def _read_shader_diffuse(material):
        """Read the diffuse color from a UsdShade.Material's shader network.

        Returns (r, g, b) tuple or None.
        """
        from pxr import UsdShade, Sdf

        # Walk the shader outputs to find a UsdPreviewSurface
        surface_output = material.GetSurfaceOutput()
        if not surface_output:
            return None

        connected_source = surface_output.GetConnectedSources()
        if not connected_source or not connected_source[0]:
            return None

        for source_info in connected_source[0]:
            shader_prim = source_info.source.GetPrim()
            shader = UsdShade.Shader(shader_prim)
            if not shader:
                continue

            # Try diffuseColor input
            diffuse_input = shader.GetInput("diffuseColor")
            if diffuse_input and diffuse_input.HasValue():
                val = diffuse_input.Get()
                if val is not None:
                    # Could be Gf.Vec3f or tuple
                    try:
                        return (float(val[0]), float(val[1]), float(val[2]))
                    except (TypeError, IndexError):
                        pass

        return None

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        # Position control: target = action_scale * action + default_joint_pos
        targets = self.cfg.action_scale * self.actions + self.robot.data.default_joint_pos
        self.robot.set_joint_position_target(targets)

    def _get_observations(self) -> dict:
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel
        obs = torch.cat((self.joint_pos, self.joint_vel), dim=-1)
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        # Placeholder reward: alive bonus only
        return torch.ones(self.num_envs, device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]

        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
