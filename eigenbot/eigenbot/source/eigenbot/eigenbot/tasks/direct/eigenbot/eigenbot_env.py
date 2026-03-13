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
        """Overwrite robot visual materials with textures extracted from the USDZ asset.

        Opens the USDZ file, extracts material-to-mesh bindings, copies those materials
        into the simulation stage, and binds them to the corresponding URDF-converted prims.
        Falls back to creating materials from the PNG textures in the meshes directory if
        USDZ material extraction fails.
        """
        from pxr import Usd, UsdShade, UsdGeom, Sdf

        import omni.usd

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            print("[WARN] No USD stage available for texture overwriting")
            return

        # --- Try USDZ-based material transfer first ---
        usdz_materials_applied = False
        if os.path.isfile(_USDZ_PATH):
            try:
                usdz_materials_applied = self._apply_materials_from_usdz(stage)
            except Exception as e:
                print(f"[WARN] USDZ material transfer failed: {e}, falling back to PNG textures")

        # --- Fallback: create materials from PNG textures ---
        if not usdz_materials_applied:
            self._apply_materials_from_pngs(stage)

    def _apply_materials_from_usdz(self, stage) -> bool:
        """Extract materials from USDZ and apply them to the robot in the simulation stage.

        Returns True if materials were successfully applied.
        """
        from pxr import Usd, UsdShade, UsdGeom, Sdf

        usdz_stage = Usd.Stage.Open(_USDZ_PATH)
        if not usdz_stage:
            return False

        # Collect all materials from USDZ stage
        usdz_materials = {}
        for prim in usdz_stage.Traverse():
            if prim.IsA(UsdShade.Material):
                usdz_materials[prim.GetName()] = prim.GetPath()

        if not usdz_materials:
            return False

        # Collect mesh -> material bindings from USDZ
        mesh_material_map = {}  # mesh_name_lower -> material_path_in_usdz
        for prim in usdz_stage.Traverse():
            if prim.IsA(UsdGeom.Mesh) or prim.IsA(UsdGeom.Subset):
                binding_api = UsdShade.MaterialBindingAPI(prim)
                bound_mat, _ = binding_api.ComputeBoundMaterial()
                if bound_mat:
                    mesh_material_map[prim.GetName().lower()] = bound_mat.GetPath()

        # Copy materials from USDZ stage to simulation stage
        materials_root = Sdf.Path("/World/Materials")
        if not stage.GetPrimAtPath(materials_root).IsValid():
            stage.DefinePrim(materials_root, "Scope")

        copied_materials = set()
        for mat_name, mat_path in usdz_materials.items():
            dest_path = materials_root.AppendChild(mat_name)
            if not stage.GetPrimAtPath(dest_path).IsValid():
                # Copy the material spec from USDZ layer to sim layer
                success = Sdf.CopySpec(
                    usdz_stage.GetRootLayer(), mat_path,
                    stage.GetRootLayer(), dest_path,
                )
                if success:
                    copied_materials.add(mat_name)

        if not copied_materials:
            return False

        # Build mapping: link name prefix -> material dest path
        # URDF link names: eigenbody_*, bendy_input_*, bendy_output_*, static_elbow_*, foot_input_*
        link_to_material = self._build_link_material_mapping(
            mesh_material_map, materials_root, usdz_stage
        )

        # Apply materials to robot prims in env_0 (will be cloned to other envs)
        robot_prim = stage.GetPrimAtPath("/World/envs/env_0/Robot")
        if not robot_prim.IsValid():
            print("[WARN] Robot prim not found at /World/envs/env_0/Robot")
            return False

        applied_count = 0
        for prim in Usd.PrimRange(robot_prim):
            if not (prim.IsA(UsdGeom.Mesh) or prim.IsA(UsdGeom.Subset)):
                continue

            # Walk up to find the link-level prim name
            link_name = self._find_link_name(prim)
            if not link_name:
                continue

            # Match link name to material
            mat_path = self._match_link_to_material(link_name, link_to_material)
            if mat_path and stage.GetPrimAtPath(mat_path).IsValid():
                binding_api = UsdShade.MaterialBindingAPI.Apply(prim)
                mat = UsdShade.Material(stage.GetPrimAtPath(mat_path))
                binding_api.Bind(mat)
                applied_count += 1

        print(f"[INFO] Applied {applied_count} USDZ material bindings to robot meshes")
        return applied_count > 0

    def _build_link_material_mapping(self, mesh_material_map, materials_root, usdz_stage):
        """Build a mapping from URDF link name prefixes to material paths in sim stage."""
        from pxr import Sdf

        link_to_material = {}

        # Try to auto-map based on mesh names in the USDZ
        # Common patterns: mesh names contain 'eigenbody', 'bendy', 'elbow', 'foot'
        for mesh_name, mat_path in mesh_material_map.items():
            mat_name = mat_path.name
            dest_mat = materials_root.AppendChild(mat_name)

            name = mesh_name.lower()
            if "eigenbody" in name or "eigenbot_base" in name or "body" in name:
                link_to_material["eigenbody"] = dest_mat
            elif "bendyin" in name or "bendy_in" in name or "bendy_input" in name:
                link_to_material["bendy_input"] = dest_mat
            elif "bendyout" in name or "bendy_out" in name or "bendy_output" in name:
                link_to_material["bendy_output"] = dest_mat
            elif "elbow" in name:
                link_to_material["static_elbow"] = dest_mat
            elif "foot" in name:
                link_to_material["foot_input"] = dest_mat

        return link_to_material

    def _find_link_name(self, prim):
        """Walk up the prim hierarchy to find the URDF link name."""
        current = prim
        while current and current.GetPath().pathString != "/":
            name = current.GetName()
            # URDF link names follow these patterns
            if any(name.startswith(p) for p in (
                "eigenbody_", "bendy_input_", "bendy_output_",
                "static_elbow_", "foot_input_",
            )):
                return name
            current = current.GetParent()
        return None

    def _match_link_to_material(self, link_name, link_to_material):
        """Match a URDF link name to a material path."""
        from pxr import Sdf

        for prefix, mat_path in link_to_material.items():
            if link_name.startswith(prefix):
                return mat_path
        return None

    def _apply_materials_from_pngs(self, stage):
        """Fallback: create PBR materials from PNG textures and apply to robot meshes."""
        from pxr import Usd, UsdShade, UsdGeom, Sdf, Gf

        # Mapping: link name prefix -> PNG texture filename
        texture_map = {
            "eigenbody": "Eigenbody.png",
            "bendy_input": "Bendy_In.png",
            "bendy_output": "Bendy_Out.png",
            "static_elbow": "Elbow.png",
            "foot_input": "Foot.png",
        }

        materials_root = Sdf.Path("/World/Materials")
        if not stage.GetPrimAtPath(materials_root).IsValid():
            stage.DefinePrim(materials_root, "Scope")

        # Create a PreviewSurface material for each texture
        created_materials = {}
        for prefix, tex_file in texture_map.items():
            tex_path = os.path.join(_MESHES_DIR, tex_file)
            if not os.path.isfile(tex_path):
                print(f"[WARN] Texture not found: {tex_path}")
                continue

            mat_prim_path = materials_root.AppendChild(prefix + "_material")
            if stage.GetPrimAtPath(mat_prim_path).IsValid():
                created_materials[prefix] = mat_prim_path
                continue

            # Create material
            material = UsdShade.Material.Define(stage, mat_prim_path)
            shader_path = mat_prim_path.AppendChild("Shader")
            shader = UsdShade.Shader.Define(stage, shader_path)
            shader.CreateIdAttr("UsdPreviewSurface")

            # Create texture reader for diffuse color
            tex_reader_path = mat_prim_path.AppendChild("DiffuseTexture")
            tex_reader = UsdShade.Shader.Define(stage, tex_reader_path)
            tex_reader.CreateIdAttr("UsdUVTexture")
            # Use forward-slash path for USD compatibility
            tex_reader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
                tex_path.replace("\\", "/")
            )
            tex_reader.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
            tex_reader.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
            tex_reader.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)

            # Create UV coordinate reader
            uv_reader_path = mat_prim_path.AppendChild("UVReader")
            uv_reader = UsdShade.Shader.Define(stage, uv_reader_path)
            uv_reader.CreateIdAttr("UsdPrimvarReader_float2")
            uv_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
            uv_reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)

            # Connect UV reader to texture
            tex_reader.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
                uv_reader.GetOutput("result")
            )

            # Connect texture to shader diffuse color
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
                tex_reader.GetOutput("rgb")
            )
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
            shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
            shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)

            # Wire shader to material surface output
            material.CreateSurfaceOutput().ConnectToSource(shader.GetOutput("surface"))

            created_materials[prefix] = mat_prim_path

        # Apply materials to robot meshes in env_0
        robot_prim = stage.GetPrimAtPath("/World/envs/env_0/Robot")
        if not robot_prim.IsValid():
            print("[WARN] Robot prim not found at /World/envs/env_0/Robot")
            return

        applied_count = 0
        for prim in Usd.PrimRange(robot_prim):
            if not prim.IsA(UsdGeom.Mesh):
                continue

            link_name = self._find_link_name(prim)
            if not link_name:
                continue

            for prefix, mat_path in created_materials.items():
                if link_name.startswith(prefix):
                    binding_api = UsdShade.MaterialBindingAPI.Apply(prim)
                    mat = UsdShade.Material(stage.GetPrimAtPath(mat_path))
                    binding_api.Bind(mat)
                    applied_count += 1
                    break

        print(f"[INFO] Applied {applied_count} PNG-based material bindings to robot meshes")

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
