# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Golden runner for one bundled rendering-scene case."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import numpy as np
import torch
import warp as wp
from golden_image import camera_output_image, compare_to_golden
from isaaclab_newton.physics import reset_mpm_solver_state
from isaaclab_newton.renderers import NewtonWarpRendererCfg
from isaaclab_ov.renderers import OVRTXRendererCfg
from isaaclab_physx.renderers import IsaacRtxRendererCfg
from newton import StateFlags
from rendering_cases import RenderCase
from rendering_runtime import build_rendering_scene
from rendering_scene_cfgs import make_rendering_scene_spec

from isaaclab.assets import VisualMaterial
from isaaclab.renderers.output_contract import RenderBufferKind
from isaaclab.utils.seed import configure_seed

_GOLDEN_ROOT = Path(__file__).parent / "golden_images" / "renderers"
_ARTIFACT_DIR = Path.cwd() / "tests" / "comparison-images" / "images"
_NO_SSIM = {
    RenderBufferKind.DEPTH,
    RenderBufferKind.DISTANCE_TO_CAMERA,
    RenderBufferKind.DISTANCE_TO_IMAGE_PLANE,
    RenderBufferKind.INSTANCE_SEGMENTATION,
    RenderBufferKind.INSTANCE_ID_SEGMENTATION_FAST,
    RenderBufferKind.MOTION_VECTORS,
}
_ALPHA_ONLY_AOVS = {RenderBufferKind.INSTANCE_SEGMENTATION, RenderBufferKind.INSTANCE_ID_SEGMENTATION_FAST}
# Class colors in the synthetic scene's references; OVRTX's runtime IDs may arrive in another order.
_OVRTX_SEMANTIC_COLORS = {
    "ground": (27, 186, 239, 255),
    "robot": (240, 4, 111, 255),
    "moving_cube": (33, 243, 3, 255),
    "table": (220, 179, 62, 255),
    "cylinder": (97, 5, 219, 255),
    "sphere": (15, 238, 97, 255),
}


def run_rendering_case(
    case: RenderCase,
    request: Any,
    *,
    golden_namespace: str | None = "kit",
    artifact_namespace: str | None = None,
) -> None:
    """Capture the declared scene state and all compatible camera outputs."""
    configure_seed(42, torch_deterministic=True)
    scene = make_rendering_scene_spec(case.scene, case.physics)
    scene.cfg.camera.renderer_cfg = {
        "isaac_rtx": IsaacRtxRendererCfg(),
        "newton_warp": NewtonWarpRendererCfg(),
        "ovrtx": OVRTXRendererCfg(async_rendering=case.variant == "async"),
    }[case.renderer]
    if case.scene == "rendering_scene":
        # Explicit colors and shadows belong to this synthetic fixture, not every scene.
        if case.renderer == "ovrtx":
            scene.cfg.camera.renderer_cfg.enable_shadows = True
        else:
            scene.cfg.camera.renderer_cfg.semantic_filter = ["class"]
            scene.cfg.camera.renderer_cfg.semantic_segmentation_mapping = {
                "class:ground": (94, 112, 131, 255),
                "class:robot": (246, 184, 59, 255),
                "class:moving_cube": (38, 141, 242, 255),
                "class:table": (153, 74, 33, 255),
                "class:cylinder": (239, 62, 54, 255),
                "class:sphere": (65, 201, 87, 255),
            }
    with build_rendering_scene(
        scene.cfg,
        case.physics,
        data_types=case.aovs,
        background_color=case.background_color,
        physics_cfg=scene.physics_cfg,
        dt=scene.dt,
    ) as runtime:
        if case.scene == "shadow_hand":
            # A static image tests the declared pose, not contacts during PhysX's initialization steps.
            cube = runtime.scene["object"]
            pose = cube.data.default_root_pose.torch.clone()
            pose[:, :3] += runtime.scene.env_origins
            cube.write_root_pose_to_sim_index(root_pose=pose)
            runtime.sim.forward()
        if case.scene == "kuka_materials":
            robot = runtime.scene["robot"]
            robot.write_joint_position_to_sim_index(position=robot.data.default_joint_pos.torch)
            robot.write_joint_velocity_to_sim_index(velocity=robot.data.default_joint_vel.torch)
            robot.reset()
        if case.scene == "mpm":
            # Fixed reset inputs captured from the original 8c9b995 test; no runtime IK or RL managers.
            with np.load(_GOLDEN_ROOT / "mpm/reset_state.npz") as state:
                robot, media = runtime.scene["robot"], runtime.scene["media"]
                assert robot.joint_names == state["joint_names"].tolist()
                robot.write_root_pose_to_sim_index(
                    root_pose=torch.tensor(state["root_pose"], device=runtime.sim.device)
                )
                robot.write_joint_position_to_sim_index(
                    position=torch.tensor(state["joint_pos"], device=runtime.sim.device)
                )
                robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(robot.data.joint_vel.torch))
                particle_state = media.data.default_particle_state_w.torch.clone()
                particle_state[..., :3] = torch.tensor(state["particle_pos"], device=runtime.sim.device)
                particle_state[..., 3:] = 0.0
                media.write_particle_state_to_sim_index(particle_state)
            runtime.sim.backend_manager.backend.model.particle_max_velocity = 10.0
            # The MPM reset snapshots the paddle transform after the joint writes.
            runtime.sim.forward()
            world_mask = wp.array([True] * runtime.scene.num_envs + [False], dtype=wp.bool, device=runtime.sim.device)
            reset_mpm_solver_state(
                runtime.sim.backend_manager.backend, world_mask=world_mask, flags=StateFlags.BODY | StateFlags.PARTICLE
            )
        for step in range(scene.steps_before_capture):
            if case.scene == "mpm" and step % 2 == 0:
                robot = runtime.scene["robot"]
                limits = robot.data.soft_joint_pos_limits.torch
                targets = robot.data.joint_pos.torch.clamp(limits[..., 0] + 0.04, limits[..., 1] - 0.04)
                robot.actuators.target_command.set_position_index(value=targets)
            if case.scene in ("franka_cloth", "franka_cable"):
                robot = runtime.scene["robot"]
                targets = robot.data.joint_pos.torch.clone()
                fingers, _ = robot.find_joints("panda_finger_joint1")
                targets[:, fingers] = robot.data.soft_joint_pos_limits.torch[:, fingers].mean(dim=-1)
                robot.actuators.target_command.set_position_index(value=targets)
            render_interval = 2 if case.scene == "mpm" else 4
            runtime.step(render=(step + 1) % render_interval == 0)
        if case.scene in ("rendering_scene", "kuka_materials"):
            runtime.stabilize_camera()
        if case.scene == "kuka_materials":
            before = runtime.camera.data.output["rgb"].torch.clone()
            materials = tuple(runtime.scene.visual_materials.values())
            if case.renderer == "isaac_rtx":
                usd_inputs = tuple(
                    runtime.sim.stage.GetAttributeAtPath(f"{path}.inputs:{material._input_names['color']}")
                    for material in materials
                    for path in material._shader_paths
                )
                authored_colors = tuple(attribute.Get() for attribute in usd_inputs)
            generator = torch.Generator(device="cpu").manual_seed(42)
            colors = (torch.rand((len(materials), runtime.scene.num_envs, 3), generator=generator) * 0.9 + 0.05).to(
                runtime.sim.device
            )
            VisualMaterial.write_channels(materials, {"color": colors})
            runtime.stabilize_camera()
            assert (before != runtime.camera.data.output["rgb"].torch).any(), "Material writes did not reach pixels"
            if case.renderer == "isaac_rtx":
                assert tuple(attribute.Get() for attribute in usd_inputs) == authored_colors
        outputs, info = runtime.camera_outputs()
        # Preserve first-frame references; only wait when an RTX output has not been populated yet.
        for _ in range(10):
            if all(output.count_nonzero() for name, output in outputs.items() if name != "motion_vectors"):
                break
            runtime.render_camera()
            outputs, info = runtime.camera_outputs()
        if case.scene == "mpm":
            rgb = outputs["rgb"].to(torch.int16)
            tan = (rgb[..., 0] - rgb[..., 2] > 50) & (rgb[..., 1] - rgb[..., 2] > 20)
            assert tan.flatten(1).sum(dim=1).min() > 20, "Every camera must see simulated particles"
            assert outputs["instance_id_segmentation_fast"][..., 3][tan].count_nonzero() > 20
        if RenderBufferKind.MOTION_VECTORS in case.aovs:
            step = runtime.sim.get_physics_step_count()
            runtime.step(render=False)
            runtime.render_camera()
            motion_outputs, _ = runtime.camera_outputs()
            assert runtime.sim.get_physics_step_count() == step + 1
            motion = motion_outputs[RenderBufferKind.MOTION_VECTORS]
            validate_motion_vectors(motion)
            outputs[RenderBufferKind.MOTION_VECTORS] = motion

        if scene.expected_instances:
            validate_segmentation(
                outputs,
                info,
                scene.expected_instances,
                min_semantic_pixels=scene.min_semantic_pixels,
                exact_instance_metadata=case.renderer != "newton_warp",
                semantic_colors=(
                    _OVRTX_SEMANTIC_COLORS if case.scene == "rendering_scene" and case.renderer == "ovrtx" else None
                ),
            )
        failures = []
        for aov in case.golden_aovs:
            image_max_diff_pct, min_ssim = scene.image_tolerance(case.renderer, aov)
            golden_filename = case.golden_filename(aov, golden_namespace)
            artifact_label = "-".join(
                part for part in (case.scene, artifact_namespace, golden_filename.removesuffix(".png")) if part
            )
            comparison = compare_to_golden(
                camera_output_image(outputs[aov], aov),
                _GOLDEN_ROOT / case.scene / golden_filename,
                label=artifact_label,
                artifact_dir=_ARTIFACT_DIR,
                max_diff_pct=image_max_diff_pct,
                min_ssim=None if aov in _NO_SSIM else min_ssim,
                alpha_only=aov in _ALPHA_ONLY_AOVS,
            )
            comparison.record(request)
            if not comparison.passed:
                failures.append(f"{aov}: {comparison.error}")
        assert not failures, "\n".join([f"{case.id} failed:", *failures])


def validate_segmentation(
    outputs: dict[str, torch.Tensor],
    info: dict[str, Any] | None,
    expected_instances: dict[str, str],
    *,
    min_semantic_pixels: dict[str, int] | None = None,
    exact_instance_metadata: bool = True,
    semantic_colors: dict[str, tuple[int, int, int, int]] | None = None,
) -> None:
    """Check label coverage and optionally recolor semantic output into the reference palette."""
    min_semantic_pixels = min_semantic_pixels or {}
    required_labels = frozenset(expected_instances.values())
    semantic = outputs.get(RenderBufferKind.SEMANTIC_SEGMENTATION)
    if semantic is not None:
        assert info is not None and RenderBufferKind.SEMANTIC_SEGMENTATION in info
        metadata = info[RenderBufferKind.SEMANTIC_SEGMENTATION]
        assert metadata is not None and "idToLabels" in metadata
        id_to_labels = metadata["idToLabels"]
        channels = min(semantic.shape[-1], 4)
        canonical = semantic.clone() if semantic_colors is not None else None
        normalized = semantic.is_floating_point() and semantic.max().item() <= 1.0
        for label in required_labels:
            colors = [
                key for key, entry in id_to_labels.items() if isinstance(entry, dict) and entry.get("class") == label
            ]
            assert colors, f"The semantic metadata does not contain required label {label!r}."
            rendered_pixels = 0
            for value in colors:
                value = ast.literal_eval(value) if isinstance(value, str) else value
                color = torch.tensor(value[:channels], device=semantic.device, dtype=semantic.dtype)
                if normalized:
                    color = color / 255.0
                mask = torch.all(semantic[..., :channels] == color, dim=-1)
                rendered_pixels += int(mask.sum().item())
                if canonical is not None:
                    color = semantic.new_tensor(semantic_colors[label][:channels])
                    canonical[mask] = color / 255.0 if normalized else color
            assert rendered_pixels, f"The semantic output does not contain required label {label!r}."
            if label in min_semantic_pixels:
                minimum = min_semantic_pixels[label]
                assert rendered_pixels >= minimum, (
                    f"Semantic label {label!r} has {rendered_pixels} visible pixels; expected at least {minimum}."
                )

        labels = {
            entry["class"]
            for entry in id_to_labels.values()
            if isinstance(entry, dict) and "class" in entry and entry["class"] not in {"BACKGROUND", "UNLABELLED"}
        }
        assert labels == required_labels, f"Semantic labels differ: expected {required_labels}, got {labels}."
        if canonical is not None:
            outputs[RenderBufferKind.SEMANTIC_SEGMENTATION] = canonical

    instance = outputs.get(RenderBufferKind.INSTANCE_SEGMENTATION)
    if instance is None:
        return
    assert expected_instances, "Scenes requesting instance segmentation must declare their expected instances."
    assert instance.shape[-1] == 4, f"Expected colorized instance segmentation, got shape {instance.shape}."
    assert info is not None and RenderBufferKind.INSTANCE_SEGMENTATION in info
    metadata = info[RenderBufferKind.INSTANCE_SEGMENTATION]
    assert metadata is not None and {"idToLabels", "idToSemantics"} <= set(metadata)

    def normalize(mapping: dict[Any, Any]) -> dict[tuple[int, ...], Any]:
        normalized = {
            tuple(int(channel) for channel in (ast.literal_eval(key) if isinstance(key, str) else key)): value
            for key, value in mapping.items()
        }
        assert len(normalized) == len(mapping), "Instance-segmentation metadata contains duplicate color keys."
        return normalized

    id_to_labels = normalize(metadata["idToLabels"])
    id_to_semantics = normalize(metadata["idToSemantics"])
    reserved = {
        (0, 0, 0, 0): ("BACKGROUND", {"class": "BACKGROUND"}),
        (0, 0, 0, 255): ("UNLABELLED", {"class": "UNLABELLED"}),
    }
    image_colors = {
        tuple(int(channel) for channel in color)
        for color in torch.unique(instance.reshape(-1, instance.shape[-1]), dim=0).cpu().tolist()
    }
    required_colors = image_colors | set(reserved)
    label_colors, semantic_colors = set(id_to_labels), set(id_to_semantics)
    assert label_colors == semantic_colors, "Instance metadata maps use different color keys."
    missing_colors = required_colors - label_colors
    assert not missing_colors, f"Instance metadata is missing rendered colors: {sorted(missing_colors)}."
    if exact_instance_metadata:
        extra_colors = label_colors - required_colors
        assert not extra_colors, f"Instance metadata contains unrendered colors: {sorted(extra_colors)}."
    for color, (label, semantics) in reserved.items():
        assert id_to_labels[color] == label
        assert id_to_semantics[color] == semantics

    actual_instances = {}
    for color in set(id_to_labels) - set(reserved):
        path = id_to_labels[color]
        semantics = id_to_semantics[color]
        assert isinstance(path, str) and path not in actual_instances
        assert isinstance(semantics, dict) and set(semantics) == {"class"}
        actual_instances[path] = semantics["class"]
    if exact_instance_metadata:
        assert actual_instances == expected_instances, (
            f"Instance-segmentation metadata mismatch.\n  expected: {expected_instances}\n  actual: {actual_instances}"
        )
    else:
        # Newton-Warp builds scene-static mappings from compiled collision shapes. Those mappings may include
        # occluded shapes, while legacy USD cloning can expose only env_0 paths. The reported path/class pairs
        # must still be a non-empty, scene-valid subset, and the checks above require every rendered color to map.
        actual_items, expected_items = set(actual_instances.items()), set(expected_instances.items())
        assert actual_items and actual_items <= expected_items, (
            f"Instance-segmentation metadata is not a scene-valid subset.\n"
            f"  expected: {expected_instances}\n  actual: {actual_instances}"
        )


def validate_motion_vectors(motion: torch.Tensor) -> None:
    """Require finite, non-zero motion confined to a minority of pixels in each fixed view."""
    assert torch.isfinite(motion).all(), "Motion vectors contain non-finite values."
    magnitude = motion[..., :2].abs().amax(dim=-1)
    # Static RTX pixels have roundoff near 1e-5 pixels; visible motion is well above that floor.
    moving_pixels = (magnitude > 1.0e-3).flatten(1).sum(dim=1)
    view_pixels = magnitude.shape[1] * magnitude.shape[2]
    assert torch.all(moving_pixels > 20), f"Too few moving pixels: {moving_pixels.tolist()}"
    assert torch.all(moving_pixels < view_pixels // 2), f"Motion is not localized: {moving_pixels.tolist()}"
