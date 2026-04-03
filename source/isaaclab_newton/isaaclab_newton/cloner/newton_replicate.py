# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import re
from collections.abc import Callable

import torch
import warp as wp
from newton import GeoType, ModelBuilder, solvers
from newton._src.usd.schemas import SchemaResolverNewton, SchemaResolverPhysx

from pxr import Usd, UsdGeom

from isaaclab.physics import PhysicsManager
from isaaclab.physics.scene_data_requirements import VisualizerPrebuiltArtifacts

from isaaclab_newton.physics import NewtonManager


def _build_newton_builder_from_mapping(
    stage: Usd.Stage,
    sources: list[str],
    env_ids: torch.Tensor,
    mapping: torch.Tensor,
    positions: torch.Tensor | None = None,
    quaternions: torch.Tensor | None = None,
    up_axis: str = "Z",
    simplify_meshes: bool = True,
) -> tuple[ModelBuilder, object]:
    """Build a Newton model builder from clone mapping inputs.

    Args:
        stage: USD stage containing source assets.
        sources: Source prim paths used for cloning.
        env_ids: Environment ids for destination worlds.
        mapping: Boolean source-to-environment mapping matrix.
        positions: Optional per-environment world positions.
        quaternions: Optional per-environment orientations in xyzw order.
        up_axis: Up axis for the Newton model builder.
        simplify_meshes: Whether to run convex-hull mesh approximation.

    Returns:
        Tuple of the populated Newton model builder and stage metadata returned by ``add_usd``.
    """
    if positions is None:
        positions = torch.zeros((mapping.size(1), 3), device=mapping.device, dtype=torch.float32)
    if quaternions is None:
        quaternions = torch.zeros((mapping.size(1), 4), device=mapping.device, dtype=torch.float32)
        quaternions[:, 3] = 1.0

    schema_resolvers = [SchemaResolverNewton(), SchemaResolverPhysx()]

    builder = ModelBuilder(up_axis=up_axis)
    stage_info = builder.add_usd(
        stage,
        ignore_paths=["/World/envs"] + sources,
        schema_resolvers=schema_resolvers,
    )

    # The prototype is built from env_0 in absolute world coordinates.
    # add_builder xforms are deltas from env_0 so positions don't get double-counted.
    env0_pos = positions[0]

    # SDF collision requires original triangle meshes for mesh.build_sdf().
    # Convex hull approximation destroys the source geometry, so shapes
    # matching SDF patterns must be excluded from approximation here.
    # _apply_sdf_config() builds the SDF on each prototype after approximation.
    cfg = PhysicsManager._cfg
    sdf_cfg = getattr(cfg, "sdf_cfg", None) if cfg is not None else None
    body_pats = [re.compile(x) for x in sdf_cfg.body_patterns] if sdf_cfg and sdf_cfg.body_patterns else None
    shape_pats = [re.compile(x) for x in sdf_cfg.shape_patterns] if sdf_cfg and sdf_cfg.shape_patterns else None
    has_sdf_patterns = body_pats is not None or shape_pats is not None

    protos: dict[str, ModelBuilder] = {}
    for src_path in sources:
        p = ModelBuilder(up_axis=up_axis)
        solvers.SolverMuJoCo.register_custom_attributes(p)
        p.add_usd(
            stage,
            root_path=src_path,
            load_visual_shapes=True,
            skip_mesh_approximation=True,
            schema_resolvers=schema_resolvers,
        )
        if simplify_meshes:
            if has_sdf_patterns:
                sdf_bodies: set[int] = set()
                if body_pats is not None:
                    for bi in range(len(p.body_label)):
                        if any(pat.search(p.body_label[bi]) for pat in body_pats):
                            sdf_bodies.add(bi)

                approx_indices = []
                for i in range(len(p.shape_type)):
                    if p.shape_type[i] != GeoType.MESH:
                        continue
                    # Skip shapes that will use SDF (matched by body or shape pattern)
                    if p.shape_body[i] in sdf_bodies:
                        continue
                    if shape_pats is not None:
                        lbl = p.shape_label[i] if i < len(p.shape_label) else ""
                        if any(pat.search(lbl) for pat in shape_pats):
                            continue
                    approx_indices.append(i)
                if approx_indices:
                    p.approximate_meshes("convex_hull", shape_indices=approx_indices, keep_visual_shapes=True)
            else:
                p.approximate_meshes("convex_hull", keep_visual_shapes=True)
        # Build SDF on prototype before add_builder copies it N times.
        # Mesh objects are shared by reference, so SDF is built once and
        # all environments inherit it.
        NewtonManager._apply_sdf_config(p)
        protos[src_path] = p

    # create a separate world for each environment (heterogeneous spawning)
    # Newton assigns sequential world IDs (0, 1, 2, ...), so we need to track the mapping
    for col, _ in enumerate(env_ids.tolist()):
        # begin a new world context (Newton assigns world ID = col)
        builder.begin_world()
        # add all active sources for this world
        delta_pos = (positions[col] - env0_pos).tolist()
        for row in torch.nonzero(mapping[:, col], as_tuple=True)[0].tolist():
            builder.add_builder(
                protos[sources[row]],
                xform=wp.transform(delta_pos, quaternions[col].tolist()),
            )
        # end the world context
        builder.end_world()

    return builder, stage_info


def _rename_builder_labels(
    builder: ModelBuilder, sources: list[str], destinations: list[str], env_ids: torch.Tensor, mapping: torch.Tensor
) -> None:
    """Rename builder labels/keys from source roots to destination roots.

    Args:
        builder: Newton model builder to update in-place.
        sources: Source prim root paths.
        destinations: Destination prim path templates.
        env_ids: Environment ids corresponding to mapping columns.
        mapping: Boolean source-to-environment mapping matrix.
    """
    # per-source, per-world renaming (strict prefix swap), compact style preserved
    for i, src_path in enumerate(sources):
        src_prefix_len = len(src_path.rstrip("/"))
        swap = lambda name, new_root: new_root + name[src_prefix_len:]  # noqa: E731
        world_cols = torch.nonzero(mapping[i], as_tuple=True)[0].tolist()
        # Map Newton world IDs (sequential) to destination paths using env_ids
        world_roots = {int(env_ids[c]): destinations[i].format(int(env_ids[c])) for c in world_cols}

        for t in ("body", "joint", "shape", "articulation"):
            labels = getattr(builder, f"{t}_label", None)
            if labels is None:
                labels = getattr(builder, f"{t}_key")
            worlds_arr = getattr(builder, f"{t}_world")
            for k, w in enumerate(worlds_arr):
                world_id = int(w)
                if world_id in world_roots and labels[k].startswith(src_path):
                    labels[k] = swap(labels[k], world_roots[world_id])


def newton_physics_replicate(
    stage: Usd.Stage,
    sources: list[str],
    destinations: list[str],
    env_ids: torch.Tensor,
    mapping: torch.Tensor,
    positions: torch.Tensor | None = None,
    quaternions: torch.Tensor | None = None,
    device: str = "cpu",
    up_axis: str = "Z",
    simplify_meshes: bool = True,
):
    """Replicate prims into a Newton ``ModelBuilder`` using a per-source mapping.

    Args:
        stage: USD stage containing source assets.
        sources: Source prim paths used for cloning.
        destinations: Destination prim path templates.
        env_ids: Environment ids for destination worlds.
        mapping: Boolean source-to-environment mapping matrix.
        positions: Optional per-environment world positions.
        quaternions: Optional per-environment orientations in xyzw order.
        device: Device used by the finalized Newton model builder.
        up_axis: Up axis for the Newton model builder.
        simplify_meshes: Whether to run convex-hull mesh approximation.

    Returns:
        Tuple of the populated Newton model builder and stage metadata.
    """
    builder, stage_info = _build_newton_builder_from_mapping(
        stage=stage,
        sources=sources,
        env_ids=env_ids,
        mapping=mapping,
        positions=positions,
        quaternions=quaternions,
        up_axis=up_axis,
        simplify_meshes=simplify_meshes,
    )
    _rename_builder_labels(builder, sources, destinations, env_ids, mapping)
    NewtonManager.set_builder(builder)
    NewtonManager._num_envs = mapping.size(1)
    return builder, stage_info


def newton_visualizer_prebuild(
    stage: Usd.Stage,
    sources: list[str],
    destinations: list[str],
    env_ids: torch.Tensor,
    mapping: torch.Tensor,
    positions: torch.Tensor | None = None,
    quaternions: torch.Tensor | None = None,
    device: str = "cpu",
    up_axis: str = "Z",
    simplify_meshes: bool = True,
):
    """Replicate a clone plan into a finalized Newton model/state for visualization.

    Unlike :func:`newton_physics_replicate`, this path does not mutate ``NewtonManager`` and is intended
    for prebuilding visualizer-only artifacts that can be consumed by scene data providers.

    Args:
        stage: USD stage containing source assets.
        sources: Source prim paths used for cloning.
        destinations: Destination prim path templates.
        env_ids: Environment ids for destination worlds.
        mapping: Boolean source-to-environment mapping matrix.
        positions: Optional per-environment world positions.
        quaternions: Optional per-environment orientations in xyzw order.
        device: Device used by the finalized Newton model.
        up_axis: Up axis for the Newton model builder.
        simplify_meshes: Whether to run convex-hull mesh approximation.

    Returns:
        Tuple of finalized Newton model and state.
    """
    builder, _ = _build_newton_builder_from_mapping(
        stage=stage,
        sources=sources,
        env_ids=env_ids,
        mapping=mapping,
        positions=positions,
        quaternions=quaternions,
        up_axis=up_axis,
        simplify_meshes=simplify_meshes,
    )
    _rename_builder_labels(builder, sources, destinations, env_ids, mapping)
    model = builder.finalize(device=device)
    state = model.state()
    return model, state


def create_newton_visualizer_prebuild_clone_fn(
    stage,
    set_visualizer_artifact: Callable[[VisualizerPrebuiltArtifacts | None], None],
):
    """Create a cloner callback that prebuilds Newton visualizer artifacts.

    Args:
        stage: USD stage used by the clone callback.
        set_visualizer_artifact: Callback used to store the produced prebuilt artifact.

    Returns:
        Clone callback that builds and stores visualizer prebuilt artifacts.
    """
    up_axis = UsdGeom.GetStageUpAxis(stage)

    def _visualizer_clone_fn(
        stage,
        sources,
        destinations,
        env_ids,
        mapping,
        positions=None,
        quaternions=None,
        device="cpu",
    ):
        """Prebuild Newton model/state and store visualizer artifacts for clone consumers."""
        model, state = newton_visualizer_prebuild(
            stage=stage,
            sources=sources,
            destinations=destinations,
            env_ids=env_ids,
            mapping=mapping,
            positions=positions,
            quaternions=quaternions,
            device=device,
            up_axis=up_axis,
        )
        set_visualizer_artifact(
            VisualizerPrebuiltArtifacts(
                model=model,
                state=state,
                rigid_body_paths=list(getattr(model, "body_label", None) or getattr(model, "body_key", [])),
                articulation_paths=list(
                    getattr(model, "articulation_label", None) or getattr(model, "articulation_key", [])
                ),
                num_envs=int(mapping.size(1)),
            )
        )

    return _visualizer_clone_fn
