# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Simulation-owned OVStage resources and shared attribute-column helpers.

``ovstage`` is a hard dependency of ``isaaclab_ov``, so it is imported unconditionally here.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterator, Sequence

import numpy as np
import ovstage
import warp as wp

from pxr import Sdf, Usd, UsdRender, UsdUtils

from isaaclab.sim import BackendCfg
from isaaclab.utils import configclass

from .ovstage_compat import OVSTAGE_REQUIRES_CPU_HIERARCHY_BEFORE_CLONE, populate_stage

logger = logging.getLogger(__name__)

# DLDataType for a 4x4 double matrix (``omni:xform`` column). ovstage stores omni:xform as one
# 16-lane float64 element per prim; wp.mat44d maps to the same layout via __dlpack__.
OVSTAGE_XFORM_DTYPE = ovstage.DLDataType(code=ovstage.DLDataTypeCode.kDLFloat, bits=64, lanes=16)

# DLDataType for a float32 3-vector (``points`` column). ovstage stores ``point3f[] points`` as one
# 3-lane float32 element per vertex.
OVSTAGE_POINT_DTYPE = ovstage.DLDataType(code=ovstage.DLDataTypeCode.kDLFloat, bits=32, lanes=3)


def _handle_raw_id(handle: object) -> int:
    """Return a numeric ID from an OVStage integer or typed token/path handle.

    OVStage 0.4 handles reject implicit integer conversion. Extract the ID only at
    numeric buffer boundaries; native OVStage calls should retain typed handles.
    """
    return getattr(handle, "raw_id", handle)


def create_ovstage(name: str) -> ovstage.Stage:
    """Create an ovstage stage using Isaac Lab's process-wide stage configuration.

    ovstage's hierarchy computation model drives its automatic world-transform updates. It is
    process-scoped rather than per-stage: ovstage applies it when the first process reference is
    acquired and raises if a later stage asks for a conflicting model while another stage is live.
    Every Isaac Lab stage is therefore created through this helper so the whole process agrees on
    one model.

    Args:
        name: Instance name used for ovstage diagnostics.

    Returns:
        The created :class:`ovstage.Stage`.
    """
    config = ovstage.StageConfig(
        runtime_default_hierarchy_computation_model=ovstage.HierarchyComputationModel.GPU_INCREMENTAL
    )
    return ovstage.Stage(name, config=config)


def xform_tensor_from_numpy(xforms: np.ndarray) -> ovstage.DLTensor:
    """Wrap a ``(N, 4, 4)`` float64 host array as a 16-lane DLTensor for ``omni:xform`` writes.

    Args:
        xforms: Array of shape ``(N, 4, 4)`` with dtype ``float64``.

    Returns:
        A :class:`ovstage.DLTensor` with shape ``[N]`` and ``lanes=16``.
    """
    flat = np.ascontiguousarray(xforms, dtype=np.float64).reshape(-1)
    return ovstage.make_dltensor(flat, dtype=OVSTAGE_XFORM_DTYPE, shape=[xforms.shape[0]])


def xform_tensor_from_warp(xforms: wp.array) -> ovstage.DLTensor:
    """Describe a warp ``mat44d`` array as a 16-lane DLTensor for ``omni:xform`` writes.

    The array is consumed zero-copy through DLPack: a warp ``mat44d`` exports as ``(N, 4, 4)``
    ``lanes=1``, and ovstage folds the trailing matrix axes into the ``lanes=16`` the column
    expects. A device array therefore reaches ovstage without a host round-trip.

    The caller owns the data: the returned tensor must stay alive until the consuming write
    completes, and that write must be ordered against the kernels that produced
    :paramref:`xforms` — pass their Warp stream as ``write_attribute(cuda_stream=...)``.

    Args:
        xforms: Warp array of shape ``[N]`` and dtype :class:`warp.mat44d`.

    Returns:
        A :class:`ovstage.DLTensor` with shape ``[N]`` and ``lanes=16``.
    """
    return ovstage.make_dltensor(xforms, dtype=OVSTAGE_XFORM_DTYPE)


def points_tensor_from_warp(points: wp.array) -> ovstage.DLTensor:
    """Describe a warp ``vec3f`` array as a 3-lane DLTensor for ``points`` writes.

    The array is consumed zero-copy through DLPack: a warp ``vec3f`` exports as ``(N, 3)``
    ``lanes=1``, and ovstage folds the trailing component axis into the ``lanes=3`` the
    ``point3f[]`` column expects. A device array therefore reaches ovstage without a host
    round-trip.

    The caller owns the data: the returned tensor must stay alive until the consuming write
    completes, and that write must be ordered against the kernels that produced
    :paramref:`points` — pass their Warp stream as ``write_attribute(cuda_stream=...)``.

    Args:
        points: Warp array of shape ``[N]`` and dtype :class:`warp.vec3f`.

    Returns:
        A :class:`ovstage.DLTensor` with shape ``[N]`` and ``lanes=3``.
    """
    return ovstage.make_dltensor(points, dtype=OVSTAGE_POINT_DTYPE)


def ovstage_replicate(
    stage: ovstage.Stage,
    paths: ovstage.PathDictionary,
    copies: Sequence[tuple[str, Sequence[str]]],
    env_paths: Sequence[str],
    positions: np.ndarray | None = None,
    *,
    ordinal: int,
) -> None:
    """Apply the same prepared copies and placement to an OVStage scene.

    On OVStage 0.4, compute the source hierarchy on CPU at ``ordinal`` before cloning
    when any destination paths exist. The runtime hierarchy model remains unchanged.

    Args:
        stage: Native stage holding the source prims.
        paths: Path dictionary of ``stage``.
        copies: Source paths paired with destination paths, ordered with parents before children.
            Self-copies must be excluded.
        env_paths: Environment-root paths in placement order.
        positions: Optional environment positions [m], shape ``[len(env_paths), 3]``.
        ordinal: Input and output ordinal for hierarchy preparation, and write ordinal
            for the clones and environment placement.
    """
    if OVSTAGE_REQUIRES_CPU_HIERARCHY_BEFORE_CLONE and any(targets for _, targets in copies):
        # Propagate source transforms on CPU before cloning to avoid copying
        # GPU-resident transform columns per clone. Keep the runtime GPU default.
        # OVSTAGE tracking NVBUG 7027326, when fixed, we can remove this compute call.
        stage.compute_hierarchy(
            input_ordinal=ordinal,
            output_ordinal=ordinal,
            model=ovstage.HierarchyComputationModel.CPU_INCREMENTAL,
        )
    _apply_ovstage_copies(stage, paths, copies, env_paths, positions, ordinal=ordinal)


@configclass
class OvstageBackendCfg(BackendCfg):
    """Configuration of a simulation-owned stage and its populated domains."""

    class_type: type[OvstageBackend] | str = "{DIR}.stage:OvstageBackend"
    scene_key: BackendCfg | int | None = None
    """None selects the simulation scene; isolated consumers use their configuration or instance identity."""
    population_domains: ovstage.PopulationDomain = ovstage.PopulationDomain.RENDERING
    """Native USD domains to populate, independent of the physics population API."""
    populate_physics: bool = False
    """Include physics data for OVPhysX, using the population API supported by the installed OVStage."""
    defer_render_copies: bool = False
    """Populate physics originals first, completing render copies only after native physics bindings exist."""


class OvstageBackend:
    """Own a detached stage and its paths until the simulation has closed its consumers."""

    def __init__(self, cfg: OvstageBackendCfg):
        """Create the stage identified by its consumer configuration and population domains.

        Args:
            cfg: Scene identity and USD domains to populate.
        """
        self.cfg = cfg
        self.clone_copies: list[tuple[str, list[str]]] = []
        self.clone_env_paths: list[str] = []
        self.population_env_paths: list[str] = []
        self.clone_positions: np.ndarray | None = None
        self._deferred_render_copies: list[tuple[str, list[str]]] | None = None
        self._render_copy_error: Exception | None = None
        self.resources = contextlib.ExitStack()
        """Release borrowers before paths and stage, in reverse registration order."""
        self.reset()

    @property
    def render_copies_pending(self) -> bool:
        """Whether population reserved a render-copy commit that consumers have not completed."""
        return self._deferred_render_copies is not None

    def populate(self, usda: str) -> None:
        """Import the exported prototypes and apply the clone context's prepared operations.

        Population commits ordinal 1. With early physics attachment, ordinal 2 is reserved for
        deferred render copies after native physics cloning and binding acquisition. Otherwise,
        consumers may author render products and poses starting at ordinal 2.

        Args:
            usda: USD scene containing the routed prototypes and their materials.
        """
        if self.ordinal:
            return

        if self.cfg.defer_render_copies:
            self._populate_originals(usda)
            return

        # Ordinal 0 is the empty state; population and cloning form the first committed write.
        populate_stage(
            self.stage, usda, ordinal=1, domains=self.cfg.population_domains, physics=self.cfg.populate_physics
        )
        ovstage_replicate(
            self.stage, self.paths, self.clone_copies, self.clone_env_paths, self.clone_positions, ordinal=1
        )
        self.ordinal = 1
        self.commit()

    def complete_render_copies(self) -> None:
        """Seal render copies at ordinal 2 after physics cloning and binding acquisition.

        This ordinal is rendering-only and must never be passed to OVPhysX's stage-update API.
        Repeated calls after success do nothing. A failure requires a stage reset because native
        copies may already have been applied; population then rebuilds the pending copies.
        """
        if self._deferred_render_copies is None:
            return
        if self._render_copy_error is not None:
            raise RuntimeError(
                "Render-copy preparation failed; reset the stage before retrying."
            ) from self._render_copy_error
        try:
            if self.ordinal != 2:
                raise RuntimeError(f"Expected reserved render-copy ordinal 2; found {self.ordinal}.")
            _apply_ovstage_copies(
                self.stage,
                self.paths,
                self._deferred_render_copies,
                self.clone_env_paths,
                self.clone_positions,
                ordinal=2,
            )
            self.commit()
        except Exception as exc:
            self._render_copy_error = exc
            raise
        self._deferred_render_copies = None

    @contextlib.contextmanager
    def query(self, prim_paths: Sequence[str]) -> Iterator[ovstage.Query]:
        """Retain a query and its path list together until the caller leaves the context.

        Args:
            prim_paths: Prim paths in the query's binding order.

        Yields:
            The query, released before its caller-owned path list.
        """
        paths = self.paths.create_path_list_from_strings(prim_paths)
        try:
            with self.stage.query_from_path_list(paths) as query:
                yield query
        finally:
            self.paths.destroy_path_list(paths)

    def commit(self) -> int:
        """Seal pending writes and return their ordinal; all consumers share the next write ordinal."""
        ordinal = self.ordinal
        self.stage.advance_write_floor(ordinal=ordinal).wait()
        self.ordinal += 1
        return ordinal

    def close(self) -> None:
        """Release borrowers, then the path dictionary and stage."""
        self._deferred_render_copies = None
        self._render_copy_error = None
        self.resources.close()
        self.stage = self.paths = None

    def reset(self) -> None:
        """Recreate the native stage after all consumers detach, retaining the prepared clone operations."""
        self.close()
        self.ordinal = self.next_camera_id = 0
        with contextlib.ExitStack() as resources:
            self.stage = resources.enter_context(create_ovstage("isaaclab.scene"))
            self.paths = resources.enter_context(ovstage.PathDictionary(self.stage))
            self.resources = resources.pop_all()

    def _populate_originals(self, usda: str) -> None:
        """Populate complete physics originals and retain the remaining render copies."""
        layer = Sdf.Layer.CreateAnonymous()
        if not layer.ImportFromString(usda):
            raise ValueError("Cannot parse the original-world export for early physics attachment.")
        original_stage = Usd.Stage.Open(layer)
        _author_render_settings(original_stage)
        for path in self.clone_env_paths:
            prim = original_stage.GetPrimAtPath(path)
            if not prim or prim.GetAllChildren() or prim.GetAppliedSchemas() or prim.HasAuthoredMetadata("apiSchemas"):
                continue
            if any(prop.GetName().startswith(("physics:", "physx")) for prop in prim.GetAuthoredProperties()):
                continue
            original_stage.RemovePrim(path)
        # Serialization may materialize additional assets in original-bearing worlds. Retain
        # only missing destinations, including render-only descendants of complete originals.
        copies = [
            (source, [target for target in targets if target != source and not original_stage.GetPrimAtPath(target)])
            for source, targets in self.clone_copies
        ]
        for source, targets in copies:
            if targets and not original_stage.GetPrimAtPath(source):
                raise ValueError(f"The original-world export omitted render clone source {source!r}.")
        populate_stage(self.stage, layer.ExportToString(), ordinal=1, domains=self.cfg.population_domains, physics=True)
        if OVSTAGE_REQUIRES_CPU_HIERARCHY_BEFORE_CLONE and any(targets for _, targets in copies):
            self.stage.compute_hierarchy(
                input_ordinal=1, output_ordinal=1, model=ovstage.HierarchyComputationModel.CPU_INCREMENTAL
            )
        self.ordinal = 1
        self.commit()
        self._deferred_render_copies = copies


def export_stage_to_string(
    stage: Usd.Stage,
    clone_env_paths: Sequence[str],
    source_paths: tuple[str, ...],
    keep_env_roots: bool = True,
    *,
    env_paths: Sequence[str] = (),
    preserve_references: bool = False,
) -> str:
    """Export routed prototypes without mutating the simulation's authored USD stage.

    An anonymous session layer hides non-source subtrees, including copies already authored
    by USD replication. Native clone contexts recreate environment frames from the clone plan;
    callers preserving authored frames can retain them with ``keep_env_roots``.

    Args:
        stage: USD stage to export.
        clone_env_paths: Declared environment roots. Source filtering is skipped for one environment.
        source_paths: Prototype subtrees retained with their materials and descendants.
        keep_env_roots: Whether to retain non-source environment roots and their transforms.
        env_paths: Environment frames to create in the private export before native cloning.
        preserve_references: Retain clean file-backed references and instances instead of baking
            their geometry into text. Anonymous or unsaved referenced layers require full
            composition so that the returned string does not depend on their Python lifetime.

    Returns:
        USDA text containing the selected prototypes. By default, composition arcs are baked.
    """
    export_session = Sdf.Layer.CreateAnonymous()
    export_session.subLayerPaths = [stage.GetSessionLayer().identifier]
    export_stage = Usd.Stage.Open(stage.GetRootLayer(), export_session)
    export_stage.SetEditTarget(export_session)
    _author_render_settings(export_stage)
    prim_paths = []
    if len(clone_env_paths) > 1:
        source_path_set = frozenset(map(Sdf.Path, source_paths))
        retained_env_paths = frozenset(map(Sdf.Path, env_paths))
        ancestors = {prefix for source in source_path_set for prefix in source.GetPrefixes()}
        for env_path in clone_env_paths:
            env_root = export_stage.GetPrimAtPath(env_path)
            if not env_root:
                continue
            prims = iter(Usd.PrimRange(env_root))
            for prim in prims:
                path = prim.GetPath()
                if path in source_path_set:
                    prims.PruneChildren()
                elif (
                    path not in ancestors
                    and path not in retained_env_paths
                    and not (keep_env_roots and path == env_root.GetPath())
                ):
                    # Kit may already have USD copies; routing may also exclude a prototype.
                    prim_paths.append(path)
                    prims.PruneChildren()

        with Sdf.ChangeBlock():
            for prim_path in prim_paths:
                Sdf.CreatePrimInLayer(export_session, prim_path).active = False
                logger.debug("Deactivated prim: %s", prim_path)
        logger.info("Deactivated %d prims in total", len(prim_paths))
    # CPU hierarchy propagation requires USD-populated frames, including clone ancestors.
    # Author them only in the export; native cloning writes their poses from the plan.
    # The local layer stack is baked either way. External in-memory edits cannot be borrowed
    # by a detached string; retain the fully composed export for those scenes and legacy callers.
    external_layers = set(export_stage.GetUsedLayers()) - set(export_stage.GetLayerStack())
    if preserve_references and all(not (layer.anonymous or layer.dirty) for layer in external_layers):
        # Keeping reference arcs avoids duplicating dense geometry and preserves material
        # bindings whose targets cross nested instance scopes.
        layer = UsdUtils.FlattenLayerStack(export_stage)
        # A remaining arc can also point back into the baked root/session stack, directly
        # or through a clean referenced layer. Such unsaved data must be baked as well.
        dependencies = (
            Sdf.Layer.Find(Sdf.ComputeAssetPathRelativeToLayer(source, path))
            for source in (layer, *external_layers)
            for path in source.GetCompositionAssetDependencies()
            if path
        )
        if any(dependency is not None and (dependency.anonymous or dependency.dirty) for dependency in dependencies):
            layer = export_stage.Flatten()
        elif prim_paths:
            # Unlike Stage.Flatten, layer-stack flattening retains inactive authored copies.
            # Remove them, restoring an inactive override if a reference still supplies a prim.
            edits = Sdf.BatchNamespaceEdit()
            for path in prim_paths:
                edits.Add(path, Sdf.Path.emptyPath)
            if not layer.Apply(edits):
                raise RuntimeError("Failed to remove unrouted native USD prototypes.")
            composed = Usd.Stage.Open(layer)
            referenced_paths = [path for path in prim_paths if composed.GetPrimAtPath(path)]
            with Sdf.ChangeBlock():
                for path in referenced_paths:
                    Sdf.CreatePrimInLayer(layer, path).active = False
    else:
        layer = export_stage.Flatten()
    with Sdf.ChangeBlock():
        for path in env_paths:
            prim = Sdf.CreatePrimInLayer(layer, path)
            if not prim.typeName:
                prim.typeName = "Xform"
            while prim and prim.specifier == Sdf.SpecifierOver:
                prim.specifier = Sdf.SpecifierDef
                prim = prim.nameParent
    return layer.ExportToString()


def _apply_ovstage_copies(
    stage: ovstage.Stage,
    paths: ovstage.PathDictionary,
    copies: Sequence[tuple[str, Sequence[str]]],
    env_paths: Sequence[str],
    positions: np.ndarray | None,
    *,
    ordinal: int,
) -> None:
    """Apply prepared copies and placement after source hierarchy preparation."""
    for source, targets in copies:
        if targets:
            stage.clone(source, targets, ordinal=ordinal)
    if positions is not None and env_paths:
        xforms = np.tile(np.eye(4, dtype=np.float64), (len(env_paths), 1, 1))
        xforms[:, 3, :3] = positions
        path_list = paths.create_path_list_from_strings(env_paths)
        try:
            with stage.query_from_path_list(path_list) as query:
                stage.write_attribute(
                    query,
                    "omni:xform",
                    ordinal=ordinal,
                    tensors=xform_tensor_from_numpy(xforms),
                    is_array=False,
                    semantic=ovstage.AttributeSemantic.MATRIX,
                ).wait()
        finally:
            paths.destroy_path_list(path_list)


def _author_render_settings(stage: Usd.Stage) -> None:
    """Preserve render settings and scene partition defaults in a private scene export."""
    # Perspective cameras see every partition; sensor cameras retain their explicit partition.
    settings_path = stage.GetMetadata("renderSettingsPrimPath") or "/Render/IsaacLabSettings"
    settings = UsdRender.Settings.Define(stage, settings_path)
    settings.GetPrim().AddAppliedSchema("OmniRtxSettingsGlobalCommonAPI_1")
    settings.GetPrim().CreateAttribute(
        "omni:rtx:scenePartitioning:showAllPartitionsByDefault", Sdf.ValueTypeNames.Bool
    ).Set(True)
    stage.SetMetadata("renderSettingsPrimPath", settings_path)
