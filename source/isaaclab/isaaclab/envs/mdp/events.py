# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to enable different events.

Events include anything related to altering the simulation state. This includes changing the physics
materials, applying external forces, and resetting the state of the asset.

The functions can be passed to the :class:`isaaclab.managers.EventTermCfg` object to enable
the event introduced by the function.
"""

from __future__ import annotations

import logging
import math
import re
from typing import TYPE_CHECKING, Literal

import torch
import warp as wp

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.utils.version import compare_versions

if TYPE_CHECKING:
    from isaaclab_physx.assets import DeformableObject

    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.terrains import TerrainImporter

    from .. import ManagerBasedEnv

logger = logging.getLogger(__name__)


def randomize_rigid_body_scale(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    scale_range: tuple[float, float] | dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
    relative_child_path: str | None = None,
):
    """Randomize the scale of a rigid body asset in the USD stage.

    This function modifies the "xformOp:scale" property of all the prims corresponding to the asset.

    It takes a tuple or dictionary for the scale ranges. If it is a tuple, then the scaling along
    individual axis is performed equally. If it is a dictionary, the scaling is independent across each dimension.
    The keys of the dictionary are ``x``, ``y``, and ``z``. The values are tuples of the form ``(min, max)``.

    If the dictionary does not contain a key, the range is set to one for that axis.

    Relative child path can be used to randomize the scale of a specific child prim of the asset.
    For example, if the asset at prim path expression ``/World/envs/env_.*/Object`` has a child
    with the path ``/World/envs/env_.*/Object/mesh``, then the relative child path should be ``mesh`` or
    ``/mesh``.

    .. attention::
        Since this function modifies USD properties that are parsed by the physics engine once the simulation
        starts, the term should only be used before the simulation starts playing. This corresponds to the
        event mode named "usd". Using it at simulation time, may lead to unpredictable behaviors.

    .. note::
        When randomizing the scale of individual assets, please make sure to set
        :attr:`isaaclab.scene.InteractiveSceneCfg.replicate_physics` to False. This ensures that physics
        parser will parse the individual asset properties separately.
    """
    from isaaclab.assets import BaseArticulation  # noqa: PLC0415

    if env.sim.is_playing():
        raise RuntimeError(
            "Randomizing scale while simulation is running leads to unpredictable behaviors."
            " Please ensure that the event term is called before the simulation starts by using the 'usd' mode."
        )

    asset: RigidObject = env.scene[asset_cfg.name]
    if isinstance(asset, BaseArticulation):
        raise ValueError(
            "Scaling an articulation randomly is not supported, as it affects joint attributes and can cause"
            " unexpected behavior. To achieve different scales, we recommend generating separate USD files for"
            " each version of the articulation and using multi-asset spawning. For more details, refer to:"
            " https://isaac-sim.github.io/IsaacLab/main/source/how-to/multi_asset_spawning.html"
        )

    env_ids = _resolve_env_ids(env, env_ids, "cpu")
    stage = env.sim.stage
    prim_paths = sim_utils.find_matching_prim_paths(asset.cfg.prim_path)

    if isinstance(scale_range, dict):
        rand_samples = _sample_uniform_ranges(scale_range, _POSITION_KEYS, len(env_ids), "cpu", default=(1.0, 1.0))
    else:
        rand_samples = math_utils.sample_uniform(*scale_range, (len(env_ids), 1), device="cpu").repeat(1, 3)
    rand_samples = rand_samples.tolist()

    # an empty child path scales the asset prim itself
    if relative_child_path is None:
        relative_child_path = ""
    elif not relative_child_path.startswith("/"):
        relative_child_path = "/" + relative_child_path

    # use an Sdf change block for faster processing of USD properties (pxr is only available with Kit)
    from pxr import Gf, Sdf, UsdGeom, Vt  # noqa: PLC0415

    with Sdf.ChangeBlock():
        for env_id, scale in zip(env_ids.tolist(), rand_samples):
            prim_path = prim_paths[env_id] + relative_child_path
            prim_spec = Sdf.CreatePrimInLayer(stage.GetRootLayer(), prim_path)

            scale_spec = prim_spec.GetAttributeAtPath(prim_path + ".xformOp:scale")
            has_scale_attr = scale_spec is not None
            if not has_scale_attr:
                scale_spec = Sdf.AttributeSpec(prim_spec, prim_path + ".xformOp:scale", Sdf.ValueTypeNames.Double3)
            scale_spec.default = Gf.Vec3f(*scale)

            # A newly created scale op must be appended to the transform stack. Assets authored through
            # Isaac Sim already follow this ordering, so existing scale attributes are left untouched.
            if not has_scale_attr:
                op_order_spec = prim_spec.GetAttributeAtPath(prim_path + ".xformOpOrder")
                if op_order_spec is None:
                    op_order_spec = Sdf.AttributeSpec(
                        prim_spec, UsdGeom.Tokens.xformOpOrder, Sdf.ValueTypeNames.TokenArray
                    )
                op_order_spec.default = Vt.TokenArray(["xformOp:translate", "xformOp:orient", "xformOp:scale"])


class _RandomizeRigidBodyMaterialPhysx:
    """PhysX backend implementation for material randomization.

    Uses the bucket-based approach required by PhysX's 64000 unique material limit.
    Materials are pre-sampled into buckets and randomly assigned to shapes.
    """

    def __init__(
        self, cfg: EventTermCfg, env: ManagerBasedEnv, asset: RigidObject | Articulation, asset_cfg: SceneEntityCfg
    ):
        from isaaclab.assets import BaseArticulation  # noqa: PLC0415

        # materials are sampled once; afterwards they are randomly assigned to the geometries of the asset
        self.material_buckets = _sample_material_buckets(cfg)
        self.asset = asset
        self.asset_cfg = asset_cfg

        # Articulations do not expose per-body shape counts directly, so they are read from per-link
        # PhysX views. ``body_ids`` are public IDs and are converted once to backend order.
        if isinstance(asset, BaseArticulation) and asset_cfg.body_ids != slice(None):
            self.num_shapes_per_body = [
                asset._physics_sim_view.create_rigid_body_view(link_path).max_shapes  # type: ignore
                for link_path in asset.root_view.link_paths[0]
            ]
            self._backend_body_ids = asset.map_body_ids_to_backend(asset_cfg.body_ids)
            num_shapes = sum(self.num_shapes_per_body)
            expected_shapes = asset.root_view.max_shapes
            if num_shapes != expected_shapes:
                raise ValueError(
                    "Randomization term 'randomize_rigid_body_material' failed to parse the number of shapes per body."
                    f" Expected total shapes: {expected_shapes}, but got: {num_shapes}."
                )
        else:
            self.num_shapes_per_body = None
            self._backend_body_ids = None

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        static_friction_range: tuple[float, float],
        dynamic_friction_range: tuple[float, float],
        restitution_range: tuple[float, float],
        num_buckets: int,
        asset_cfg: SceneEntityCfg,
        make_consistent: bool = False,
    ):
        env_ids = _resolve_env_ids(env, env_ids, "cpu", torch.int32)

        # randomly assign pre-sampled bucket materials to the geometries: (num_env_ids, total_num_shapes, 3)
        total_num_shapes = self.asset.root_view.max_shapes
        bucket_ids = torch.randint(0, num_buckets, (len(env_ids), total_num_shapes), device="cpu")
        material_samples = self.material_buckets[bucket_ids]

        materials = wp.to_torch(self.asset.root_view.get_material_properties())
        if self.num_shapes_per_body is not None:
            for body_id in self._backend_body_ids:
                start_idx = sum(self.num_shapes_per_body[:body_id])
                end_idx = start_idx + self.num_shapes_per_body[body_id]
                materials[env_ids, start_idx:end_idx] = material_samples[:, start_idx:end_idx]
        else:
            materials[env_ids] = material_samples

        self.asset.root_view.set_material_properties(
            wp.from_torch(materials, dtype=wp.float32), wp.from_torch(env_ids, dtype=wp.int32)
        )


class _RandomizeRigidBodyMaterialNewton:
    """Newton backend implementation for material randomization.

    Newton can assign arbitrary friction/restitution per shape (no bucket limitation).
    Samples friction (mu) and restitution continuously from the given ranges.
    Newton uses a single friction coefficient (mu), so ``dynamic_friction_range``
    and ``num_buckets`` are ignored.

    The Kamino solver deduplicates contact materials globally by ``(mu, restitution)`` and
    shares them across environments, so it cannot accept per-shape or per-env overrides. When
    Kamino is active, one value is sampled per build-time material group and broadcast to every
    environment (no per-env variation). All other Newton solvers keep the per-shape sampling.
    """

    def __init__(
        self, cfg: EventTermCfg, env: ManagerBasedEnv, asset: RigidObject | Articulation, asset_cfg: SceneEntityCfg
    ):
        import isaaclab_newton.physics.newton_manager as newton_manager_module  # noqa: PLC0415
        from isaaclab_newton.assets import Articulation as NewtonArticulation  # noqa: PLC0415
        from newton import ModelFlags  # noqa: PLC0415
        from newton.solvers import SolverKamino  # noqa: PLC0415

        self.asset = asset
        self.asset_cfg = asset_cfg
        self._newton_manager = newton_manager_module.NewtonManager
        self._notify_shape_properties = ModelFlags.SHAPE_PROPERTIES
        # Kamino deduplicates contact materials globally by (mu, restitution) at build time and
        # shares them across environments, so its in-place material update rejects per-shape /
        # per-env overrides. When Kamino is active we instead sample one value per build-time
        # material group and broadcast it to every environment. The grouping is derived lazily on
        # the first call, when the shape bindings still hold their build-time values.
        self._solver_kamino_cls = SolverKamino
        self._kamino_group_inverse: torch.Tensor | None = None
        self._kamino_num_groups = 0

        # cache friction/restitution ranges for continuous per-shape sampling
        self._static_friction_range = cfg.params.get("static_friction_range", (1.0, 1.0))
        self._restitution_range = cfg.params.get("restitution_range", (0.0, 0.0))

        # get friction/restitution view-level bindings
        model = self._newton_manager.get_model()
        self._friction_binding = asset._root_view.get_attribute("shape_material_mu", model)[:, 0]  # type: ignore
        self._restitution_binding = asset._root_view.get_attribute("shape_material_restitution", model)[:, 0]  # type: ignore

        # compute shape indices for body-specific randomization
        if isinstance(asset, NewtonArticulation) and asset_cfg.body_ids != slice(None):
            # ``body_ids`` are public IDs, while shape bindings use backend order; convert the
            # selected IDs once and index the backend-ordered shape counts directly.
            num_shapes_per_body = asset.backend_num_shapes_per_body
            shape_indices_list = []
            backend_body_ids = asset.map_body_ids_to_backend(asset_cfg.body_ids)
            for body_id in backend_body_ids:
                start_idx = sum(num_shapes_per_body[:body_id])
                end_idx = start_idx + num_shapes_per_body[body_id]
                shape_indices_list.extend(range(start_idx, end_idx))
            self._shape_indices = torch.tensor(shape_indices_list, dtype=torch.long)
        else:
            self._shape_indices = torch.arange(self._friction_binding.shape[1], dtype=torch.long)

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        static_friction_range: tuple[float, float],
        dynamic_friction_range: tuple[float, float],
        restitution_range: tuple[float, float],
        num_buckets: int,
        asset_cfg: SceneEntityCfg,
        make_consistent: bool = False,
    ):
        device = env.device
        env_ids = _resolve_env_ids(env, env_ids, device, torch.int32)
        num_shapes = len(self._shape_indices)
        shape_idx = self._shape_indices.to(device)

        friction_range = torch.tensor(self._static_friction_range, device=device)
        restitution_range_t = torch.tensor(self._restitution_range, device=device)
        friction_view = wp.to_torch(self._friction_binding)
        restitution_view = wp.to_torch(self._restitution_binding)

        if isinstance(self._newton_manager._solver, self._solver_kamino_cls):
            # Kamino: sample one value per build-time material group and broadcast across every
            # environment. Per-shape / per-env variation is impossible because Kamino shares each
            # contact material across all shapes and environments that were built with identical
            # (mu, restitution).
            if self._kamino_group_inverse is None:
                build_keys = torch.stack((friction_view[0, shape_idx], restitution_view[0, shape_idx]), dim=-1)
                _, inverse = torch.unique(build_keys, dim=0, return_inverse=True)
                self._kamino_group_inverse = inverse
                self._kamino_num_groups = int(inverse.max().item()) + 1 if inverse.numel() else 0
            inverse = self._kamino_group_inverse
            friction_groups = math_utils.sample_uniform(
                friction_range[0], friction_range[1], (self._kamino_num_groups,), device=device
            )
            restitution_groups = math_utils.sample_uniform(
                restitution_range_t[0], restitution_range_t[1], (self._kamino_num_groups,), device=device
            )
            friction_view[:, shape_idx] = friction_groups[inverse]
            restitution_view[:, shape_idx] = restitution_groups[inverse]
        else:
            # sample friction (mu) and restitution continuously per shape
            friction_samples = math_utils.sample_uniform(
                friction_range[0], friction_range[1], (len(env_ids), num_shapes), device=device
            )
            restitution_samples = math_utils.sample_uniform(
                restitution_range_t[0], restitution_range_t[1], (len(env_ids), num_shapes), device=device
            )
            # write only the affected env_ids to the warp binding
            friction_view[env_ids[:, None], shape_idx] = friction_samples
            restitution_view[env_ids[:, None], shape_idx] = restitution_samples

        # notify the physics engine
        self._newton_manager.add_model_change(self._notify_shape_properties)


def _is_all_body_selection(body_ids: list[int] | slice, num_bodies: int) -> bool:
    """Return whether a body selector covers the entire asset."""
    if body_ids == slice(None):
        return True
    return sorted(body_ids) == list(range(num_bodies))


class _RandomizeRigidBodyMaterialOvPhysx:
    """OVPhysX backend implementation for material randomization.

    OVPhysX runs the PhysX solver, so PhysX's 64000 unique-material limit applies and this
    mirrors the PhysX bucket approach: ``num_buckets`` materials are pre-sampled once and
    randomly assigned to shapes. Materials are written through the asset's
    :class:`~isaaclab_ov.sim.views.OvPhysxView` on the per-collision-shape
    ``shape_friction_and_restitution`` binding (shape ``[N, S, 3]`` = static friction,
    dynamic friction, restitution).

    Whole-articulation randomization uses the articulation material binding. For a
    body subset, individual articulation links are addressed through a rigid-body
    material binding, whose rows expose the exact link prim paths and per-link shape
    storage.
    """

    def __init__(
        self, cfg: EventTermCfg, env: ManagerBasedEnv, asset: RigidObject | Articulation, asset_cfg: SceneEntityCfg
    ):
        import isaaclab_ov.tensor_types as ovphysx_tt  # noqa: PLC0415
        from isaaclab_ov.sim.views.ovphysx_view import OvPhysxView  # noqa: PLC0415

        from isaaclab.assets import BaseArticulation  # noqa: PLC0415

        # sample material buckets once (PhysX-style; the 64000 unique-material limit applies)
        self.material_buckets = _sample_material_buckets(cfg)
        self.asset = asset
        self.asset_cfg = asset_cfg
        self._material_view = asset.root_view
        self._material_rows_by_env = torch.arange(asset.num_instances, dtype=torch.long).unsqueeze(-1)

        if isinstance(asset, BaseArticulation):
            self._material_type = ovphysx_tt.SHAPE_FRICTION_AND_RESTITUTION
            if not _is_all_body_selection(asset_cfg.body_ids, asset.num_bodies):
                from pxr import UsdPhysics  # noqa: PLC0415

                body_ids = [int(body_id) for body_id in asset_cfg.body_ids]
                if len(body_ids) == 0:
                    self._material_view = None
                    self._material_rows_by_env = torch.empty((asset.num_instances, 0), dtype=torch.long)
                    return

                selected_body_names = [asset.body_names[body_id] for body_id in body_ids]
                asset_root_paths = sim_utils.find_matching_prim_paths(asset.cfg.prim_path)
                articulation_root_paths = asset.root_view.prim_paths
                if len(articulation_root_paths) != asset.num_instances:
                    raise RuntimeError(
                        "Failed to map OVPhysX articulation material rows to asset instances: "
                        f"expected {asset.num_instances} articulation roots, got {len(articulation_root_paths)}."
                    )

                # With replicated physics, only the source asset may exist as a concrete
                # USD prim even though the tensor binding contains every environment. Find
                # the articulation-root suffix in that source asset, then strip the same
                # suffix from every binding row to recover its concrete asset root.
                source_pairs = [
                    (asset_root_path, articulation_root_path)
                    for asset_root_path in asset_root_paths
                    for articulation_root_path in articulation_root_paths
                    if articulation_root_path == asset_root_path
                    or articulation_root_path.startswith(f"{asset_root_path}/")
                ]
                if not source_pairs:
                    raise RuntimeError(
                        "Failed to find a source asset root containing an OVPhysX articulation root. "
                        f"Asset roots: {asset_root_paths}; articulation roots: {articulation_root_paths}."
                    )
                source_asset_root, source_articulation_root = max(source_pairs, key=lambda pair: len(pair[0]))
                articulation_root_suffix = source_articulation_root[len(source_asset_root) :]
                if articulation_root_suffix:
                    if not all(path.endswith(articulation_root_suffix) for path in articulation_root_paths):
                        raise RuntimeError(
                            "OVPhysX articulation roots do not share the source asset's relative root suffix "
                            f"'{articulation_root_suffix}': {articulation_root_paths}."
                        )
                    instance_root_paths = [path[: -len(articulation_root_suffix)] for path in articulation_root_paths]
                else:
                    instance_root_paths = articulation_root_paths

                selected_relative_paths = []
                for body_name in selected_body_names:

                    def is_selected_rigid_body(prim, expected_name=body_name):
                        return prim.GetName() == expected_name and prim.HasAPI(UsdPhysics.RigidBodyAPI)

                    source_matches = sim_utils.resolve_matching_prims_from_source(
                        asset.cfg.prim_path,
                        predicate=is_selected_rigid_body,
                        expected_num_matches=1,
                    )
                    source_body_path = source_matches[0][0].GetPath().pathString
                    if not (
                        source_body_path == source_asset_root or source_body_path.startswith(f"{source_asset_root}/")
                    ):
                        raise RuntimeError(
                            f"OVPhysX body '{body_name}' at '{source_body_path}' is not below source asset root "
                            f"'{source_asset_root}'."
                        )
                    selected_relative_paths.append(source_body_path[len(source_asset_root) :])

                selected_paths = []
                for instance_root_path in instance_root_paths:
                    selected_paths.extend(
                        f"{instance_root_path}{relative_path}" for relative_path in selected_relative_paths
                    )

                self._material_type = ovphysx_tt.RIGID_BODY_SHAPE_FRICTION_AND_RESTITUTION
                self._material_view = OvPhysxView(
                    asset._ovphysx,  # type: ignore[attr-defined]
                    prim_paths=selected_paths,
                    device=asset.device,
                )
                selected_binding = self._material_view.binding_for(self._material_type)
                resolved_paths = selected_binding.prim_paths
                if len(resolved_paths) != len(selected_paths) or set(resolved_paths) != set(selected_paths):
                    raise RuntimeError(
                        "OVPhysX rigid-body material binding did not resolve the requested articulation links. "
                        f"Requested {selected_paths}, resolved {resolved_paths}."
                    )
                row_by_path = {path: row for row, path in enumerate(resolved_paths)}
                self._material_rows_by_env = torch.tensor(
                    [row_by_path[path] for path in selected_paths], dtype=torch.long
                ).reshape(asset.num_instances, len(selected_body_names))
        else:
            self._material_type = ovphysx_tt.RIGID_BODY_SHAPE_FRICTION_AND_RESTITUTION
            if not _is_all_body_selection(asset_cfg.body_ids, asset.num_bodies):
                raise NotImplementedError(
                    "randomize_rigid_body_material on the OVPhysX backend cannot apply per-body selection to a "
                    "standalone rigid object. Use the default body selection."
                )

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        static_friction_range: tuple[float, float],
        dynamic_friction_range: tuple[float, float],
        restitution_range: tuple[float, float],
        num_buckets: int,
        asset_cfg: SceneEntityCfg,
        make_consistent: bool = False,
    ):
        if self._material_view is None:
            return

        view = self._material_view
        # read the current per-shape material [N, S, 3] on the binding's native CPU device
        materials = wp.to_torch(view.get_attribute(self._material_type))
        num_shapes = materials.shape[1]

        # Resolve environment ids to rows of the active material binding. A subset
        # articulation view contains one row per selected body and environment.
        if env_ids is None:
            material_rows = self._material_rows_by_env.flatten()
        else:
            material_rows = self._material_rows_by_env[env_ids.to(device="cpu", dtype=torch.long)].flatten()
        if material_rows.numel() == 0:
            return
        material_rows_device = material_rows.to(materials.device)

        # randomly assign pre-sampled bucket materials to every shape of the selected envs
        bucket_ids = torch.randint(0, num_buckets, (len(material_rows), num_shapes), device="cpu")
        material_samples = self.material_buckets[bucket_ids].to(materials.device)
        materials[material_rows_device] = material_samples

        # The wheel requires a full-shaped source buffer even for indexed writes.
        indices = wp.from_torch(material_rows_device.to(dtype=torch.int32))
        view.set_attribute(
            self._material_type,
            wp.from_torch(materials.contiguous(), dtype=wp.float32),
            indices=indices,
        )


class randomize_rigid_body_material(ManagerTermBase):
    """Randomize the physics materials on all geometries of the asset.

    This function creates a set of physics materials with random static friction, dynamic friction, and restitution
    values and assigns them to the geometries of the asset.

    For articulations, :attr:`SceneEntityCfg.body_ids` selects bodies in public articulation order. The backend
    implementations convert those IDs to backend shape ranges; callers must not pre-swizzle body IDs.

    Automatically detects the active physics backend (PhysX, Newton, or OVPhysX) and delegates
    to the appropriate backend-specific implementation:

    - **PhysX**: Uses the 3-tuple material format (static_friction, dynamic_friction, restitution)
      with bucket-based assignment (limited to 64000 unique materials). Applied via the PhysX
      tensor API (``root_view.set_material_properties``).
    - **Newton**: Samples friction (mu) and restitution continuously per shape (no bucket
      limitation). Newton uses a single friction coefficient, so ``dynamic_friction_range``
      and ``num_buckets`` are ignored. Applied directly to Newton's view-level bindings. The
      Kamino solver shares contact materials across shapes and environments, so it instead
      samples one value per build-time material group and broadcasts it to every environment.
    - **OVPhysX**: Runs the PhysX solver, so the same 3-tuple, bucket-based assignment is used,
      written through the :class:`~isaaclab_ov.sim.views.OvPhysxView` on the per-shape
      ``shape_friction_and_restitution`` binding. Articulation body subsets are addressed through
      rigid-body material bindings for the selected links.

    If the flag ``make_consistent`` is set to ``True``, the dynamic friction is set to be less than or equal to
    the static friction (PhysX and OVPhysX only). This obeys the physics constraint on friction values.

    .. attention::
        On PhysX, this function uses CPU tensors to assign the material properties. It is recommended to
        use this function only during the initialization of the environment.

    .. note::
        PhysX only allows 64000 unique physics materials in the scene. If the number of materials exceeds this
        limit, the simulation will crash. Due to this reason, we sample the materials only once during initialization.
        Afterwards, these materials are randomly assigned to the geometries of the asset.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the event term.
            env: The environment instance.

        Raises:
            ValueError: If the asset is not a RigidObject or an Articulation.
        """
        from isaaclab.assets import BaseArticulation, BaseRigidObject

        super().__init__(cfg, env)

        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: RigidObject | Articulation = env.scene[self.asset_cfg.name]

        if not isinstance(self.asset, (BaseRigidObject, BaseArticulation)):
            raise ValueError(
                f"Randomization term 'randomize_rigid_body_material' not supported for asset: '{self.asset_cfg.name}'"
                f" with type: '{type(self.asset)}'."
            )

        # detect physics backend and instantiate the appropriate implementation.
        # Check ``ovphysxmanager`` first: it contains the substring ``physx`` so
        # would otherwise be caught by the ``"physx" in ...`` branch below and
        # routed to the PhysX impl, which assumes a ``root_view`` with
        # ``.link_paths`` — OVPhysX's per-tensor-type bindings dict does not
        # satisfy that contract.  Newton's subclasses (``NewtonMJWarpManager``,
        # ``NewtonKaminoManager``, ...) are caught by the substring branch.
        manager_name = env.sim.physics_manager.__name__.lower()
        if manager_name == "ovphysxmanager":
            self._impl = _RandomizeRigidBodyMaterialOvPhysx(cfg, env, self.asset, self.asset_cfg)
        elif "newton" in manager_name:
            self._impl = _RandomizeRigidBodyMaterialNewton(cfg, env, self.asset, self.asset_cfg)
        elif "physx" in manager_name:
            self._impl = _RandomizeRigidBodyMaterialPhysx(cfg, env, self.asset, self.asset_cfg)
        else:
            raise ValueError(f"Unsupported physics manager for randomize_rigid_body_material: {manager_name!r}")

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        static_friction_range: tuple[float, float],
        dynamic_friction_range: tuple[float, float],
        restitution_range: tuple[float, float],
        num_buckets: int,
        asset_cfg: SceneEntityCfg,
        make_consistent: bool = False,
    ):
        self._impl(
            env,
            env_ids,
            static_friction_range,
            dynamic_friction_range,
            restitution_range,
            num_buckets,
            asset_cfg,
            make_consistent,
        )


class randomize_rigid_body_mass(ManagerTermBase):
    """Randomize the mass of the bodies by adding, scaling, or setting random values.

    This function allows randomizing the mass of the bodies of the asset. The function samples random
    values from the given distribution parameters and adds, scales, or sets the values into the physics
    simulation based on the operation.

    If the :attr:`recompute_inertia` flag is set to :obj:`True`, the function recomputes the inertia tensor
    of the bodies after setting the mass. This is useful when the mass is changed significantly, as the
    inertia tensor depends on the mass. It assumes the body is a uniform density object. If the body is not
    a uniform density object, the inertia tensor may not be accurate.

    .. tip::
        This function uses CPU tensors to assign the body masses. It is recommended to use this function
        only during the initialization of the environment.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the event term.
            env: The environment instance.

        Raises:
            TypeError: If `params` is not a tuple of two numbers.
            ValueError: If the operation is not supported.
            ValueError: If the lower bound is negative or zero when not allowed.
            ValueError: If the upper bound is less than the lower bound.
        """
        super().__init__(cfg, env)

        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: RigidObject | Articulation = env.scene[self.asset_cfg.name]
        _check_operation("randomize_rigid_body_mass", cfg, {"mass_distribution_params": False})
        min_mass = cfg.params.get("min_mass")
        if min_mass is not None and min_mass < 1e-6:
            raise ValueError(
                "Randomization term 'randomize_rigid_body_mass' does not support 'min_mass' less than 1e-6 to avoid"
                " physics errors."
            )

        self.default_mass = None
        self.default_inertia = None

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        mass_distribution_params: tuple[float, float],
        operation: Literal["add", "scale", "abs"],
        distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
        recompute_inertia: bool = True,
        min_mass: float = 1e-6,
    ):
        # cache defaults on the first call so repeated randomization starts from the same values
        if self.default_mass is None:
            self.default_mass = self.asset.data.body_mass.torch.clone()
        if self.default_inertia is None:
            self.default_inertia = self.asset.data.body_inertia.torch.clone()

        env_ids = _resolve_env_ids(env, env_ids, self.asset.device, torch.int32)
        body_ids = _resolve_body_ids(self.asset, self.asset_cfg.body_ids)
        selection = (env_ids[:, None], body_ids)

        # randomize the default masses of the selected bodies, shape (len(env_ids), len(body_ids))
        masses = self.default_mass[selection]
        masses = _randomize_prop_by_op(
            masses, mass_distribution_params, None, slice(None), operation=operation, distribution=distribution
        )
        masses.clamp_(min=min_mass)
        self.asset.set_masses_index(masses=masses, body_ids=body_ids, env_ids=env_ids)

        if recompute_inertia:
            # scale the default inertia tensors by the mass ratios, shape (len(env_ids), len(body_ids), 9)
            ratios = masses / self.default_mass[selection]
            inertias = self.default_inertia[selection] * ratios[..., None]
            self.asset.set_inertias_index(inertias=inertias, body_ids=body_ids, env_ids=env_ids)


class randomize_rigid_body_inertia(ManagerTermBase):
    """Randomize the inertia tensor of rigid bodies by adding, scaling, or setting values.

    This function modifies body inertia tensors independently of mass. The inertia tensor
    is a 3x3 symmetric matrix stored as 9 elements: ``[Ixx, Ixy, Ixz, Iyx, Iyy, Iyz, Izx, Izy, Izz]``.

    Two modes are supported via the :attr:`diagonal_only` parameter:

    - **diagonal_only=True** (default): Only modifies diagonal elements (Ixx, Iyy, Izz at
      indices 0, 4, 8). This is useful for adding numerical stability (armature/regularization)
      without changing rotational coupling between axes. The diagonal elements represent
      resistance to rotation about each principal axis.

    - **diagonal_only=False**: Modifies all 9 elements of the inertia tensor. This can
      simulate manufacturing variations or asymmetric mass distributions. Off-diagonal
      elements represent coupling between rotations about different axes.

    .. note::
        Unlike :class:`randomize_rigid_body_mass` which recomputes inertia based on mass
        ratios, this function modifies inertia directly without affecting mass.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the event term.
            env: The environment instance.

        Raises:
            ValueError: If the operation is not supported.
            ValueError: If the lower bound is negative or zero when not allowed for scale operation.
            ValueError: If the upper bound is less than the lower bound.
        """
        from isaaclab.assets import BaseArticulation, BaseRigidObject, BaseRigidObjectCollection

        super().__init__(cfg, env)

        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: RigidObject | Articulation = env.scene[self.asset_cfg.name]

        if not isinstance(self.asset, (BaseRigidObject, BaseArticulation, BaseRigidObjectCollection)):
            raise ValueError(
                f"Randomization term 'randomize_rigid_body_inertia' not supported for asset: '{self.asset_cfg.name}'"
                f" with type: '{type(self.asset)}'."
            )
        _check_operation("randomize_rigid_body_inertia", cfg, {"inertia_distribution_params": False})

        self.default_inertia = None
        # cache inertia indices: diagonal (0, 4, 8) for regularization, or all elements
        diagonal_only = cfg.params.get("diagonal_only", True)
        self._inertia_idx = torch.tensor([0, 4, 8], device=self.asset.device) if diagonal_only else slice(None)

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        inertia_distribution_params: tuple[float, float],
        operation: Literal["add", "scale", "abs"] = "add",
        distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
        diagonal_only: bool = True,
    ):
        """Randomize body inertia tensors.

        Args:
            env: The environment instance.
            env_ids: The environment indices to randomize. If None, all environments are randomized.
            asset_cfg: The asset configuration specifying the asset and body names.
            inertia_distribution_params: Distribution parameters as a tuple of two floats
                ``(low, high)`` for sampling inertia modification values.
            operation: The operation to apply. Options: ``"add"``, ``"scale"``, ``"abs"``.
                Defaults to ``"add"`` which is typical for regularization/armature.
            distribution: The distribution to sample from. Options: ``"uniform"``,
                ``"log_uniform"``, ``"gaussian"``. Defaults to ``"uniform"``.
            diagonal_only: If True, only modify diagonal elements (Ixx, Iyy, Izz) for
                numerical stability. If False, modify all 9 elements. Defaults to True.
        """
        # cache defaults on the first call so repeated randomization starts from the same values
        if self.default_inertia is None:
            self.default_inertia = self.asset.data.body_inertia.torch.clone()

        env_ids = _resolve_env_ids(env, env_ids, self.asset.device, torch.int32)
        body_ids = _resolve_body_ids(self.asset, self.asset_cfg.body_ids)

        # advanced indexing copies the defaults, shape (len(env_ids), len(body_ids), 9)
        inertias = self.default_inertia[env_ids[:, None], body_ids]

        # one random value per (env, body) is shared by all selected inertia elements of that body
        dist_fn = _resolve_distribution_fn(distribution)
        random_values = dist_fn(*inertia_distribution_params, (len(env_ids), len(body_ids)), device=self.asset.device)
        if operation == "add":
            inertias[:, :, self._inertia_idx] += random_values[..., None]
        elif operation == "scale":
            inertias[:, :, self._inertia_idx] *= random_values[..., None]
        elif operation == "abs":
            inertias[:, :, self._inertia_idx] = random_values[..., None]

        self.asset.set_inertias_index(inertias=inertias, body_ids=body_ids, env_ids=env_ids)


class randomize_rigid_body_com(ManagerTermBase):
    """Randomize the center of mass (CoM) of rigid bodies by adding a random value sampled from the given ranges.

    This class tracks the original CoM values and randomizes from those defaults on each call,
    ensuring repeatable randomization across resets.

    Automatically detects the active physics backend:

    - **PhysX**: Passes the full CoM pose (position + quaternion) to ``set_coms_index``.
    - **Newton**: Passes position-only (vec3) to ``set_coms_index``. Note that on Newton
      (MuJoCo Warp), runtime CoM changes may cause simulation instability because
      ``notify_model_changed(BODY_INERTIAL_PROPERTIES)`` does not fully recompute the
      mass matrix after ``body_ipos`` changes. Use with caution until this is fixed upstream.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the event term.
            env: The environment instance.
        """
        super().__init__(cfg, env)

        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: RigidObject | Articulation = env.scene[self.asset_cfg.name]
        self._is_newton = "newton" in env.sim.physics_manager.__name__.lower()
        self.default_com = None

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        com_range: dict[str, tuple[float, float]],
        asset_cfg: SceneEntityCfg,
    ):
        # cache defaults on the first call so repeated randomization starts from the same values
        if self.default_com is None:
            self.default_com = self.asset.data.body_com_pose_b.torch.clone()

        env_ids = _resolve_env_ids(env, env_ids, self.asset.device)
        body_ids = _resolve_body_ids(self.asset, self.asset_cfg.body_ids)

        # offset the default CoM positions with one sample per environment, shared by the selected bodies
        rand_samples = _sample_uniform_ranges(com_range, _POSITION_KEYS, len(env_ids), self.asset.device)
        coms = self.default_com[env_ids[:, None], body_ids]
        coms[..., :3] += rand_samples.unsqueeze(1)

        # Newton expects position-only (vec3f), PhysX expects the full pose (pos + quat)
        if self._is_newton:
            coms = coms[..., :3].contiguous()
        self.asset.set_coms_index(coms=coms, body_ids=body_ids, env_ids=env_ids)


class _RandomizeRigidBodyColliderOffsetsPhysx:
    """PhysX backend implementation for collider offset randomization.

    Uses rest offset and contact offset directly via the PhysX tensor API.
    """

    def __init__(self, asset: RigidObject | Articulation):
        self.asset = asset
        self.default_rest_offsets = wp.to_torch(asset.root_view.get_rest_offsets()).clone()
        self.default_contact_offsets = wp.to_torch(asset.root_view.get_contact_offsets()).clone()

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        rest_offset_distribution_params: tuple[float, float] | None = None,
        contact_offset_distribution_params: tuple[float, float] | None = None,
        distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
    ):
        env_ids = _resolve_env_ids(env, env_ids, "cpu", torch.int32)
        wp_env_ids = wp.from_torch(env_ids, dtype=wp.int32)

        if rest_offset_distribution_params is not None:
            rest_offset = _sample_absolute(self.default_rest_offsets, rest_offset_distribution_params, distribution)
            self.asset.root_view.set_rest_offsets(wp.from_torch(rest_offset), wp_env_ids)

        if contact_offset_distribution_params is not None:
            contact_offset = _sample_absolute(
                self.default_contact_offsets, contact_offset_distribution_params, distribution
            )
            self.asset.root_view.set_contact_offsets(wp.from_torch(contact_offset), wp_env_ids)


class _RandomizeRigidBodyColliderOffsetsOvPhysx:
    """OVPhysX backend implementation for collider offset randomization.

    OVPhysX runs the PhysX solver, so rest and contact offsets are written directly, per collision
    shape, through the asset's :class:`~isaaclab_ov.sim.views.OvPhysxView`. Articulations use the
    articulation offset bindings and rigid objects the rigid-body ones; both are CPU-resident
    ``[N, S]`` buffers, so the full tensor is read-modify-written on the host with the selected
    environments as write indices.
    """

    def __init__(self, asset: RigidObject | Articulation):
        import isaaclab_ov.tensor_types as ovphysx_tt  # noqa: PLC0415

        from isaaclab.assets import BaseArticulation  # noqa: PLC0415

        self.asset = asset
        if isinstance(asset, BaseArticulation):
            self._rest_offset_type = ovphysx_tt.REST_OFFSET
            self._contact_offset_type = ovphysx_tt.CONTACT_OFFSET
        else:
            self._rest_offset_type = ovphysx_tt.RIGID_BODY_REST_OFFSET
            self._contact_offset_type = ovphysx_tt.RIGID_BODY_CONTACT_OFFSET
        self.default_rest_offsets = wp.to_torch(asset.root_view.get_attribute(self._rest_offset_type)).clone()
        self.default_contact_offsets = wp.to_torch(asset.root_view.get_attribute(self._contact_offset_type)).clone()

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        rest_offset_distribution_params: tuple[float, float] | None = None,
        contact_offset_distribution_params: tuple[float, float] | None = None,
        distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
    ):
        env_ids = _resolve_env_ids(env, env_ids, "cpu", torch.int32)
        wp_env_ids = wp.from_torch(env_ids, dtype=wp.int32)

        # the wheel requires a full-shaped source buffer even for indexed writes
        if rest_offset_distribution_params is not None:
            rest_offset = _sample_absolute(self.default_rest_offsets, rest_offset_distribution_params, distribution)
            self.asset.root_view.set_attribute(
                self._rest_offset_type, wp.from_torch(rest_offset.contiguous(), dtype=wp.float32), indices=wp_env_ids
            )

        if contact_offset_distribution_params is not None:
            contact_offset = _sample_absolute(
                self.default_contact_offsets, contact_offset_distribution_params, distribution
            )
            self.asset.root_view.set_attribute(
                self._contact_offset_type,
                wp.from_torch(contact_offset.contiguous(), dtype=wp.float32),
                indices=wp_env_ids,
            )


class _RandomizeRigidBodyColliderOffsetsNewton:
    """Newton backend implementation for collider offset randomization.

    Maps PhysX concepts to Newton's geometry properties:

    - ``rest_offset`` -> ``shape_margin`` (Newton margin)
    - ``contact_offset`` -> ``shape_gap`` (Newton gap = contact_offset - margin)

    See the `Newton collision schema`_ for details.

    .. _Newton collision schema: https://newton-physics.github.io/newton/latest/concepts/collisions.html
    """

    def __init__(self, asset: RigidObject | Articulation):
        import isaaclab_newton.physics.newton_manager as newton_manager_module  # noqa: PLC0415
        from newton import ModelFlags  # noqa: PLC0415

        self.asset = asset
        self._newton_manager = newton_manager_module.NewtonManager
        self._notify_shape_properties = ModelFlags.SHAPE_PROPERTIES

        model = self._newton_manager.get_model()
        self._sim_bind_shape_margin = asset._root_view.get_attribute("shape_margin", model)[:, 0]  # type: ignore
        self._sim_bind_shape_gap = asset._root_view.get_attribute("shape_gap", model)[:, 0]  # type: ignore

        self.default_margin = wp.to_torch(self._sim_bind_shape_margin).clone()
        self.default_gap = wp.to_torch(self._sim_bind_shape_gap).clone()

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        rest_offset_distribution_params: tuple[float, float] | None = None,
        contact_offset_distribution_params: tuple[float, float] | None = None,
        distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
    ):
        env_ids = _resolve_env_ids(env, env_ids, env.device, torch.int32)

        if rest_offset_distribution_params is not None:
            margin = _sample_absolute(self.default_margin, rest_offset_distribution_params, distribution)
            self.default_margin[env_ids] = margin[env_ids]
            wp.to_torch(self._sim_bind_shape_margin)[env_ids] = margin[env_ids]

        if contact_offset_distribution_params is not None:
            contact_offset = _sample_absolute(self.default_gap, contact_offset_distribution_params, distribution)
            gap = torch.clamp(contact_offset - self.default_margin, min=0.0)
            self.default_gap[env_ids] = gap[env_ids]
            wp.to_torch(self._sim_bind_shape_gap)[env_ids] = gap[env_ids]

        if rest_offset_distribution_params is not None or contact_offset_distribution_params is not None:
            self._newton_manager.add_model_change(self._notify_shape_properties)


class randomize_rigid_body_collider_offsets(ManagerTermBase):
    """Randomize the collider parameters of rigid bodies by setting random values.

    This function allows randomizing the collider parameters of the asset, such as rest and contact offsets.
    These correspond to the physics engine collider properties that affect collision checking.

    Automatically detects the active physics backend (PhysX, OVPhysX or Newton) and delegates to
    the appropriate backend-specific implementation:

    - **PhysX**: Uses rest offset and contact offset directly via the PhysX tensor API
      (``root_view.set_rest_offsets`` / ``root_view.set_contact_offsets``).
    - **OVPhysX**: Uses rest offset and contact offset directly, written per collision shape
      through the asset's :class:`~isaaclab_ov.sim.views.OvPhysxView`.
    - **Newton**: Maps PhysX concepts to Newton's geometry properties. PhysX ``rest_offset``
      maps to Newton ``shape_margin``, and PhysX ``contact_offset`` is converted to Newton
      ``shape_gap`` via ``gap = contact_offset - margin``.

    The function samples random values from the given distribution parameters and applies them
    as absolute values to the collider properties. If the distribution parameters are not
    provided for a particular property, the function does not modify it.

    .. tip::
        This function uses CPU tensors (PhysX, OVPhysX) or GPU tensors (Newton) to assign the collision
        properties. It is recommended to use this function only during the initialization of
        the environment.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the event term.
            env: The environment instance.

        Raises:
            ValueError: If the asset is not a RigidObject or an Articulation.
        """
        from isaaclab.assets import BaseArticulation, BaseRigidObject

        super().__init__(cfg, env)

        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: RigidObject | Articulation = env.scene[self.asset_cfg.name]

        if not isinstance(self.asset, (BaseRigidObject, BaseArticulation)):
            raise ValueError(
                f"Randomization term 'randomize_rigid_body_collider_offsets' not supported for asset:"
                f" '{self.asset_cfg.name}' with type: '{type(self.asset)}'."
            )

        # detect physics backend and instantiate the appropriate implementation.
        # Check ``ovphysxmanager`` first: it contains the substring ``physx`` so would otherwise
        # be routed to the PhysX impl, whose ``root_view`` offset accessors do not exist on
        # OVPhysX's ``OvPhysxView`` (see ``randomize_rigid_body_material``).
        manager_name = env.sim.physics_manager.__name__.lower()
        if manager_name == "ovphysxmanager":
            self._impl = _RandomizeRigidBodyColliderOffsetsOvPhysx(self.asset)
        elif "newton" in manager_name:
            self._impl = _RandomizeRigidBodyColliderOffsetsNewton(self.asset)
        elif "physx" in manager_name:
            self._impl = _RandomizeRigidBodyColliderOffsetsPhysx(self.asset)
        else:
            raise ValueError(f"Unsupported physics manager for randomize_rigid_body_collider_offsets: {manager_name!r}")

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        rest_offset_distribution_params: tuple[float, float] | None = None,
        contact_offset_distribution_params: tuple[float, float] | None = None,
        distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
    ):
        self._impl(
            env,
            env_ids,
            asset_cfg,
            rest_offset_distribution_params,
            contact_offset_distribution_params,
            distribution,
        )


class randomize_physics_scene_gravity(ManagerTermBase):
    """Randomize gravity by adding, scaling, or setting random values.

    Automatically detects the active physics backend (PhysX, OvPhysX, or Newton) and applies
    the appropriate gravity randomization strategy:

    - **PhysX**: samples a single gravity vector and sets it scene-wide via the PhysX
      simulation view.  All environments share the same gravity.
    - **OvPhysX**: samples a single gravity vector and applies a sealed OvStage control
      update. All environments share the same gravity.
    - **Newton**: samples per-environment gravity vectors and writes them in-place to
      the Newton model's per-world gravity array on GPU.

    The distribution parameters are tuples of two lists with three floats each,
    representing the lower and upper bounds for the x, y, and z gravity components [m/s^2].

    Args:
        cfg: The configuration of the event term.
        env: The environment instance.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._last_gravity_params: tuple | None = None

        manager_name = env.sim.physics_manager.__name__.lower()
        if "newton" in manager_name:
            self._backend = "newton"
            self._init_newton(cfg, env)
        elif "ovphysx" in manager_name:
            self._backend = "ovphysx"
            self._init_ovphysx(env)
        else:
            self._backend = "physx"
            self._init_physx(env)

        self._distribution = cfg.params.get("distribution", "uniform")
        self._dist_fn = _resolve_distribution_fn(self._distribution)

        operation = cfg.params["operation"]
        if operation not in ("add", "scale", "abs"):
            raise NotImplementedError(
                f"Unknown operation: '{operation}' for gravity randomization. Please use 'add', 'scale', or 'abs'."
            )

        gravity_distribution_params = cfg.params["gravity_distribution_params"]
        self._dist_param_0 = torch.tensor(gravity_distribution_params[0], device=env.device, dtype=torch.float32)
        self._dist_param_1 = torch.tensor(gravity_distribution_params[1], device=env.device, dtype=torch.float32)

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        gravity_distribution_params: tuple[list[float], list[float]],
        operation: Literal["add", "scale", "abs"],
        distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
    ):
        """Randomize gravity for the specified environments.

        Args:
            env: The environment instance.
            env_ids: The environment IDs to randomize. If None, all environments are randomized.
            gravity_distribution_params: Distribution parameters as a tuple of two lists, each
                with 3 floats corresponding to (x, y, z) gravity components [m/s^2]. Updated
                into pre-allocated tensors each call to support curriculum-driven range changes.
            operation: The operation to apply ('add', 'scale', or 'abs').
            distribution: The distribution type (cached at init, param ignored at runtime).
        """
        # rewrite the baked device tensors only when the curriculum-driven ranges change
        params = (tuple(gravity_distribution_params[0]), tuple(gravity_distribution_params[1]))
        if params != self._last_gravity_params:
            self._last_gravity_params = params
            self._dist_param_0.copy_(torch.tensor(params[0], dtype=torch.float32))
            self._dist_param_1.copy_(torch.tensor(params[1], dtype=torch.float32))

        if self._backend == "newton":
            self._call_newton(env, env_ids, operation)
        elif self._backend == "ovphysx":
            self._ovphysx_manager.set_gravity(tuple(self._sample_scene_gravity(env, operation)))
        else:
            self._physics_sim_view.set_gravity(self._carb.Float3(*self._sample_scene_gravity(env, operation)))

    def _init_newton(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        """Cache Newton manager reference and solver notification flag."""
        import isaaclab_newton.physics.newton_manager as newton_manager_module  # noqa: PLC0415
        from newton import ModelFlags  # noqa: PLC0415

        self._newton_manager = newton_manager_module.NewtonManager
        self._notify_model_properties = ModelFlags.MODEL_PROPERTIES

    def _call_newton(self, env: ManagerBasedEnv, env_ids: torch.Tensor | None, operation: str):
        """Apply per-environment gravity via Newton's per-world gravity array on GPU."""
        model = self._newton_manager.get_model()
        if model is None or model.gravity is None:
            raise RuntimeError("Newton model is not initialized. Cannot randomize gravity.")

        gravity = wp.to_torch(model.gravity)
        if env_ids is None:
            env_ids = env.scene._ALL_INDICES
        num = len(env_ids)
        if num == 0:
            return

        random_values = self._dist_fn(
            self._dist_param_0.unsqueeze(0).expand(num, -1),
            self._dist_param_1.unsqueeze(0).expand(num, -1),
            (num, 3),
            device=env.device,
        )
        if operation == "abs":
            gravity[env_ids] = random_values
        elif operation == "add":
            gravity[env_ids] += random_values
        elif operation == "scale":
            gravity[env_ids] *= random_values

        self._newton_manager.add_model_change(self._notify_model_properties)

    def _init_physx(self, env: ManagerBasedEnv):
        """Cache the ``carb`` module and PhysX simulation view for scene-wide gravity updates."""
        import carb  # noqa: PLC0415

        self._carb = carb
        self._physics_sim_view = sim_utils.SimulationContext.instance().physics_sim_view

    def _init_ovphysx(self, env: ManagerBasedEnv):
        """Cache the OvPhysX manager for scene-wide gravity updates."""
        self._ovphysx_manager = env.sim.physics_manager

    def _sample_scene_gravity(self, env: ManagerBasedEnv, operation: str) -> list[float]:
        """Sample a single scene-wide gravity vector [m/s^2] from the configured default gravity."""
        gravity = torch.tensor(env.sim.cfg.gravity, device="cpu").unsqueeze(0)
        gravity = _randomize_prop_by_op(
            gravity,
            (self._dist_param_0.cpu(), self._dist_param_1.cpu()),
            None,
            slice(None),
            operation=operation,
            distribution=self._distribution,
        )
        return gravity[0].tolist()


class randomize_actuator_gains(ManagerTermBase):
    """Randomize the actuator gains in an articulation by adding, scaling, or setting random values.

    This function allows randomizing the actuator stiffness and damping gains.

    The function samples random values from the given distribution parameters and applies the operation to
    the joint properties. It then sets the values into the actuator models. If the distribution parameters
    are not provided for a particular property, the function does not modify the property.

    .. tip::
        For implicit actuators, this function uses CPU tensors to assign the actuator gains into the simulation.
        In such cases, it is recommended to use this function only during the initialization of the environment.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the event term.
            env: The environment instance.

        Raises:
            TypeError: If `params` is not a tuple of two numbers.
            ValueError: If the operation is not supported.
            ValueError: If the lower bound is negative or zero when not allowed.
            ValueError: If the upper bound is less than the lower bound.
        """
        super().__init__(cfg, env)

        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: RigidObject | Articulation = env.scene[self.asset_cfg.name]

        self.default_joint_stiffness = self.asset.data.joint_stiffness.torch.clone()
        self.default_joint_damping = self.asset.data.joint_damping.torch.clone()

        # Ownership decides the gain source and write path per group: implicit groups are
        # articulation-owned, Newton-executed groups are controller-owned (their mapping
        # entries are the Newton actuator objects), and Lab explicit groups own their tensors.
        from isaaclab.actuators import IdealPDActuator  # noqa: PLC0415

        collection = self.asset.actuators
        self._native_group_names = getattr(collection, "_native_group_names", set())
        self._gain_actuators = {
            name: actuator
            for name, actuator in collection.items()
            if name in self._native_group_names
            or getattr(actuator, "is_implicit_model", False)
            or isinstance(actuator, IdealPDActuator)
        }
        group_joint_indices = getattr(collection, "_group_joint_indices", None)
        self._group_joint_indices = {
            name: (group_joint_indices[name] if group_joint_indices is not None else actuator.joint_indices)
            for name, actuator in self._gain_actuators.items()
        }
        self.default_actuator_stiffness: dict[str, torch.Tensor] = {}
        self.default_actuator_damping: dict[str, torch.Tensor] = {}
        from isaaclab.actuators.newton import read_group_parameter  # noqa: PLC0415

        for name, actuator in self._gain_actuators.items():
            joint_ids = self._group_joint_indices[name]
            if name in self._native_group_names:
                stiffness = read_group_parameter(collection, name, "controller", "kp")
                damping = read_group_parameter(collection, name, "controller", "kd")
            else:
                stiffness = actuator.stiffness
                damping = actuator.damping
            if not getattr(actuator, "is_implicit_model", False):
                # Explicit and Newton PD gains replace the zeroed solver gains in the defaults.
                self.default_joint_stiffness[:, joint_ids] = stiffness
                self.default_joint_damping[:, joint_ids] = damping
            self.default_actuator_stiffness[name] = stiffness.clone()
            self.default_actuator_damping[name] = damping.clone()

        _check_operation(
            "randomize_actuator_gains",
            cfg,
            {"stiffness_distribution_params": False, "damping_distribution_params": True},
        )

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        stiffness_distribution_params: tuple[float, float] | None = None,
        damping_distribution_params: tuple[float, float] | None = None,
        operation: Literal["add", "scale", "abs"] = "abs",
        distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
    ):
        from isaaclab.actuators.newton import write_group_parameter  # noqa: PLC0415

        env_ids = _resolve_env_ids(env, env_ids, self.asset.device)
        gains = (
            ("stiffness", "kp", stiffness_distribution_params, self.default_actuator_stiffness),
            ("damping", "kd", damping_distribution_params, self.default_actuator_damping),
        )

        for actuator_name, actuator in self._gain_actuators.items():
            group_joint_indices = self._group_joint_indices[actuator_name]
            # ``actuator_indices`` select columns within the actuator group; ``global_indices`` are asset joints
            if isinstance(self.asset_cfg.joint_ids, slice):
                actuator_indices = slice(None)
                if isinstance(group_joint_indices, slice):
                    global_indices = slice(None)
                elif isinstance(group_joint_indices, torch.Tensor):
                    global_indices = group_joint_indices.to(self.asset.device)
                else:
                    raise TypeError("Actuator joint indices must be a slice or a torch.Tensor.")
            elif isinstance(group_joint_indices, slice):
                global_indices = actuator_indices = torch.tensor(self.asset_cfg.joint_ids, device=self.asset.device)
            else:
                # intersect the actuator joints with the asset config joints
                asset_joint_ids = torch.tensor(self.asset_cfg.joint_ids, device=self.asset.device)
                actuator_indices = torch.nonzero(torch.isin(group_joint_indices, asset_joint_ids)).view(-1)
                if len(actuator_indices) == 0:
                    continue
                global_indices = group_joint_indices[actuator_indices]
            if isinstance(global_indices, slice):
                writer_joint_ids = torch.arange(self.asset.num_joints, device=self.asset.device, dtype=torch.long)
            else:
                writer_joint_ids = global_indices.to(device=self.asset.device, dtype=torch.long)
            is_native = actuator_name in self._native_group_names
            is_implicit = getattr(actuator, "is_implicit_model", False)
            # Native group writes are group-targeted: they take positions within the group's joints.
            group_columns = None if isinstance(actuator_indices, slice) else actuator_indices

            for gain_name, native_key, params, defaults in gains:
                if params is None:
                    continue
                if is_native:
                    # Native gains are controller-owned; randomization always starts from the defaults.
                    values = defaults[actuator_name][env_ids].clone()
                else:
                    values = getattr(actuator, gain_name)[env_ids].clone()
                    values[:, actuator_indices] = defaults[actuator_name][env_ids][:, actuator_indices]
                _randomize_prop_by_op(
                    values, params, None, actuator_indices, operation=operation, distribution=distribution
                )
                if is_implicit:
                    writer = getattr(self.asset, f"write_joint_{gain_name}_to_sim_index")
                    writer(**{gain_name: values[:, actuator_indices]}, joint_ids=writer_joint_ids, env_ids=env_ids)
                elif is_native:
                    write_group_parameter(
                        self.asset.actuators,
                        actuator_name,
                        "controller",
                        native_key,
                        values=values[:, actuator_indices],
                        env_ids=env_ids,
                        joint_ids=group_columns,
                    )
                else:
                    getattr(actuator, gain_name)[env_ids] = values


class randomize_joint_parameters(ManagerTermBase):
    """Randomize the simulated joint parameters of an articulation by adding, scaling, or setting random values.

    This function allows randomizing the joint parameters of the asset. These correspond to the physics engine
    joint properties that affect the joint behavior. The properties include the joint friction coefficient, armature,
    and joint position limits.

    The function samples random values from the given distribution parameters and applies the operation to the
    joint properties. It then sets the values into the physics simulation. If the distribution parameters are
    not provided for a particular property, the function does not modify the property.

    .. tip::
        This function uses CPU tensors to assign the joint properties. It is recommended to use this function
        only during the initialization of the environment.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the event term.
            env: The environment instance.

        Raises:
            TypeError: If `params` is not a tuple of two numbers.
            ValueError: If the operation is not supported.
            ValueError: If the lower bound is negative or zero when not allowed.
            ValueError: If the upper bound is less than the lower bound.
        """
        super().__init__(cfg, env)

        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: Articulation = env.scene[self.asset_cfg.name]
        self._backend = "newton" if "newton" in env.sim.physics_manager.__name__.lower() else "physx"

        self.default_joint_friction_coeff = self.asset.data.joint_friction_coeff.torch.clone()
        self.default_joint_armature = self.asset.data.joint_armature.torch.clone()
        self.default_joint_pos_limits = self.asset.data.joint_pos_limits.torch.clone()
        # Newton supports static friction and passive viscous damping but not dynamic friction.
        self.default_viscous_joint_friction_coeff = self.asset.data.joint_viscous_friction_coeff.torch.clone()
        if self._backend == "physx":
            self.default_dynamic_joint_friction_coeff = self.asset.data.joint_dynamic_friction_coeff.torch.clone()

        _check_operation(
            "randomize_joint_parameters",
            cfg,
            {"friction_distribution_params": True, "armature_distribution_params": True},
        )

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        friction_distribution_params: tuple[float, float] | None = None,
        armature_distribution_params: tuple[float, float] | None = None,
        lower_limit_distribution_params: tuple[float, float] | None = None,
        upper_limit_distribution_params: tuple[float, float] | None = None,
        operation: Literal["add", "scale", "abs"] = "abs",
        distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
    ):
        env_ids = _resolve_env_ids(env, env_ids, self.asset.device)
        if self.asset_cfg.joint_ids == slice(None):
            joint_ids = slice(None)
            env_ids_for_slice = env_ids
        else:
            joint_ids = torch.tensor(self.asset_cfg.joint_ids, dtype=torch.int, device=self.asset.device)
            env_ids_for_slice = env_ids[:, None]

        def randomize(defaults: torch.Tensor, params: tuple[float, float]) -> torch.Tensor:
            """Randomize a copy of ``defaults`` for the selected environments and joints."""
            return _randomize_prop_by_op(
                defaults.clone(), params, env_ids, joint_ids, operation=operation, distribution=distribution
            )

        if friction_distribution_params is not None:
            # friction coefficients are non-negative; the selected entries are indexed once for the writes
            static_friction_coeff = randomize(self.default_joint_friction_coeff, friction_distribution_params).clamp_(
                min=0.0
            )
            viscous_friction_coeff = randomize(
                self.default_viscous_joint_friction_coeff, friction_distribution_params
            ).clamp_(min=0.0)[env_ids_for_slice, joint_ids]

            if self._backend == "newton":
                self.asset.write_joint_friction_coefficient_to_sim_index(
                    joint_friction_coeff=static_friction_coeff[env_ids_for_slice, joint_ids],
                    joint_ids=joint_ids,
                    env_ids=env_ids,
                )
                self.asset.write_joint_viscous_friction_coefficient_to_sim_index(
                    joint_viscous_friction_coeff=viscous_friction_coeff,
                    joint_ids=joint_ids,
                    env_ids=env_ids,
                )
            else:
                # dynamic friction must not exceed static friction
                dynamic_friction_coeff = randomize(
                    self.default_dynamic_joint_friction_coeff, friction_distribution_params
                ).clamp_(min=0.0)
                dynamic_friction_coeff = torch.minimum(dynamic_friction_coeff, static_friction_coeff)
                self.asset.write_joint_friction_coefficient_to_sim_index(
                    joint_friction_coeff=static_friction_coeff[env_ids_for_slice, joint_ids],
                    joint_dynamic_friction_coeff=dynamic_friction_coeff[env_ids_for_slice, joint_ids],
                    joint_viscous_friction_coeff=viscous_friction_coeff,
                    joint_ids=joint_ids,
                    env_ids=env_ids,
                )

        if armature_distribution_params is not None:
            armature = randomize(self.asset.data.default_joint_armature.torch, armature_distribution_params)
            self.asset.write_joint_armature_to_sim(
                armature[env_ids_for_slice, joint_ids], joint_ids=joint_ids, env_ids=env_ids
            )

        if lower_limit_distribution_params is not None or upper_limit_distribution_params is not None:
            joint_pos_limits = self.default_joint_pos_limits.clone()
            for limit_idx, params in enumerate((lower_limit_distribution_params, upper_limit_distribution_params)):
                if params is not None:
                    _randomize_prop_by_op(
                        joint_pos_limits[..., limit_idx],
                        params,
                        env_ids,
                        joint_ids,
                        operation=operation,
                        distribution=distribution,
                    )
            joint_pos_limits = joint_pos_limits[env_ids_for_slice, joint_ids]
            if (joint_pos_limits[..., 0] > joint_pos_limits[..., 1]).any():
                raise ValueError(
                    "Randomization term 'randomize_joint_parameters' is setting lower joint limits that are greater"
                    " than upper joint limits. Please check the distribution parameters for the joint position limits."
                )
            self.asset.write_joint_position_limit_to_sim_index(
                limits=joint_pos_limits, joint_ids=joint_ids, env_ids=env_ids, warn_limit_violation=False
            )


class randomize_fixed_tendon_parameters(ManagerTermBase):
    """Randomize the simulated fixed tendon parameters of an articulation by adding, scaling, or setting random values.

    This function allows randomizing the fixed tendon parameters of the asset.
    These correspond to the physics engine tendon properties that affect the joint behavior.

    The function samples random values from the given distribution parameters and applies the operation to
    the tendon properties. It then sets the values into the physics simulation. If the distribution parameters
    are not provided for a particular property, the function does not modify the property.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the event term.
            env: The environment instance.

        Raises:
            TypeError: If `params` is not a tuple of two numbers.
            ValueError: If the operation is not supported.
            ValueError: If the lower bound is negative or zero when not allowed.
            ValueError: If the upper bound is less than the lower bound.
        """
        super().__init__(cfg, env)

        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: RigidObject | Articulation = env.scene[self.asset_cfg.name]
        _check_operation(
            "randomize_fixed_tendon_parameters",
            cfg,
            {
                "stiffness_distribution_params": False,
                "damping_distribution_params": True,
                "limit_stiffness_distribution_params": True,
            },
        )

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        stiffness_distribution_params: tuple[float, float] | None = None,
        damping_distribution_params: tuple[float, float] | None = None,
        limit_stiffness_distribution_params: tuple[float, float] | None = None,
        lower_limit_distribution_params: tuple[float, float] | None = None,
        upper_limit_distribution_params: tuple[float, float] | None = None,
        rest_length_distribution_params: tuple[float, float] | None = None,
        offset_distribution_params: tuple[float, float] | None = None,
        operation: Literal["add", "scale", "abs"] = "abs",
        distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
    ):
        is_newton = "newton" in env.sim.physics_manager.__name__.lower()
        env_ids = _resolve_env_ids(env, env_ids, self.asset.device)
        if self.asset_cfg.fixed_tendon_ids == slice(None):
            tendon_ids = slice(None)
        else:
            tendon_ids = torch.tensor(self.asset_cfg.fixed_tendon_ids, dtype=torch.int, device=self.asset.device)
        selection = (env_ids[:, None], tendon_ids)

        def randomize(current: torch.Tensor, params: tuple[float, float]) -> torch.Tensor:
            """Randomize a copy of ``current`` and return the entries of the selected environments and tendons."""
            values = _randomize_prop_by_op(
                current.clone(), params, env_ids, tendon_ids, operation=operation, distribution=distribution
            )
            return values[selection]

        data = self.asset.data
        if stiffness_distribution_params is not None:
            stiffness = randomize(data.fixed_tendon_stiffness.torch, stiffness_distribution_params)
            self.asset.set_fixed_tendon_stiffness_index(
                stiffness=stiffness, fixed_tendon_ids=tendon_ids, env_ids=env_ids
            )

        if damping_distribution_params is not None:
            damping = randomize(data.fixed_tendon_damping.torch, damping_distribution_params)
            self.asset.set_fixed_tendon_damping_index(damping=damping, fixed_tendon_ids=tendon_ids, env_ids=env_ids)

        # the remaining properties are only exposed by PhysX
        if limit_stiffness_distribution_params is not None:
            if is_newton:
                raise NotImplementedError("Limit stiffness is not supported in Newton.")
            limit_stiffness = randomize(data.fixed_tendon_limit_stiffness.torch, limit_stiffness_distribution_params)
            self.asset.set_fixed_tendon_limit_stiffness(limit_stiffness, tendon_ids, env_ids)

        if lower_limit_distribution_params is not None or upper_limit_distribution_params is not None:
            if is_newton:
                raise NotImplementedError("Position limits are not yet implemented with Newton.")
            limit = data.fixed_tendon_pos_limits.torch.clone()
            for limit_idx, params in enumerate((lower_limit_distribution_params, upper_limit_distribution_params)):
                if params is not None:
                    _randomize_prop_by_op(
                        limit[..., limit_idx],
                        params,
                        env_ids,
                        tendon_ids,
                        operation=operation,
                        distribution=distribution,
                    )
            tendon_limits = limit[selection]
            if (tendon_limits[..., 0] > tendon_limits[..., 1]).any():
                raise ValueError(
                    "Randomization term 'randomize_fixed_tendon_parameters' is setting lower tendon limits that are"
                    " greater than upper tendon limits."
                )
            self.asset.set_fixed_tendon_position_limit_index(
                limit=tendon_limits, fixed_tendon_ids=tendon_ids, env_ids=env_ids
            )

        if rest_length_distribution_params is not None:
            if is_newton:
                raise NotImplementedError("Rest length is not yet implemented with Newton.")
            rest_length = randomize(data.fixed_tendon_rest_length.torch, rest_length_distribution_params)
            self.asset.set_fixed_tendon_rest_length_index(
                rest_length=rest_length, fixed_tendon_ids=tendon_ids, env_ids=env_ids
            )

        if offset_distribution_params is not None:
            if is_newton:
                raise NotImplementedError("Offset is not supported in Newton.")
            offset = randomize(data.fixed_tendon_offset.torch, offset_distribution_params)
            self.asset.set_fixed_tendon_offset_index(offset=offset, fixed_tendon_ids=tendon_ids, env_ids=env_ids)

        self.asset.write_fixed_tendon_properties_to_sim_index(env_ids=env_ids)


def apply_external_force_torque(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    force_range: tuple[float, float],
    torque_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Randomize the external forces and torques applied to the bodies.

    This function creates a set of random forces and torques sampled from the given ranges. The number of forces
    and torques is equal to the number of bodies times the number of environments. The forces and torques are
    applied to the bodies by calling ``asset.set_external_force_and_torque``. The forces and torques are only
    applied when ``asset.write_data_to_sim()`` is called in the environment.
    """
    # a zero wrench range leaves the previously set wrenches untouched
    if all(bound == 0.0 for bound in (*force_range, *torque_range)):
        return

    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    env_ids = _resolve_env_ids(env, env_ids, asset.device, torch.int32)
    num_bodies = len(asset_cfg.body_ids) if isinstance(asset_cfg.body_ids, list) else asset.num_bodies

    size = (len(env_ids), num_bodies, 3)
    forces = math_utils.sample_uniform(*force_range, size, asset.device)
    torques = math_utils.sample_uniform(*torque_range, size, asset.device)
    # the wrenches are applied when ``asset.write_data_to_sim()`` is called
    asset.permanent_wrench_composer.set_forces_and_torques_index(
        forces=forces, torques=torques, body_ids=asset_cfg.body_ids, env_ids=env_ids
    )


def push_by_setting_velocity(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Push the asset by setting the root velocity to a random value within the given ranges.

    This creates an effect similar to pushing the asset with a random impulse that changes the asset's velocity.
    It samples the root velocity from the given ranges and sets the velocity into the physics simulation.

    The function takes a dictionary of velocity ranges for each axis and rotation. The keys of the dictionary
    are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``. The values are tuples of the form ``(min, max)``.
    If the dictionary does not contain a key, the velocity is set to zero for that axis.
    """
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    # tensor indexing returns a copy, so the increment does not touch the simulation buffer
    vel_w = asset.data.root_vel_w.torch[env_ids]
    vel_w += _sample_uniform_ranges(velocity_range, _POSE_KEYS, len(env_ids), asset.device)
    asset.write_root_velocity_to_sim_index(root_velocity=vel_w, env_ids=env_ids)


class reset_root_state_uniform(ManagerTermBase):
    """Reset the asset root state to a random position and velocity uniformly within the given ranges.

    This term randomizes the root position and velocity of the asset.

    * It samples the root position from the given ranges and adds them to the default root position, before setting
      them into the physics simulation.
    * It samples the root orientation from the given ranges and sets them into the physics simulation.
    * It samples the root velocity from the given ranges and sets them into the physics simulation.

    The term takes a dictionary of pose and velocity ranges for each axis and rotation. The keys of the
    dictionary are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``. The values are tuples of the form
    ``(min, max)``. If the dictionary does not contain a key, the position or velocity is set to zero for that axis.

    The range dictionaries are materialized as device tensors once at construction.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        pose_range = cfg.params.get("pose_range", {})
        velocity_range = cfg.params.get("velocity_range", {})
        self._pose_ranges = torch.tensor([pose_range.get(key, (0.0, 0.0)) for key in _POSE_KEYS], device=env.device)
        self._velocity_ranges = torch.tensor(
            [velocity_range.get(key, (0.0, 0.0)) for key in _POSE_KEYS], device=env.device
        )

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        pose_range: dict[str, tuple[float, float]],
        velocity_range: dict[str, tuple[float, float]],
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ):
        asset: RigidObject | Articulation = env.scene[asset_cfg.name]
        # tensor indexing already returns a copy, and the values are only read below
        default_root_pose = asset.data.default_root_pose.torch[env_ids]
        default_root_vel = asset.data.default_root_vel.torch[env_ids]
        shape = (len(env_ids), len(_POSE_KEYS))

        ranges = self._pose_ranges
        rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], shape, device=asset.device)
        positions = default_root_pose[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
        orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        orientations = math_utils.quat_mul(default_root_pose[:, 3:7], orientations_delta)

        ranges = self._velocity_ranges
        velocities = default_root_vel + math_utils.sample_uniform(
            ranges[:, 0], ranges[:, 1], shape, device=asset.device
        )

        asset.write_root_pose_to_sim_index(root_pose=torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
        asset.write_root_velocity_to_sim_index(root_velocity=velocities, env_ids=env_ids)


def reset_root_state_with_random_orientation(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the asset root position and velocities sampled randomly within the given ranges
    and the asset root orientation sampled randomly from the SO(3).

    This function randomizes the root position and velocity of the asset.

    * It samples the root position from the given ranges and adds them to the default root position, before setting
      them into the physics simulation.
    * It samples the root orientation uniformly from the SO(3) and sets them into the physics simulation.
    * It samples the root velocity from the given ranges and sets them into the physics simulation.

    The function takes a dictionary of position and velocity ranges for each axis and rotation:

    * :attr:`pose_range` - a dictionary of position ranges for each axis. The keys of the dictionary are ``x``,
      ``y``, and ``z``. The orientation is sampled uniformly from the SO(3).
    * :attr:`velocity_range` - a dictionary of velocity ranges for each axis and rotation. The keys of the dictionary
      are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``.

    The values are tuples of the form ``(min, max)``. If the dictionary does not contain a particular key,
    the position is set to zero for that axis.
    """
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    num_envs = len(env_ids)
    # tensor indexing already returns a copy, and the values are only read below
    default_root_pose = asset.data.default_root_pose.torch[env_ids]
    default_root_vel = asset.data.default_root_vel.torch[env_ids]

    rand_samples = _sample_uniform_ranges(pose_range, _POSITION_KEYS, num_envs, asset.device)
    positions = default_root_pose[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples
    orientations = math_utils.random_orientation(num_envs, device=asset.device)
    velocities = default_root_vel + _sample_uniform_ranges(velocity_range, _POSE_KEYS, num_envs, asset.device)

    asset.write_root_pose_to_sim_index(root_pose=torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim_index(root_velocity=velocities, env_ids=env_ids)


def reset_root_state_from_terrain(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the asset root state by sampling a random valid pose from the terrain.

    This function samples a random valid pose(based on flat patches) from the terrain and sets the root state
    of the asset to this position. The function also samples random velocities from the given ranges and sets them
    into the physics simulation.

    The function takes a dictionary of position and velocity ranges for each axis and rotation:

    * :attr:`pose_range` - a dictionary of pose ranges for each axis. The keys of the dictionary are ``roll``,
      ``pitch``, and ``yaw``. The position is sampled from the flat patches of the terrain.
    * :attr:`velocity_range` - a dictionary of velocity ranges for each axis and rotation. The keys of the dictionary
      are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``.

    The values are tuples of the form ``(min, max)``. If the dictionary does not contain a particular key,
    the position is set to zero for that axis.

    Note:
        The function expects the terrain to have valid flat patches under the key "init_pos". The flat patches
        are used to sample the random pose for the robot.

    Raises:
        ValueError: If the terrain does not have valid flat patches under the key "init_pos".
    """
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    num_envs = len(env_ids)

    valid_positions: torch.Tensor = terrain.flat_patches.get("init_pos")
    if valid_positions is None:
        raise ValueError(
            "The event term 'reset_root_state_from_terrain' requires valid flat patches under 'init_pos'."
            f" Found: {list(terrain.flat_patches.keys())}"
        )

    # sample one flat patch per environment and offset it by the default root position
    ids = torch.randint(0, valid_positions.shape[2], size=(num_envs,), device=env.device)
    positions = valid_positions[terrain.terrain_levels[env_ids], terrain.terrain_types[env_ids], ids]
    positions += asset.data.default_root_pose.torch[env_ids, :3]

    rand_samples = _sample_uniform_ranges(pose_range, _ROTATION_KEYS, num_envs, asset.device)
    orientations = math_utils.quat_from_euler_xyz(rand_samples[:, 0], rand_samples[:, 1], rand_samples[:, 2])
    velocities = asset.data.default_root_vel.torch[env_ids] + _sample_uniform_ranges(
        velocity_range, _POSE_KEYS, num_envs, asset.device
    )

    asset.write_root_pose_to_sim_index(root_pose=torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim_index(root_velocity=velocities, env_ids=env_ids)


def reset_joints_by_scale(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the robot joints by scaling the default position and velocity by the given ranges.

    This function samples random values from the given ranges and scales the default joint positions and velocities
    by these values. The scaled values are then set into the physics simulation.
    """
    _reset_joints_from_default(env, env_ids, position_range, velocity_range, asset_cfg, operation="scale")


def reset_joints_by_offset(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the robot joints with offsets around the default position and velocity by the given ranges.

    This function samples random values from the given ranges and biases the default joint positions and velocities
    by these values. The biased values are then set into the physics simulation.
    """
    _reset_joints_from_default(env, env_ids, position_range, velocity_range, asset_cfg, operation="add")


def _reset_joints_from_default(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
    asset_cfg: SceneEntityCfg,
    operation: Literal["add", "scale"],
):
    """Perturb the default joint state of the selected joints, clamp it to the soft limits, and write it to the sim."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    # broadcast the environment indices against explicit joint indices
    iter_env_ids = env_ids if joint_ids == slice(None) else env_ids[:, None]

    joint_pos = asset.data.default_joint_pos.torch[iter_env_ids, joint_ids].clone()
    joint_vel = asset.data.default_joint_vel.torch[iter_env_ids, joint_ids].clone()
    pos_samples = math_utils.sample_uniform(*position_range, joint_pos.shape, joint_pos.device)
    vel_samples = math_utils.sample_uniform(*velocity_range, joint_vel.shape, joint_vel.device)
    if operation == "scale":
        joint_pos *= pos_samples
        joint_vel *= vel_samples
    else:
        joint_pos += pos_samples
        joint_vel += vel_samples

    joint_pos_limits = asset.data.soft_joint_pos_limits.torch[iter_env_ids, joint_ids]
    joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
    joint_vel_limits = asset.data.soft_joint_vel_limits.torch[iter_env_ids, joint_ids]
    joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)

    asset.write_joint_position_to_sim_index(position=joint_pos, joint_ids=joint_ids, env_ids=env_ids)
    asset.write_joint_velocity_to_sim_index(velocity=joint_vel, joint_ids=joint_ids, env_ids=env_ids)


class reset_joints_within_limits_range(ManagerTermBase):
    """Reset an articulation's joints to a random position in the given limit ranges.

    This function samples random values for the joint position and velocities from the given limit ranges.
    The values are then set into the physics simulation.

    The parameters to the function are:

    * :attr:`position_range` - a dictionary of position ranges for each joint. The keys of the dictionary are the
      joint names (or regular expressions) of the asset.
    * :attr:`velocity_range` - a dictionary of velocity ranges for each joint. The keys of the dictionary are the
      joint names (or regular expressions) of the asset.
    * :attr:`use_default_offset` - a boolean flag to indicate if the ranges are offset by the default joint state.
      Defaults to False.
    * :attr:`asset_cfg` - the configuration of the asset to reset. Defaults to the entity named "robot" in the scene.
    * :attr:`operation` - whether the ranges are scaled values of the joint limits, or absolute limits.
       Defaults to "abs".

    The dictionary values are a tuple of the form ``(a, b)``. Based on the operation, these values are
    interpreted differently:

    * If the operation is "abs", the values are the absolute minimum and maximum values for the joint, i.e.
      the joint range becomes ``[a, b]``.
    * If the operation is "scale", the values are the scaling factors for the joint limits, i.e. the joint range
      becomes ``[a * min_joint_limit, b * max_joint_limit]``.

    If the ``a`` or the ``b`` value is ``None``, the joint limits are used instead.

    Note:
        If the dictionary does not contain a key, the joint position or joint velocity is set to the default value for
        that joint.

    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        # check if the cfg has the required parameters
        if "position_range" not in cfg.params or "velocity_range" not in cfg.params:
            raise ValueError(
                "The term 'reset_joints_within_limits_range' requires parameters: 'position_range' and"
                f" 'velocity_range'. Received: {list(cfg.params.keys())}."
            )

        # parse the parameters
        asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        use_default_offset = cfg.params.get("use_default_offset", False)
        operation = cfg.params.get("operation", "abs")
        # check if the operation is valid
        if operation not in ["abs", "scale"]:
            raise ValueError(
                f"For event 'reset_joints_within_limits_range', unknown operation: '{operation}'."
                " Please use 'abs' or 'scale'."
            )

        self._asset: Articulation = env.scene[asset_cfg.name]
        data = self._asset.data

        # the joint limits of the first instance define the ranges of unspecified bounds
        soft_joint_vel_limits = data.soft_joint_vel_limits.torch[0]
        self._pos_joint_ids, self._pos_ranges = self._parse_joint_ranges(
            cfg.params["position_range"],
            data.soft_joint_pos_limits.torch[0].clone(),
            data.default_joint_pos.torch[0] if use_default_offset else None,
            operation,
        )
        self._vel_joint_ids, self._vel_ranges = self._parse_joint_ranges(
            cfg.params["velocity_range"],
            torch.stack([-soft_joint_vel_limits, soft_joint_vel_limits], dim=1),
            data.default_joint_vel.torch[0] if use_default_offset else None,
            operation,
        )

    def _parse_joint_ranges(
        self,
        joint_ranges: dict[str, tuple[float | None, float | None]],
        limits: torch.Tensor,
        default_offsets: torch.Tensor | None,
        operation: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Resolve per-joint ``(low, high)`` ranges against the joint limits.

        Args:
            joint_ranges: Ranges keyed by joint name or regular expression. ``None`` bounds keep the limit.
            limits: Joint limits used as the base ranges, shape (num_joints, 2). Modified in place.
            default_offsets: Default joint values added to the ranges, or None to skip the offset.
            operation: ``"abs"`` overwrites the limits with the bounds, ``"scale"`` multiplies them.

        Returns:
            The selected joint indices and their ranges, shapes (num_selected,) and (num_selected, 2).
        """
        selected_joint_ids = []
        for joint_name, joint_range in joint_ranges.items():
            joint_ids = self._asset.find_joints(joint_name)[0]
            selected_joint_ids.extend(joint_ids)
            for bound_idx, bound in enumerate(joint_range):
                if bound is None:
                    continue
                if operation == "abs":
                    limits[joint_ids, bound_idx] = bound
                else:
                    limits[joint_ids, bound_idx] *= bound
            if default_offsets is not None:
                limits[joint_ids] += default_offsets[joint_ids].unsqueeze(1)
        selected_joint_ids = torch.tensor(selected_joint_ids, device=limits.device)
        return selected_joint_ids, limits[selected_joint_ids]

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        position_range: dict[str, tuple[float | None, float | None]],
        velocity_range: dict[str, tuple[float | None, float | None]],
        use_default_offset: bool = False,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        operation: Literal["abs", "scale"] = "abs",
    ):
        # unspecified joints keep their default state
        joint_pos = self._asset.data.default_joint_pos.torch[env_ids].clone()
        joint_vel = self._asset.data.default_joint_vel.torch[env_ids].clone()

        if len(self._pos_joint_ids) > 0:
            joint_pos_shape = (len(env_ids), len(self._pos_joint_ids))
            joint_pos[:, self._pos_joint_ids] = math_utils.sample_uniform(
                self._pos_ranges[:, 0], self._pos_ranges[:, 1], joint_pos_shape, device=joint_pos.device
            )
            joint_pos_limits = self._asset.data.soft_joint_pos_limits.torch[0, self._pos_joint_ids]
            joint_pos = joint_pos.clamp(joint_pos_limits[:, 0], joint_pos_limits[:, 1])

        if len(self._vel_joint_ids) > 0:
            joint_vel_shape = (len(env_ids), len(self._vel_joint_ids))
            joint_vel[:, self._vel_joint_ids] = math_utils.sample_uniform(
                self._vel_ranges[:, 0], self._vel_ranges[:, 1], joint_vel_shape, device=joint_vel.device
            )
            joint_vel_limits = self._asset.data.soft_joint_vel_limits.torch[0, self._vel_joint_ids]
            joint_vel = joint_vel.clamp(-joint_vel_limits, joint_vel_limits)

        self._asset.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
        self._asset.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)


def reset_nodal_state_uniform(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the asset nodal state to a random position and velocity uniformly within the given ranges.

    This function randomizes the nodal position and velocity of the asset.

    * It samples the root position from the given ranges and adds them to the default nodal position, before setting
      them into the physics simulation.
    * It samples the root velocity from the given ranges and sets them into the physics simulation.

    The function takes a dictionary of position and velocity ranges for each axis. The keys of the
    dictionary are ``x``, ``y``, ``z``. The values are tuples of the form ``(min, max)``.
    If the dictionary does not contain a key, the position or velocity is set to zero for that axis.
    """
    asset: DeformableObject = env.scene[asset_cfg.name]
    num_envs = len(env_ids)
    # one offset per environment is shared by all nodes
    nodal_state = asset.data.default_nodal_state_w.torch[env_ids].clone()
    nodal_state[..., :3] += _sample_uniform_ranges(position_range, _POSITION_KEYS, num_envs, asset.device).unsqueeze(1)
    nodal_state[..., 3:] += _sample_uniform_ranges(velocity_range, _POSITION_KEYS, num_envs, asset.device).unsqueeze(1)
    asset.write_nodal_state_to_sim(nodal_state, env_ids=env_ids)


def reset_scene_to_default(env: ManagerBasedEnv, env_ids: torch.Tensor, reset_joint_targets: bool = False):
    """Reset the scene to the default state specified in the scene configuration.

    If :attr:`reset_joint_targets` is True, the joint position and velocity targets of the articulations are
    also reset to their default values. This might be useful for some cases to clear out any previously set targets.
    However, this is not the default behavior as based on our experience, it is not always desired to reset
    targets to default values, especially when the targets should be handled by action terms and not event terms.
    """
    env_origins = env.scene.env_origins[env_ids]
    # the default root poses are stored relative to the environment origins
    for asset in (*env.scene.rigid_objects.values(), *env.scene.articulations.values()):
        default_root_pose = asset.data.default_root_pose.torch[env_ids].clone()
        default_root_pose[:, :3] += env_origins
        asset.write_root_pose_to_sim_index(root_pose=default_root_pose, env_ids=env_ids)
        asset.write_root_velocity_to_sim_index(
            root_velocity=asset.data.default_root_vel.torch[env_ids], env_ids=env_ids
        )
    for articulation in env.scene.articulations.values():
        default_joint_pos = articulation.data.default_joint_pos.torch[env_ids].clone()
        default_joint_vel = articulation.data.default_joint_vel.torch[env_ids].clone()
        articulation.write_joint_position_to_sim_index(position=default_joint_pos, env_ids=env_ids)
        articulation.write_joint_velocity_to_sim_index(velocity=default_joint_vel, env_ids=env_ids)
        if reset_joint_targets:
            articulation.set_joint_position_target_index(target=default_joint_pos, env_ids=env_ids)
            articulation.set_joint_velocity_target_index(target=default_joint_vel, env_ids=env_ids)
    for cable_object in env.scene.cable_objects.values():
        cable_object.write_segment_pose_to_sim_index(
            segment_pose=cable_object.data.default_segment_pose_w.torch[env_ids], env_ids=env_ids
        )
        cable_object.write_segment_velocity_to_sim_index(
            segment_velocity=cable_object.data.default_segment_velocity_w.torch[env_ids], env_ids=env_ids
        )
    for deformable_object in env.scene.deformable_objects.values():
        nodal_state = deformable_object.data.default_nodal_state_w.torch[env_ids]
        deformable_object.write_nodal_state_to_sim(nodal_state, env_ids=env_ids)


class randomize_visual_texture_material(ManagerTermBase):
    """Randomize the visual texture of bodies on an asset using Replicator API.

    This function randomizes the visual texture of the bodies of the asset using the Replicator API.
    The function samples random textures from the given texture paths and applies them to the bodies
    of the asset. The textures are projected onto the bodies and rotated by the given angles.

    .. note::
        The function assumes that the asset follows the prim naming convention as:
        "{asset_prim_path}/{body_name}/visuals" where the body name is the name of the body to
        which the texture is applied. This is the default prim ordering when importing assets
        from the asset converters in Isaac Lab.

    .. note::
        When randomizing the texture of individual assets, please make sure to set
        :attr:`isaaclab.scene.InteractiveSceneCfg.replicate_physics` to False. This ensures that physics
        parser will parse the individual asset properties separately.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the event term.
            env: The environment instance.
        """
        super().__init__(cfg, env)
        rep = _import_replicator(env, "texture material")

        asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg")
        asset = env.scene[asset_cfg.name]
        prim_path = _resolve_visual_prim_pattern(asset, asset_cfg.body_names, "texture")

        if _replicator_uses_omnigraph(rep):
            texture_paths = cfg.params.get("texture_paths")
            event_name = cfg.params.get("event_name")
            texture_rotation = tuple(math.degrees(angle) for angle in cfg.params.get("texture_rotation", (0.0, 0.0)))

            def rep_texture_randomization():
                prims_group = rep.get.prims(path_pattern=prim_path)
                with prims_group:
                    rep.randomizer.texture(
                        textures=texture_paths,
                        project_uvw=True,
                        texture_rotate=rep.distribution.uniform(*texture_rotation),
                    )
                return prims_group.node

            with rep.trigger.on_custom_event(event_name=event_name):
                rep_texture_randomization()
        else:
            self.texture_rng = rep.rng.ReplicatorRNG()
            self.material_prims = _bind_replicator_materials(rep, env, prim_path)

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        event_name: str,
        asset_cfg: SceneEntityCfg,
        texture_paths: list[str],
        texture_rotation: tuple[float, float] = (0.0, 0.0),
    ):
        # note: This triggers the nodes for all the environments.
        #   We need to investigate how to make it happen only for a subset based on env_ids.
        import omni.replicator.core as rep  # noqa: PLC0415

        if _replicator_uses_omnigraph(rep):
            rep.utils.send_og_event(event_name)
            return

        texture_paths = texture_paths or self._cfg.params.get("texture_paths")
        texture_rotation = texture_rotation or self._cfg.params.get("texture_rotation", (0.0, 0.0))
        texture_rotation = tuple(math.degrees(angle) for angle in texture_rotation)

        num_prims = len(self.material_prims)
        random_textures = self.texture_rng.generator.choice(texture_paths, size=num_prims)
        random_rotations = self.texture_rng.generator.uniform(texture_rotation[0], texture_rotation[1], size=num_prims)
        rep.functional.modify.attribute(self.material_prims, "diffuse_texture", random_textures)
        rep.functional.modify.attribute(self.material_prims, "texture_rotate", random_rotations)


class randomize_visual_color(ManagerTermBase):
    """Randomize the visual color of bodies on an asset using Replicator API.

    This function randomizes the visual color of the bodies of the asset using the Replicator API.
    The function samples random colors from the given colors and applies them to the bodies
    of the asset.

    The function assumes that the asset follows the prim naming convention as:
    "{asset_prim_path}/{mesh_name}" where the mesh name is the name of the mesh to
    which the color is applied. For instance, if the asset has a prim path "/World/asset"
    and a mesh named "body_0/mesh", the prim path for the mesh would be
    "/World/asset/body_0/mesh".

    The colors can be specified as a list of tuples of the form ``(r, g, b)`` or as a dictionary
    with the keys ``r``, ``g``, ``b`` and values as tuples of the form ``(low, high)``.
    If a dictionary is used, the function will sample random colors from the given ranges.

    .. note::
        When randomizing the color of individual assets, please make sure to set
        :attr:`isaaclab.scene.InteractiveSceneCfg.replicate_physics` to False. This ensures that physics
        parser will parse the individual asset properties separately.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        """Initialize the randomization term.

        Args:
            cfg: The configuration of the event term.
            env: The environment instance.
        """
        super().__init__(cfg, env)
        rep = _import_replicator(env, "color")

        asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg")
        mesh_name: str = cfg.params.get("mesh_name", "")  # type: ignore
        asset = env.scene[asset_cfg.name]

        # Never match the articulation root prim: authoring on it (SetInstanceable and the material binding)
        # invalidates the PhysX articulation view and crashes a later at-play body-name resolution.
        if mesh_name:
            if not mesh_name.startswith("/"):
                mesh_name = "/" + mesh_name
            mesh_prim_path = f"{asset.cfg.prim_path}{mesh_name}"
        else:
            mesh_prim_path = _resolve_visual_prim_pattern(asset, asset_cfg.body_names, "color")
        # TODO: Need to make it work for multiple meshes.

        if _replicator_uses_omnigraph(rep):
            colors = cfg.params.get("colors")
            event_name = cfg.params.get("event_name")
            if isinstance(colors, dict):
                colors = rep.distribution.uniform(*_color_bounds(colors))
            else:
                colors = list(colors)

            def rep_color_randomization():
                prims_group = rep.get.prims(path_pattern=mesh_prim_path)
                with prims_group:
                    rep.randomizer.color(colors=colors)
                return prims_group.node

            with rep.trigger.on_custom_event(event_name=event_name):
                rep_color_randomization()
        else:
            self.color_rng = rep.rng.ReplicatorRNG()
            self.material_prims = _bind_replicator_materials(rep, env, mesh_prim_path)

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        event_name: str,
        asset_cfg: SceneEntityCfg,
        colors: list[tuple[float, float, float]] | dict[str, tuple[float, float]],
        mesh_name: str = "",
    ):
        # note: This triggers the nodes for all the environments.
        #   We need to investigate how to make it happen only for a subset based on env_ids.
        import omni.replicator.core as rep  # noqa: PLC0415

        if _replicator_uses_omnigraph(rep):
            rep.utils.send_og_event(event_name)
            return

        colors = colors or self._cfg.params.get("colors")
        colors = _color_bounds(colors) if isinstance(colors, dict) else list(colors)
        num_prims = len(self.material_prims)
        random_colors = self.color_rng.generator.uniform(colors[0], colors[1], size=(num_prims, 3))
        rep.functional.modify.attribute(self.material_prims, "diffuse_color_constant", random_colors)


def _import_replicator(env: ManagerBasedEnv, purpose: str):
    """Enable and import Replicator for USD-level visual randomization.

    Raises:
        RuntimeError: If scene replication is enabled. The check lives here because visual randomization
            can run outside of ``prestartup`` mode, where the event manager does not check it.
    """
    if env.cfg.scene.replicate_physics:
        raise RuntimeError(
            f"Unable to randomize visual {purpose} with scene replication enabled."
            " For stable USD-level randomization, please disable scene replication"
            " by setting 'replicate_physics' to False in 'InteractiveSceneCfg'."
        )
    sim_utils.enable_extension("omni.replicator.core")
    import omni.replicator.core as rep  # noqa: PLC0415

    return rep


def _replicator_uses_omnigraph(rep) -> bool:
    """Return whether the installed Replicator predates the functional API (before 1.12.4)."""
    version = re.match(r"^(\d+\.\d+\.\d+)", rep.__file__.split("/")[-5][21:]).group(1)
    return compare_versions(version, "1.12.4") < 0


def _resolve_visual_prim_pattern(asset, body_names: str | list[str] | None, purpose: str) -> str:
    """Return the prim path pattern of the visual prims of the selected bodies.

    Prefers the ``{asset}/{body}/visuals`` layout produced by the asset converters and falls back to every
    descendant of the asset when no prim matches it.
    """
    body_names_regex = "|".join(body_names) if isinstance(body_names, list) else body_names
    body_names_regex = f"(?:{body_names_regex})" if isinstance(body_names_regex, str) else ".*"
    pattern_with_visuals = f"{asset.cfg.prim_path}/{body_names_regex}/visuals"
    if sim_utils.resolve_matching_prims_from_source(pattern_with_visuals, raise_if_no_matches=False):
        return pattern_with_visuals
    fallback = f"{asset.cfg.prim_path}/.*"
    logger.info(
        f"Pattern '{pattern_with_visuals}' found no prims. Falling back to '{fallback}' for {purpose} randomization."
    )
    return fallback


def _bind_replicator_materials(rep, env: ManagerBasedEnv, prim_path: str):
    """Create one OmniPBR material per matched prim and bind it, returning the material prims."""
    prims_group = rep.functional.get.prims(path_pattern=prim_path, stage=env.sim.stage)
    for prim in prims_group:
        if prim.IsInstanceable():
            prim.SetInstanceable(False)

    # Resolve OmniPBR.mdl to an absolute path so that pxr.Ar.GetResolver().Resolve() returns a valid path.
    # Kit's omni_usd_resolver intentionally returns "" for builtin MDL short-names
    # (OMNI_USD_RESOLVER_MDL_BUILTIN_BYPASS=1), which makes Replicator >= 1.13.0 pass an empty resolved
    # path into UsdMdl.RegistryUtils and fail with an invalid 'rtx::neuraylib::MdlModuleId'.
    import carb.tokens  # noqa: PLC0415

    omni_pbr_mdl = carb.tokens.get_tokens_interface().resolve("${kit}/mdl/core/Base/OmniPBR.mdl")
    # TODO: Should we specify the value when creating the material?
    return rep.functional.create_batch.material(
        mdl=omni_pbr_mdl, bind_prims=prims_group, count=len(prims_group), project_uvw=True
    )


def _color_bounds(colors: dict[str, tuple[float, float]]) -> tuple[list[float], list[float]]:
    """Split per-channel ``(low, high)`` color ranges into ``(low_rgb, high_rgb)`` lists."""
    return [colors[key][0] for key in "rgb"], [colors[key][1] for key in "rgb"]


"""
Internal helper functions.
"""

_POSE_KEYS = ("x", "y", "z", "roll", "pitch", "yaw")
_POSITION_KEYS = ("x", "y", "z")
_ROTATION_KEYS = ("roll", "pitch", "yaw")

_DISTRIBUTION_FNS = {
    "uniform": math_utils.sample_uniform,
    "log_uniform": math_utils.sample_log_uniform,
    "gaussian": math_utils.sample_gaussian,
}


def _resolve_distribution_fn(distribution: str):
    """Return the sampling function for a distribution name."""
    try:
        return _DISTRIBUTION_FNS[distribution]
    except KeyError:
        raise NotImplementedError(
            f"Unknown distribution: '{distribution}' for property randomization."
            " Please use 'uniform', 'log_uniform', or 'gaussian'."
        ) from None


def _check_operation(term_name: str, cfg: EventTermCfg, scale_params: dict[str, bool]) -> None:
    """Validate the ``operation`` parameter of a term and the ranges used with the ``scale`` operation.

    Args:
        term_name: Name of the term, used in error messages.
        cfg: The term configuration.
        scale_params: Distribution-parameter names to validate for ``scale``, mapped to whether zero is allowed.
    """
    operation = cfg.params["operation"]
    if operation == "scale":
        for name, allow_zero in scale_params.items():
            if name in cfg.params:
                _validate_scale_range(cfg.params[name], name, allow_zero=allow_zero)
    elif operation not in ("abs", "add"):
        raise ValueError(f"Randomization term '{term_name}' does not support operation: '{operation}'.")


def _resolve_env_ids(
    env: ManagerBasedEnv, env_ids: torch.Tensor | None, device: str | torch.device, dtype: torch.dtype | None = None
) -> torch.Tensor:
    """Resolve ``None`` to all environment indices and move the indices to ``device``."""
    if env_ids is None:
        return torch.arange(env.scene.num_envs, device=device, dtype=dtype)
    return env_ids.to(device=device) if dtype is None else env_ids.to(device=device, dtype=dtype)


def _resolve_body_ids(asset: RigidObject | Articulation, body_ids: list[int] | slice) -> torch.Tensor:
    """Materialize a body selector as an ``int32`` index tensor on the asset device."""
    if body_ids == slice(None):
        return torch.arange(asset.num_bodies, dtype=torch.int32, device=asset.device)
    return torch.tensor(body_ids, dtype=torch.int32, device=asset.device)


def _sample_uniform_ranges(
    ranges: dict[str, tuple[float, float]],
    keys: tuple[str, ...],
    num_samples: int,
    device: str | torch.device,
    default: tuple[float, float] = (0.0, 0.0),
) -> torch.Tensor:
    """Sample one uniform value per key for each row from ``(low, high)`` ranges keyed by axis name.

    Missing keys use ``default``. The result has shape ``(num_samples, len(keys))``.
    """
    bounds = torch.tensor([ranges.get(key, default) for key in keys], device=device)
    return math_utils.sample_uniform(bounds[:, 0], bounds[:, 1], (num_samples, len(keys)), device=device)


def _sample_material_buckets(cfg: EventTermCfg) -> torch.Tensor:
    """Pre-sample ``num_buckets`` PhysX materials ``(static friction, dynamic friction, restitution)`` on the CPU."""
    ranges = torch.tensor(
        [
            cfg.params.get("static_friction_range", (1.0, 1.0)),
            cfg.params.get("dynamic_friction_range", (1.0, 1.0)),
            cfg.params.get("restitution_range", (0.0, 0.0)),
        ],
        device="cpu",
    )
    num_buckets = int(cfg.params.get("num_buckets", 1))
    buckets = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (num_buckets, 3), device="cpu")
    if cfg.params.get("make_consistent", False):
        # dynamic friction must not exceed static friction
        buckets[:, 1] = torch.min(buckets[:, 0], buckets[:, 1])
    return buckets


def _sample_absolute(defaults: torch.Tensor, params: tuple[float, float], distribution: str) -> torch.Tensor:
    """Return a copy of ``defaults`` with every entry replaced by a sample from ``params``."""
    return _randomize_prop_by_op(
        defaults.clone(), params, None, slice(None), operation="abs", distribution=distribution
    )


def _randomize_prop_by_op(
    data: torch.Tensor,
    distribution_parameters: tuple[float | torch.Tensor, float | torch.Tensor],
    dim_0_ids: torch.Tensor | None,
    dim_1_ids: torch.Tensor | slice,
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"],
) -> torch.Tensor:
    """Perform data randomization based on the given operation and distribution.

    Args:
        data: The data tensor to be randomized in place. Shape is (dim_0, dim_1).
        distribution_parameters: The parameters for the distribution to sample values from.
        dim_0_ids: The indices of the first dimension to randomize. ``None`` selects all rows.
        dim_1_ids: The indices of the second dimension to randomize.
        operation: The operation to perform on the data. Options: 'add', 'scale', 'abs'.
        distribution: The distribution to sample from. Options: 'uniform', 'log_uniform', 'gaussian'.

    Returns:
        The data tensor after randomization. Shape is (dim_0, dim_1).

    Raises:
        NotImplementedError: If the operation or distribution is not supported.
    """
    if dim_0_ids is None:
        n_dim_0 = data.shape[0]
        dim_0_ids = slice(None)
    else:
        n_dim_0 = len(dim_0_ids)
        if not isinstance(dim_1_ids, slice):
            dim_0_ids = dim_0_ids[:, None]
    n_dim_1 = data.shape[1] if isinstance(dim_1_ids, slice) else len(dim_1_ids)

    dist_fn = _resolve_distribution_fn(distribution)
    samples = dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    if operation == "add":
        data[dim_0_ids, dim_1_ids] += samples
    elif operation == "scale":
        data[dim_0_ids, dim_1_ids] *= samples
    elif operation == "abs":
        data[dim_0_ids, dim_1_ids] = samples
    else:
        raise NotImplementedError(
            f"Unknown operation: '{operation}' for property randomization. Please use 'add', 'scale', or 'abs'."
        )
    return data


def _validate_scale_range(
    params: tuple[float, float] | None,
    name: str,
    *,
    allow_negative: bool = False,
    allow_zero: bool = True,
) -> None:
    """Validate a ``(low, high)`` tuple used with the ``scale`` operation.

    Args:
        params: The (low, high) range to validate. If None, validation is skipped.
        name: The name of the parameter being validated, used for error messages.
        allow_negative: Whether the lower bound may be negative. Defaults to False.
        allow_zero: Whether the lower bound may be zero. Defaults to True.

    Raises:
        TypeError: If ``params`` is not a tuple of two numbers.
        ValueError: If the lower bound is negative or zero when not allowed.
        ValueError: If the upper bound is less than the lower bound.
    """
    if params is None:
        return
    low, high = params
    if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
        raise TypeError(f"{name}: expected (low, high) to be a tuple of numbers, got {params}.")
    if not allow_negative and not allow_zero and low <= 0:
        raise ValueError(f"{name}: lower bound must be > 0 when using the 'scale' operation (got {low}).")
    if not allow_negative and allow_zero and low < 0:
        raise ValueError(f"{name}: lower bound must be ≥ 0 when using the 'scale' operation (got {low}).")
    if high < low:
        raise ValueError(f"{name}: upper bound ({high}) must be ≥ lower bound ({low}).")
