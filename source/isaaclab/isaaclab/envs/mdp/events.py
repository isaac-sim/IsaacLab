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

import functools
import logging
from collections.abc import Sequence
from types import ModuleType
from typing import TYPE_CHECKING, Literal

import torch

from ... import sim as sim_utils
from ...managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from ...utils import math as math_utils

if TYPE_CHECKING:
    from isaaclab_physx.assets import DeformableObject

    from omni.replicator.core.scripts.utils.rng import ReplicatorRNG
    from pxr import Usd

    from ...assets import Articulation, RigidObject
    from ...terrains import TerrainImporter
    from .. import ManagerBasedEnv

# import logger
logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=128)
def _cached_range_tensor(values: tuple[tuple[float, float], ...], device: str) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32, device=device)


def _range_tensor(ranges: dict[str, tuple[float, float]], keys: Sequence[str], device: str) -> torch.Tensor:
    """Return the ``(min, max)`` bounds of ``keys`` as a device tensor of shape ``(len(keys), 2)``.

    The tensor is cached by value, so repeated calls avoid a host-to-device copy while range edits (for
    example by a curriculum) still take effect. Callers must not modify the returned tensor.

    Args:
        ranges: The ``(min, max)`` range for each key. Missing keys use ``(0.0, 0.0)``.
        keys: The keys to read, in output row order.
        device: The device of the returned tensor.

    Returns:
        The range bounds. Shape is ``(len(keys), 2)``.
    """
    values = tuple(tuple(float(v) for v in ranges.get(key, (0.0, 0.0))) for key in keys)
    return _cached_range_tensor(values, str(device))


def randomize_rigid_body_scale(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | slice | None,
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
    # check if sim is running
    if env.sim.is_playing():
        raise RuntimeError(
            "Randomizing scale while simulation is running leads to unpredictable behaviors."
            " Please ensure that the event term is called before the simulation starts by using the 'usd' mode."
        )

    from ...assets import BaseArticulation  # noqa: PLC0415

    asset: RigidObject = env.scene[asset_cfg.name]

    if isinstance(asset, BaseArticulation):
        raise ValueError(
            "Scaling an articulation randomly is not supported, as it affects joint attributes and can cause"
            " unexpected behavior. To achieve different scales, we recommend generating separate USD files for"
            " each version of the articulation and using multi-asset spawning. For more details, refer to:"
            " https://isaac-sim.github.io/IsaacLab/main/source/how-to/multi_asset_spawning.html"
        )

    # resolve environment ids
    if env_ids is None:
        env_ids = range(env.scene.num_envs)
    elif isinstance(env_ids, slice):
        env_ids = range(env.scene.num_envs)[env_ids]
    else:
        env_ids = env_ids.cpu()

    # acquire stage
    stage = env.sim.stage
    prim_paths = sim_utils.find_matching_prim_paths(asset.cfg.prim_path)

    if isinstance(scale_range, dict):
        range_list = [scale_range.get(key, (1.0, 1.0)) for key in ["x", "y", "z"]]
        ranges = torch.tensor(range_list, device="cpu")
        rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device="cpu")
    else:
        rand_samples = math_utils.sample_uniform(*scale_range, (len(env_ids), 1), device="cpu")
        rand_samples = rand_samples.repeat(1, 3)
    # convert to list for the for loop
    rand_samples = rand_samples.tolist()

    # apply the randomization to the parent if no relative child path is provided
    # this might be useful if user wants to randomize a particular mesh in the prim hierarchy
    if relative_child_path is None:
        relative_child_path = ""
    elif not relative_child_path.startswith("/"):
        relative_child_path = "/" + relative_child_path

    # use sdf changeblock for faster processing of USD properties (local: pxr only available with Kit)
    from pxr import Gf, Sdf, UsdGeom, Vt  # noqa: PLC0415

    with Sdf.ChangeBlock():
        for i, env_id in enumerate(env_ids):
            # path to prim to randomize
            prim_path = prim_paths[env_id] + relative_child_path
            prim_spec = Sdf.CreatePrimInLayer(stage.GetRootLayer(), prim_path)

            scale_spec = prim_spec.GetAttributeAtPath(prim_path + ".xformOp:scale")
            has_scale_attr = scale_spec is not None
            if not has_scale_attr:
                scale_spec = Sdf.AttributeSpec(prim_spec, prim_path + ".xformOp:scale", Sdf.ValueTypeNames.Double3)

            # set the new scale
            scale_spec.default = Gf.Vec3f(*rand_samples[i])

            # ensure the operation is done in the right ordering if we created the scale attribute.
            # otherwise, we assume the scale attribute is already in the right order.
            # note: by default isaac sim follows this ordering for the transform stack so any asset
            #   created through it will have the correct ordering
            if not has_scale_attr:
                op_order_spec = prim_spec.GetAttributeAtPath(prim_path + ".xformOpOrder")
                if op_order_spec is None:
                    op_order_spec = Sdf.AttributeSpec(
                        prim_spec, UsdGeom.Tokens.xformOpOrder, Sdf.ValueTypeNames.TokenArray
                    )
                op_order_spec.default = Vt.TokenArray(["xformOp:translate", "xformOp:orient", "xformOp:scale"])


class randomize_rigid_body_material(ManagerTermBase):
    """Randomize the physics materials on all geometries of the asset.

    This function creates a set of physics materials with random static friction, dynamic friction, and restitution
    values and assigns them to the geometries of the asset.

    For articulations, :attr:`SceneEntityCfg.body_ids` selects bodies in public articulation order. The backend
    implementations convert those IDs to backend shape ranges; callers must not pre-swizzle body IDs.

    The active backend determines how materials are sampled and assigned:

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

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv) -> None:
        """Initialize the implementation for the active physics backend.

        Args:
            cfg: Event configuration.
            env: Environment owning this term.
        """
        super().__init__(cfg, env)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: RigidObject | Articulation = env.scene[self.asset_cfg.name]

        from ...assets import BaseArticulation, BaseRigidObject  # noqa: PLC0415

        if not isinstance(self.asset, (BaseRigidObject, BaseArticulation)):
            raise ValueError(
                f"Randomization term 'randomize_rigid_body_material' not supported for asset: '{self.asset_cfg.name}'"
                f" with type: '{type(self.asset)}'."
            )

        self._impl = _get_backend_events(env).randomize_rigid_body_material(cfg, env)

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | slice | None,
        static_friction_range: tuple[float, float],
        dynamic_friction_range: tuple[float, float],
        restitution_range: tuple[float, float],
        num_buckets: int,
        asset_cfg: SceneEntityCfg,
        make_consistent: bool = False,
    ) -> None:
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
        # check for valid operation
        if cfg.params["operation"] == "scale":
            if "mass_distribution_params" in cfg.params:
                _validate_scale_range(
                    cfg.params["mass_distribution_params"], "mass_distribution_params", allow_zero=False
                )
        elif cfg.params["operation"] not in ("abs", "add"):
            raise ValueError(
                "Randomization term 'randomize_rigid_body_mass' does not support operation:"
                f" '{cfg.params['operation']}'."
            )
        if cfg.params.get("min_mass") is not None:
            if cfg.params.get("min_mass") < 1e-6:
                raise ValueError(
                    "Randomization term 'randomize_rigid_body_mass' does not support 'min_mass' less than 1e-6 to avoid"
                    " physics errors."
                )

        self.default_mass = None
        self.default_inertia = None

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | slice | None,
        asset_cfg: SceneEntityCfg,
        mass_distribution_params: tuple[float, float],
        operation: Literal["add", "scale", "abs"],
        distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
        recompute_inertia: bool = True,
        min_mass: float = 1e-6,
    ):
        if self.default_mass is None:
            self.default_mass = self.asset.data.body_mass.torch.clone()
        if self.default_inertia is None:
            self.default_inertia = self.asset.data.body_inertia.torch.clone()
        # resolve environment ids
        if env_ids is None:
            env_ids = slice(None)
        # resolve body indices
        body_ids = self.asset_cfg.body_ids
        # index rows and columns jointly only when both are tensors; with a slice the result is already 2D
        env_rows = env_ids[:, None] if not isinstance(env_ids, slice) and not isinstance(body_ids, slice) else env_ids

        # get the current masses of the bodies (num_assets, num_bodies)
        masses = self.asset.data.body_mass.torch.clone()

        # apply randomization on default values
        # this is to make sure when calling the function multiple times, the randomization is applied on the
        # default values and not the previously randomized values
        masses[env_rows, body_ids] = self.default_mass[env_rows, body_ids].clone()

        # sample from the given range
        # note: we modify the masses in-place for all environments
        #   however, the setter takes care that only the masses of the specified environments are modified
        masses = _randomize_prop_by_op(
            masses, mass_distribution_params, env_ids, body_ids, operation=operation, distribution=distribution
        )
        masses = torch.clamp(masses, min=min_mass)  # ensure masses are positive

        # set the mass into the physics simulation
        # note: backends expect partial data of shape (len(env_ids), len(body_ids))
        self.asset.set_masses_index(masses=masses[env_rows, body_ids], body_ids=body_ids, env_ids=env_ids)

        # recompute inertia tensors if needed
        if recompute_inertia:
            # compute the ratios of the new masses to the initial masses
            ratios = masses[env_rows, body_ids] / self.default_mass[env_rows, body_ids]
            # scale the inertia tensors by the the ratios
            # since mass randomization is done on default values, we can use the default inertia tensors
            inertias = self.asset.data.body_inertia.torch.clone()
            # inertia has shape: (num_envs, num_bodies, 9) for all assets
            inertias[env_rows, body_ids] = self.default_inertia[env_rows, body_ids] * ratios[..., None]
            # set the inertia tensors into the physics simulation
            # note: backends expect partial data of shape (len(env_ids), len(body_ids), 9)
            self.asset.set_inertias_index(inertias=inertias[env_rows, body_ids], body_ids=body_ids, env_ids=env_ids)


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
        from ...assets import BaseArticulation, BaseRigidObject, BaseRigidObjectCollection

        super().__init__(cfg, env)

        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: RigidObject | Articulation = env.scene[self.asset_cfg.name]

        if not isinstance(self.asset, (BaseRigidObject, BaseArticulation, BaseRigidObjectCollection)):
            raise ValueError(
                f"Randomization term 'randomize_rigid_body_inertia' not supported for asset: '{self.asset_cfg.name}'"
                f" with type: '{type(self.asset)}'."
            )

        # check for valid operation
        if cfg.params["operation"] == "scale":
            if "inertia_distribution_params" in cfg.params:
                _validate_scale_range(
                    cfg.params["inertia_distribution_params"], "inertia_distribution_params", allow_zero=False
                )
        elif cfg.params["operation"] not in ("abs", "add"):
            raise ValueError(
                f"Randomization term 'randomize_rigid_body_inertia' does not support operation:"
                f" '{cfg.params['operation']}'."
            )

        self.default_inertia = None
        # cache inertia indices: diagonal (0, 4, 8) for regularization, or all elements
        diagonal_only = cfg.params.get("diagonal_only", True)
        self._inertia_idx = torch.tensor([0, 4, 8], device=self.asset.device) if diagonal_only else slice(None)

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | slice | None,
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
        # store default inertia on first call for repeatable randomization
        if self.default_inertia is None:
            self.default_inertia = self.asset.data.body_inertia.torch.clone()

        # resolve environment ids
        if env_ids is None:
            env_ids = slice(None)
        # resolve body indices
        body_ids = self.asset_cfg.body_ids
        # index rows and columns jointly only when both are tensors; with a slice the result is already 2D
        env_rows = env_ids[:, None] if not isinstance(env_ids, slice) and not isinstance(body_ids, slice) else env_ids

        # get default inertias for affected envs/bodies (advanced indexing creates a copy)
        # shape: (len(env_ids), len(body_ids), 9)
        inertias = self.default_inertia[env_rows, body_ids]

        # resolve the distribution function
        if distribution == "uniform":
            dist_fn = math_utils.sample_uniform
        elif distribution == "log_uniform":
            dist_fn = math_utils.sample_log_uniform
        elif distribution == "gaussian":
            dist_fn = math_utils.sample_gaussian
        else:
            raise NotImplementedError(
                f"Unknown distribution: '{distribution}' for inertia randomization."
                " Please use 'uniform', 'log_uniform', or 'gaussian'."
            )

        # sample random values once per (env, body) - shape: (len(env_ids), len(body_ids))
        random_values = dist_fn(*inertia_distribution_params, inertias.shape[:2], device=self.asset.device)

        # apply the operation with the SAME random value per body
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

    The CoM pose (position and quaternion) is passed to ``set_coms_index``; backends that model the CoM as a
    position only (Newton) ignore the orientation.

    .. note::
        On Newton (MuJoCo Warp), runtime CoM changes may cause simulation instability because
        ``notify_model_changed(BODY_INERTIAL_PROPERTIES)`` does not fully recompute the mass matrix after
        ``body_ipos`` changes. Use with caution until this is fixed upstream.
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
        self.default_com = None

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | slice | None,
        com_range: dict[str, tuple[float, float]],
        asset_cfg: SceneEntityCfg,
    ):
        # store default CoM on first call for repeatable randomization
        if self.default_com is None:
            self.default_com = self.asset.data.body_com_pose_b.torch.clone()

        # resolve environment ids
        if env_ids is None:
            env_ids = slice(None)
        # resolve body indices
        body_ids = self.asset_cfg.body_ids
        # index rows and columns jointly only when both are tensors; with a slice the result is already 2D
        env_rows = env_ids[:, None] if not isinstance(env_ids, slice) and not isinstance(body_ids, slice) else env_ids

        # sample random CoM values
        ranges = _range_tensor(com_range, ("x", "y", "z"), self.asset.device)
        num_envs = len(range(env.num_envs)[env_ids]) if isinstance(env_ids, slice) else len(env_ids)
        rand_samples = math_utils.sample_uniform(
            ranges[:, 0], ranges[:, 1], (num_envs, 3), device=self.asset.device
        ).unsqueeze(1)

        # start from defaults and add random offsets
        coms = self.default_com.clone()
        coms[env_rows, body_ids, :3] += rand_samples

        # note: pass partial data of shape (num_envs, len(body_ids), ...) to match the API
        self.asset.set_coms_index(coms=coms[env_rows, body_ids], body_ids=body_ids, env_ids=env_ids)


class randomize_rigid_body_collider_offsets(ManagerTermBase):
    """Randomize the collider parameters of rigid bodies by setting random values.

    This function allows randomizing the collider parameters of the asset, such as rest and contact offsets.
    These correspond to the physics engine collider properties that affect collision checking.

    The active backend determines how offsets are written:

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

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv) -> None:
        """Initialize the implementation for the active physics backend.

        Args:
            cfg: Event configuration.
            env: Environment owning this term.
        """
        super().__init__(cfg, env)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: RigidObject | Articulation = env.scene[self.asset_cfg.name]

        from ...assets import BaseArticulation, BaseRigidObject  # noqa: PLC0415

        if not isinstance(self.asset, (BaseRigidObject, BaseArticulation)):
            raise ValueError(
                f"Randomization term 'randomize_rigid_body_collider_offsets' not supported for asset:"
                f" '{self.asset_cfg.name}' with type: '{type(self.asset)}'."
            )

        self._impl = _get_backend_events(env).randomize_rigid_body_collider_offsets(cfg, env)

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | slice | None,
        asset_cfg: SceneEntityCfg,
        rest_offset_distribution_params: tuple[float, float] | None = None,
        contact_offset_distribution_params: tuple[float, float] | None = None,
        distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
    ) -> None:
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

    The active backend determines whether gravity is shared across environments:

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

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv) -> None:
        """Initialize the implementation for the active physics backend.

        Args:
            cfg: Event configuration.
            env: Environment owning this term.
        """
        super().__init__(cfg, env)
        self._impl = _get_backend_events(env).randomize_physics_scene_gravity(cfg, env)

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | slice | None,
        gravity_distribution_params: tuple[list[float], list[float]],
        operation: Literal["add", "scale", "abs"],
        distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
    ) -> None:
        self._impl(env, env_ids, gravity_distribution_params, operation, distribution)


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
        from ...actuators import IdealPDActuator  # noqa: PLC0415

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
        from ...actuators.newton import read_group_parameter  # noqa: PLC0415

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

        # check for valid operation
        if cfg.params["operation"] == "scale":
            if "stiffness_distribution_params" in cfg.params:
                _validate_scale_range(
                    cfg.params["stiffness_distribution_params"], "stiffness_distribution_params", allow_zero=False
                )
            if "damping_distribution_params" in cfg.params:
                _validate_scale_range(cfg.params["damping_distribution_params"], "damping_distribution_params")
        elif cfg.params["operation"] not in ("abs", "add"):
            raise ValueError(
                "Randomization term 'randomize_actuator_gains' does not support operation:"
                f" '{cfg.params['operation']}'."
            )

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | slice | None,
        asset_cfg: SceneEntityCfg,
        stiffness_distribution_params: tuple[float, float] | None = None,
        damping_distribution_params: tuple[float, float] | None = None,
        operation: Literal["add", "scale", "abs"] = "abs",
        distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
    ):
        from ...actuators.newton import write_group_parameter  # noqa: PLC0415

        # Resolve environment ids
        if env_ids is None:
            env_ids = slice(None)

        def randomize(data: torch.Tensor, params: tuple[float, float]) -> torch.Tensor:
            return _randomize_prop_by_op(
                data, params, dim_0_ids=None, dim_1_ids=actuator_indices, operation=operation, distribution=distribution
            )

        for actuator_name, actuator in self._gain_actuators.items():
            group_joint_indices = self._group_joint_indices[actuator_name]
            if isinstance(self.asset_cfg.joint_ids, slice):
                # we take all the joints of the actuator
                actuator_indices = slice(None)
                if isinstance(group_joint_indices, slice):
                    global_indices = slice(None)
                elif isinstance(group_joint_indices, torch.Tensor):
                    global_indices = group_joint_indices.to(self.asset.device)
                else:
                    raise TypeError("Actuator joint indices must be a slice or a torch.Tensor.")
            elif isinstance(group_joint_indices, slice):
                # we take the joints defined in the asset config
                global_indices = actuator_indices = self.asset_cfg.joint_ids
            else:
                # we take the intersection of the actuator joints and the asset config joints
                actuator_joint_indices = group_joint_indices
                asset_joint_ids = self.asset_cfg.joint_ids
                # the indices of the joints in the actuator that have to be randomized
                actuator_indices = torch.nonzero(torch.isin(actuator_joint_indices, asset_joint_ids)).view(-1)
                if len(actuator_indices) == 0:
                    continue
                # maps actuator indices that have to be randomized to global joint indices
                global_indices = actuator_joint_indices[actuator_indices]
            if isinstance(global_indices, slice):
                writer_joint_ids = torch.arange(self.asset.num_joints, device=self.asset.device, dtype=torch.long)
            else:
                writer_joint_ids = global_indices.to(device=self.asset.device, dtype=torch.long)
            is_native = actuator_name in self._native_group_names
            # Native group writes are group-targeted: they take positions within the group's joints.
            group_columns = None if isinstance(actuator_indices, slice) else actuator_indices
            # Randomize stiffness
            if stiffness_distribution_params is not None:
                if is_native:
                    # Native gains are controller-owned; randomization always starts from the defaults.
                    stiffness = self.default_actuator_stiffness[actuator_name][env_ids].clone()
                else:
                    stiffness = actuator.stiffness[env_ids].clone()
                    stiffness[:, actuator_indices] = self.default_actuator_stiffness[actuator_name][env_ids][
                        :, actuator_indices
                    ]
                randomize(stiffness, stiffness_distribution_params)
                if getattr(actuator, "is_implicit_model", False):
                    self.asset.write_joint_stiffness_to_sim_index(
                        stiffness=stiffness[:, actuator_indices], joint_ids=writer_joint_ids, env_ids=env_ids
                    )
                elif is_native:
                    write_group_parameter(
                        self.asset.actuators,
                        actuator_name,
                        "controller",
                        "kp",
                        values=stiffness[:, actuator_indices],
                        env_ids=env_ids,
                        joint_ids=group_columns,
                    )
                else:
                    actuator.stiffness[env_ids] = stiffness
            # Randomize damping
            if damping_distribution_params is not None:
                if is_native:
                    damping = self.default_actuator_damping[actuator_name][env_ids].clone()
                else:
                    damping = actuator.damping[env_ids].clone()
                    damping[:, actuator_indices] = self.default_actuator_damping[actuator_name][env_ids][
                        :, actuator_indices
                    ]
                randomize(damping, damping_distribution_params)
                if getattr(actuator, "is_implicit_model", False):
                    self.asset.write_joint_damping_to_sim_index(
                        damping=damping[:, actuator_indices], joint_ids=writer_joint_ids, env_ids=env_ids
                    )
                elif is_native:
                    write_group_parameter(
                        self.asset.actuators,
                        actuator_name,
                        "controller",
                        "kd",
                        values=damping[:, actuator_indices],
                        env_ids=env_ids,
                        joint_ids=group_columns,
                    )
                else:
                    actuator.damping[env_ids] = damping


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

        # cache default values
        self.default_joint_friction_coeff = self.asset.data.joint_friction_coeff.torch.clone()
        self.default_joint_armature = self.asset.data.joint_armature.torch.clone()
        self.default_joint_pos_limits = self.asset.data.joint_pos_limits.torch.clone()
        self.default_viscous_joint_friction_coeff = self.asset.data.joint_viscous_friction_coeff.torch.clone()
        # dynamic friction is only exposed by backends that model it (e.g. not Newton)
        self.default_dynamic_joint_friction_coeff = None
        if hasattr(self.asset.data, "joint_dynamic_friction_coeff"):
            self.default_dynamic_joint_friction_coeff = self.asset.data.joint_dynamic_friction_coeff.torch.clone()

        # check for valid operation
        if cfg.params["operation"] == "scale":
            if "friction_distribution_params" in cfg.params:
                _validate_scale_range(cfg.params["friction_distribution_params"], "friction_distribution_params")
            if "armature_distribution_params" in cfg.params:
                _validate_scale_range(cfg.params["armature_distribution_params"], "armature_distribution_params")
        elif cfg.params["operation"] not in ("abs", "add"):
            raise ValueError(
                "Randomization term 'randomize_joint_parameters' does not support operation:"
                f" '{cfg.params['operation']}'."
            )

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | slice | None,
        asset_cfg: SceneEntityCfg,
        friction_distribution_params: tuple[float, float] | None = None,
        armature_distribution_params: tuple[float, float] | None = None,
        lower_limit_distribution_params: tuple[float, float] | None = None,
        upper_limit_distribution_params: tuple[float, float] | None = None,
        operation: Literal["add", "scale", "abs"] = "abs",
        distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
    ):
        # resolve environment ids
        if env_ids is None:
            env_ids = slice(None)

        # resolve joint indices
        joint_ids = self.asset_cfg.joint_ids

        if not isinstance(env_ids, slice) and joint_ids != slice(None):
            env_ids_for_slice = env_ids[:, None]
        else:
            env_ids_for_slice = env_ids

        # sample joint properties from the given ranges and set into the physics simulation
        # joint friction coefficient
        if friction_distribution_params is not None:
            friction_coeff = _randomize_prop_by_op(
                self.default_joint_friction_coeff.clone(),
                friction_distribution_params,
                env_ids,
                joint_ids,
                operation=operation,
                distribution=distribution,
            )

            # ensure the friction coefficient is non-negative
            friction_coeff = torch.clamp(friction_coeff, min=0.0)

            # Always set static friction (indexed once)
            static_friction_coeff = friction_coeff[env_ids_for_slice, joint_ids]

            viscous_friction_coeff = _randomize_prop_by_op(
                self.default_viscous_joint_friction_coeff.clone(),
                friction_distribution_params,
                env_ids,
                joint_ids,
                operation=operation,
                distribution=distribution,
            )
            viscous_friction_coeff = torch.clamp(viscous_friction_coeff, min=0.0)
            viscous_friction_coeff = viscous_friction_coeff[env_ids_for_slice, joint_ids]

            dynamic_friction_coeff = None
            if self.default_dynamic_joint_friction_coeff is not None:
                dynamic_friction_coeff = _randomize_prop_by_op(
                    self.default_dynamic_joint_friction_coeff.clone(),
                    friction_distribution_params,
                    env_ids,
                    joint_ids,
                    operation=operation,
                    distribution=distribution,
                )
                # dynamic friction is non-negative and at most the static friction
                dynamic_friction_coeff = torch.minimum(torch.clamp(dynamic_friction_coeff, min=0.0), friction_coeff)
                dynamic_friction_coeff = dynamic_friction_coeff[env_ids_for_slice, joint_ids]

            self.asset.write_joint_friction_coefficient_to_sim_index(
                joint_friction_coeff=static_friction_coeff,
                joint_dynamic_friction_coeff=dynamic_friction_coeff,
                joint_viscous_friction_coeff=viscous_friction_coeff,
                joint_ids=joint_ids,
                env_ids=env_ids,
            )

        if armature_distribution_params is not None:
            armature = _randomize_prop_by_op(
                self.asset.data.default_joint_armature.torch.clone(),
                armature_distribution_params,
                env_ids,
                joint_ids,
                operation=operation,
                distribution=distribution,
            )
            self.asset.write_joint_armature_to_sim(
                armature[env_ids_for_slice, joint_ids], joint_ids=joint_ids, env_ids=env_ids
            )

        if lower_limit_distribution_params is not None or upper_limit_distribution_params is not None:
            joint_pos_limits = self.default_joint_pos_limits.clone()
            # -- randomize the lower limits
            if lower_limit_distribution_params is not None:
                joint_pos_limits[..., 0] = _randomize_prop_by_op(
                    joint_pos_limits[..., 0],
                    lower_limit_distribution_params,
                    env_ids,
                    joint_ids,
                    operation=operation,
                    distribution=distribution,
                )
            # -- randomize the upper limits
            if upper_limit_distribution_params is not None:
                joint_pos_limits[..., 1] = _randomize_prop_by_op(
                    joint_pos_limits[..., 1],
                    upper_limit_distribution_params,
                    env_ids,
                    joint_ids,
                    operation=operation,
                    distribution=distribution,
                )

            # extract the position limits for the concerned joints
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
        # check for valid operation
        if cfg.params["operation"] == "scale":
            if "stiffness_distribution_params" in cfg.params:
                _validate_scale_range(
                    cfg.params["stiffness_distribution_params"], "stiffness_distribution_params", allow_zero=False
                )
            if "damping_distribution_params" in cfg.params:
                _validate_scale_range(cfg.params["damping_distribution_params"], "damping_distribution_params")
            if "limit_stiffness_distribution_params" in cfg.params:
                _validate_scale_range(
                    cfg.params["limit_stiffness_distribution_params"], "limit_stiffness_distribution_params"
                )
        elif cfg.params["operation"] not in ("abs", "add"):
            raise ValueError(
                "Randomization term 'randomize_fixed_tendon_parameters' does not support operation:"
                f" '{cfg.params['operation']}'."
            )

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | slice | None,
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
        # resolve environment ids
        if env_ids is None:
            env_ids = slice(None)

        # resolve joint indices
        tendon_ids = self.asset_cfg.fixed_tendon_ids
        # index rows and columns jointly only when both are tensors; with a slice the result is already 2D
        env_ids_for_slice = (
            env_ids[:, None] if not isinstance(env_ids, slice) and isinstance(tendon_ids, torch.Tensor) else env_ids
        )

        # sample tendon properties from the given ranges and set into the physics simulation
        # stiffness
        if stiffness_distribution_params is not None:
            stiffness = _randomize_prop_by_op(
                self.asset.data.fixed_tendon_stiffness.torch.clone(),
                stiffness_distribution_params,
                env_ids,
                tendon_ids,
                operation=operation,
                distribution=distribution,
            )
            self.asset.set_fixed_tendon_stiffness_index(
                stiffness=stiffness[env_ids_for_slice, tendon_ids], fixed_tendon_ids=tendon_ids, env_ids=env_ids
            )

        if damping_distribution_params is not None:
            damping = _randomize_prop_by_op(
                self.asset.data.fixed_tendon_damping.torch.clone(),
                damping_distribution_params,
                env_ids,
                tendon_ids,
                operation=operation,
                distribution=distribution,
            )
            self.asset.set_fixed_tendon_damping_index(
                damping=damping[env_ids_for_slice, tendon_ids], fixed_tendon_ids=tendon_ids, env_ids=env_ids
            )

        # limit stiffness
        if limit_stiffness_distribution_params is not None:
            limit_stiffness = _randomize_prop_by_op(
                self.asset.data.fixed_tendon_limit_stiffness.torch.clone(),
                limit_stiffness_distribution_params,
                env_ids,
                tendon_ids,
                operation=operation,
                distribution=distribution,
            )
            self.asset.set_fixed_tendon_limit_stiffness_index(
                limit_stiffness=limit_stiffness[env_ids_for_slice, tendon_ids],
                fixed_tendon_ids=tendon_ids,
                env_ids=env_ids,
            )

        if lower_limit_distribution_params is not None or upper_limit_distribution_params is not None:
            limit = self.asset.data.fixed_tendon_pos_limits.torch.clone()
            # -- lower limit
            if lower_limit_distribution_params is not None:
                limit[..., 0] = _randomize_prop_by_op(
                    limit[..., 0],
                    lower_limit_distribution_params,
                    env_ids,
                    tendon_ids,
                    operation=operation,
                    distribution=distribution,
                )
            # -- upper limit
            if upper_limit_distribution_params is not None:
                limit[..., 1] = _randomize_prop_by_op(
                    limit[..., 1],
                    upper_limit_distribution_params,
                    env_ids,
                    tendon_ids,
                    operation=operation,
                    distribution=distribution,
                )

            # check if the limits are valid
            tendon_limits = limit[env_ids_for_slice, tendon_ids]
            if (tendon_limits[..., 0] > tendon_limits[..., 1]).any():
                raise ValueError(
                    "Randomization term 'randomize_fixed_tendon_parameters' is setting lower tendon limits that are"
                    " greater than upper tendon limits."
                )
            self.asset.set_fixed_tendon_position_limit_index(
                limit=tendon_limits, fixed_tendon_ids=tendon_ids, env_ids=env_ids
            )

        if rest_length_distribution_params is not None:
            rest_length = _randomize_prop_by_op(
                self.asset.data.fixed_tendon_rest_length.torch.clone(),
                rest_length_distribution_params,
                env_ids,
                tendon_ids,
                operation=operation,
                distribution=distribution,
            )
            self.asset.set_fixed_tendon_rest_length_index(
                rest_length=rest_length[env_ids_for_slice, tendon_ids], fixed_tendon_ids=tendon_ids, env_ids=env_ids
            )
        # offset
        if offset_distribution_params is not None:
            offset = _randomize_prop_by_op(
                self.asset.data.fixed_tendon_offset.torch.clone(),
                offset_distribution_params,
                env_ids,
                tendon_ids,
                operation=operation,
                distribution=distribution,
            )
            self.asset.set_fixed_tendon_offset_index(
                offset=offset[env_ids_for_slice, tendon_ids], fixed_tendon_ids=tendon_ids, env_ids=env_ids
            )

        self.asset.write_fixed_tendon_properties_to_sim_index(env_ids=env_ids)


def apply_external_force_torque(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | slice,
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
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    # resolve environment ids
    if env_ids is None:
        env_ids = slice(None)
    # resolve number of bodies
    num_bodies = asset.num_bodies if isinstance(asset_cfg.body_ids, slice) else len(asset_cfg.body_ids)

    # Skip force application if the wrench ranges are zero
    if force_range[0] == 0.0 and force_range[1] == 0.0 and torque_range[0] == 0.0 and torque_range[1] == 0.0:
        return

    # sample random forces and torques
    num_envs = len(range(env.num_envs)[env_ids]) if isinstance(env_ids, slice) else len(env_ids)
    size = (num_envs, num_bodies, 3)
    forces = math_utils.sample_uniform(*force_range, size, asset.device)
    torques = math_utils.sample_uniform(*torque_range, size, asset.device)
    # set the forces and torques into the buffers
    # note: these are only applied when you call: `asset.write_data_to_sim()`
    asset.permanent_wrench_composer.set_forces_and_torques_index(
        forces=forces,
        torques=torques,
        body_ids=asset_cfg.body_ids,
        env_ids=env_ids,
    )


def push_by_setting_velocity(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | slice,
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

    # velocities
    vel_w = asset.data.root_vel_w.torch[env_ids]
    # sample random velocities
    ranges = _range_tensor(velocity_range, ("x", "y", "z", "roll", "pitch", "yaw"), asset.device)
    vel_w += math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], vel_w.shape, device=asset.device)
    # set the velocities into the physics simulation
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
    """

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | slice,
        pose_range: dict[str, tuple[float, float]],
        velocity_range: dict[str, tuple[float, float]],
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ):
        asset: RigidObject | Articulation = env.scene[asset_cfg.name]
        # The selected defaults are read-only; slices need no copy.
        default_root_pose = asset.data.default_root_pose.torch[env_ids]
        default_root_vel = asset.data.default_root_vel.torch[env_ids]

        ranges = _range_tensor(pose_range, ("x", "y", "z", "roll", "pitch", "yaw"), asset.device)
        rand_samples = math_utils.sample_uniform(
            ranges[:, 0], ranges[:, 1], (default_root_pose.shape[0], 6), device=asset.device
        )

        positions = default_root_pose[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
        orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        orientations = math_utils.quat_mul(default_root_pose[:, 3:7], orientations_delta)
        # velocities
        ranges = _range_tensor(velocity_range, ("x", "y", "z", "roll", "pitch", "yaw"), asset.device)
        rand_samples = math_utils.sample_uniform(
            ranges[:, 0], ranges[:, 1], (default_root_pose.shape[0], 6), device=asset.device
        )

        velocities = default_root_vel + rand_samples
        asset.write_root_pose_to_sim_index(root_pose=torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
        asset.write_root_velocity_to_sim_index(root_velocity=velocities, env_ids=env_ids)


def reset_root_state_with_random_orientation(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | slice,
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
    # get default root state
    default_root_pose = asset.data.default_root_pose.torch[env_ids].clone()
    default_root_vel = asset.data.default_root_vel.torch[env_ids].clone()

    # poses
    ranges = _range_tensor(pose_range, ("x", "y", "z"), asset.device)
    rand_samples = math_utils.sample_uniform(
        ranges[:, 0], ranges[:, 1], (default_root_pose.shape[0], 3), device=asset.device
    )

    positions = default_root_pose[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples
    orientations = math_utils.random_orientation(default_root_pose.shape[0], device=asset.device)

    # velocities
    ranges = _range_tensor(velocity_range, ("x", "y", "z", "roll", "pitch", "yaw"), asset.device)
    rand_samples = math_utils.sample_uniform(
        ranges[:, 0], ranges[:, 1], (default_root_pose.shape[0], 6), device=asset.device
    )

    velocities = default_root_vel + rand_samples

    # set into the physics simulation
    asset.write_root_pose_to_sim_index(root_pose=torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim_index(root_velocity=velocities, env_ids=env_ids)


def reset_root_state_from_terrain(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | slice,
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

    valid_positions: torch.Tensor = terrain.flat_patches.get("init_pos")
    if valid_positions is None:
        raise ValueError(
            "The event term 'reset_root_state_from_terrain' requires valid flat patches under 'init_pos'."
            f" Found: {list(terrain.flat_patches.keys())}"
        )

    num_envs = len(range(env.num_envs)[env_ids]) if isinstance(env_ids, slice) else len(env_ids)

    # sample random valid poses
    ids = torch.randint(0, valid_positions.shape[2], size=(num_envs,), device=env.device)
    positions = valid_positions[terrain.terrain_levels[env_ids], terrain.terrain_types[env_ids], ids]
    positions += asset.data.default_root_pose.torch[env_ids, :3]
    # sample random orientations
    ranges = _range_tensor(pose_range, ("roll", "pitch", "yaw"), asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (num_envs, 3), device=asset.device)

    # convert to quaternions
    orientations = math_utils.quat_from_euler_xyz(rand_samples[:, 0], rand_samples[:, 1], rand_samples[:, 2])

    # sample random velocities
    ranges = _range_tensor(velocity_range, ("x", "y", "z", "roll", "pitch", "yaw"), asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (num_envs, 6), device=asset.device)

    velocities = asset.data.default_root_vel.torch[env_ids] + rand_samples

    # set into the physics simulation
    asset.write_root_pose_to_sim_index(root_pose=torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim_index(root_velocity=velocities, env_ids=env_ids)


def reset_joints_by_scale(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | slice,
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the robot joints by scaling the default position and velocity by the given ranges.

    This function samples random values from the given ranges and scales the default joint positions and velocities
    by these values. The scaled values are then set into the physics simulation.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # cast env_ids to allow broadcasting
    if not isinstance(env_ids, slice) and asset_cfg.joint_ids != slice(None):
        iter_env_ids = env_ids[:, None]
    else:
        iter_env_ids = env_ids

    # get default joint state
    joint_pos = asset.data.default_joint_pos.torch[iter_env_ids, asset_cfg.joint_ids].clone()
    joint_vel = asset.data.default_joint_vel.torch[iter_env_ids, asset_cfg.joint_ids].clone()

    # scale these values randomly
    joint_pos *= math_utils.sample_uniform(*position_range, joint_pos.shape, joint_pos.device)
    joint_vel *= math_utils.sample_uniform(*velocity_range, joint_vel.shape, joint_vel.device)

    # clamp joint pos to limits
    joint_pos_limits = asset.data.soft_joint_pos_limits.torch[iter_env_ids, asset_cfg.joint_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
    # clamp joint vel to limits
    joint_vel_limits = asset.data.soft_joint_vel_limits.torch[iter_env_ids, asset_cfg.joint_ids]
    joint_vel = joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)

    # set into the physics simulation
    asset.write_joint_position_to_sim_index(position=joint_pos, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)
    asset.write_joint_velocity_to_sim_index(velocity=joint_vel, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)


def reset_joints_by_offset(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | slice,
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the robot joints with offsets around the default position and velocity by the given ranges.

    This function samples random values from the given ranges and biases the default joint positions and velocities
    by these values. The biased values are then set into the physics simulation.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # cast env_ids to allow broadcasting
    if not isinstance(env_ids, slice) and asset_cfg.joint_ids != slice(None):
        iter_env_ids = env_ids[:, None]
    else:
        iter_env_ids = env_ids

    # get default joint state
    joint_pos = asset.data.default_joint_pos.torch[iter_env_ids, asset_cfg.joint_ids].clone()
    joint_vel = asset.data.default_joint_vel.torch[iter_env_ids, asset_cfg.joint_ids].clone()

    # bias these values randomly
    joint_pos += math_utils.sample_uniform(*position_range, joint_pos.shape, joint_pos.device)
    joint_vel += math_utils.sample_uniform(*velocity_range, joint_vel.shape, joint_vel.device)

    # clamp joint pos to limits
    joint_pos_limits = asset.data.soft_joint_pos_limits.torch[iter_env_ids, asset_cfg.joint_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
    # clamp joint vel to limits
    joint_vel_limits = asset.data.soft_joint_vel_limits.torch[iter_env_ids, asset_cfg.joint_ids]
    joint_vel = joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)

    # set into the physics simulation
    asset.write_joint_position_to_sim_index(position=joint_pos, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)
    asset.write_joint_velocity_to_sim_index(velocity=joint_vel, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)


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
        default_joint_pos = self._asset.data.default_joint_pos.torch[0]
        default_joint_vel = self._asset.data.default_joint_vel.torch[0]

        # create buffers to store the joint position range
        self._pos_ranges = self._asset.data.soft_joint_pos_limits.torch[0].clone()
        # parse joint position ranges
        pos_joint_ids = []
        for joint_name, joint_range in cfg.params["position_range"].items():
            # find the joint ids
            joint_ids = self._asset.find_joints(joint_name)[0]
            pos_joint_ids.extend(joint_ids)

            # set the joint position ranges based on the given values
            if operation == "abs":
                if joint_range[0] is not None:
                    self._pos_ranges[joint_ids, 0] = joint_range[0]
                if joint_range[1] is not None:
                    self._pos_ranges[joint_ids, 1] = joint_range[1]
            else:  # operation == "scale"
                if joint_range[0] is not None:
                    self._pos_ranges[joint_ids, 0] *= joint_range[0]
                if joint_range[1] is not None:
                    self._pos_ranges[joint_ids, 1] *= joint_range[1]
            # add the default offset
            if use_default_offset:
                self._pos_ranges[joint_ids] += default_joint_pos[joint_ids].unsqueeze(1)

        # store the joint pos ids (used later to sample the joint positions)
        self._pos_joint_ids = torch.tensor(pos_joint_ids, device=self._pos_ranges.device)
        self._pos_ranges = self._pos_ranges[self._pos_joint_ids]

        # create buffers to store the joint velocity range
        soft_joint_vel_limits_torch = self._asset.data.soft_joint_vel_limits.torch[0]
        self._vel_ranges = torch.stack([-soft_joint_vel_limits_torch, soft_joint_vel_limits_torch], dim=1)
        # parse joint velocity ranges
        vel_joint_ids = []
        for joint_name, joint_range in cfg.params["velocity_range"].items():
            # find the joint ids
            joint_ids = self._asset.find_joints(joint_name)[0]
            vel_joint_ids.extend(joint_ids)

            # set the joint velocity ranges based on the given values
            if operation == "abs":
                if joint_range[0] is not None:
                    self._vel_ranges[joint_ids, 0] = joint_range[0]
                if joint_range[1] is not None:
                    self._vel_ranges[joint_ids, 1] = joint_range[1]
            else:  # operation == "scale"
                if joint_range[0] is not None:
                    self._vel_ranges[joint_ids, 0] = joint_range[0] * self._vel_ranges[joint_ids, 0]
                if joint_range[1] is not None:
                    self._vel_ranges[joint_ids, 1] = joint_range[1] * self._vel_ranges[joint_ids, 1]
            # add the default offset
            if use_default_offset:
                self._vel_ranges[joint_ids] += default_joint_vel[joint_ids].unsqueeze(1)

        # store the joint vel ids (used later to sample the joint positions)
        self._vel_joint_ids = torch.tensor(vel_joint_ids, device=self._vel_ranges.device)
        self._vel_ranges = self._vel_ranges[self._vel_joint_ids]

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | slice,
        position_range: dict[str, tuple[float | None, float | None]],
        velocity_range: dict[str, tuple[float | None, float | None]],
        use_default_offset: bool = False,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        operation: Literal["abs", "scale"] = "abs",
    ):
        # get default joint state
        joint_pos = self._asset.data.default_joint_pos.torch[env_ids].clone()
        joint_vel = self._asset.data.default_joint_vel.torch[env_ids].clone()

        if len(self._pos_joint_ids) > 0:
            joint_pos_shape = (joint_pos.shape[0], len(self._pos_joint_ids))
            joint_pos[:, self._pos_joint_ids] = math_utils.sample_uniform(
                self._pos_ranges[:, 0], self._pos_ranges[:, 1], joint_pos_shape, device=joint_pos.device
            )
            joint_pos_limits = self._asset.data.soft_joint_pos_limits.torch[0, self._pos_joint_ids]
            joint_pos = joint_pos.clamp(joint_pos_limits[:, 0], joint_pos_limits[:, 1])

        if len(self._vel_joint_ids) > 0:
            joint_vel_shape = (joint_vel.shape[0], len(self._vel_joint_ids))
            joint_vel[:, self._vel_joint_ids] = math_utils.sample_uniform(
                self._vel_ranges[:, 0], self._vel_ranges[:, 1], joint_vel_shape, device=joint_vel.device
            )
            joint_vel_limits = self._asset.data.soft_joint_vel_limits.torch[0, self._vel_joint_ids]
            joint_vel = joint_vel.clamp(-joint_vel_limits, joint_vel_limits)

        self._asset.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
        self._asset.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)


def reset_nodal_state_uniform(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | slice,
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
    # get default root state
    nodal_state = asset.data.default_nodal_state_w.torch[env_ids].clone()

    # position
    ranges = _range_tensor(position_range, ("x", "y", "z"), asset.device)
    rand_samples = math_utils.sample_uniform(
        ranges[:, 0], ranges[:, 1], (nodal_state.shape[0], 1, 3), device=asset.device
    )

    nodal_state[..., :3] += rand_samples

    # velocities
    ranges = _range_tensor(velocity_range, ("x", "y", "z"), asset.device)
    rand_samples = math_utils.sample_uniform(
        ranges[:, 0], ranges[:, 1], (nodal_state.shape[0], 1, 3), device=asset.device
    )

    nodal_state[..., 3:] += rand_samples

    # set into the physics simulation
    asset.write_nodal_state_to_sim(nodal_state, env_ids=env_ids)


def reset_scene_to_default(env: ManagerBasedEnv, env_ids: torch.Tensor | slice, reset_joint_targets: bool = False):
    """Reset the scene to the default state specified in the scene configuration.

    If :attr:`reset_joint_targets` is True, the joint position and velocity targets of the articulations are
    also reset to their default values. This might be useful for some cases to clear out any previously set targets.
    However, this is not the default behavior as based on our experience, it is not always desired to reset
    targets to default values, especially when the targets should be handled by action terms and not event terms.
    """
    # rigid bodies
    for rigid_object in env.scene.rigid_objects.values():
        # obtain default and deal with the offset for env origins
        default_root_pose = rigid_object.data.default_root_pose.torch[env_ids].clone()
        default_root_vel = rigid_object.data.default_root_vel.torch[env_ids].clone()
        default_root_pose[:, :3] += env.scene.env_origins[env_ids]
        # set into the physics simulation
        rigid_object.write_root_pose_to_sim_index(root_pose=default_root_pose, env_ids=env_ids)
        rigid_object.write_root_velocity_to_sim_index(root_velocity=default_root_vel, env_ids=env_ids)
    # articulations
    for articulation_asset in env.scene.articulations.values():
        # obtain default and deal with the offset for env origins
        default_root_pose = articulation_asset.data.default_root_pose.torch[env_ids].clone()
        default_root_vel = articulation_asset.data.default_root_vel.torch[env_ids].clone()
        default_root_pose[:, :3] += env.scene.env_origins[env_ids]
        # set into the physics simulation
        articulation_asset.write_root_pose_to_sim_index(root_pose=default_root_pose, env_ids=env_ids)
        articulation_asset.write_root_velocity_to_sim_index(root_velocity=default_root_vel, env_ids=env_ids)
        # obtain default joint positions
        default_joint_pos = articulation_asset.data.default_joint_pos.torch[env_ids].clone()
        default_joint_vel = articulation_asset.data.default_joint_vel.torch[env_ids].clone()
        # set into the physics simulation
        articulation_asset.write_joint_position_to_sim_index(position=default_joint_pos, env_ids=env_ids)
        articulation_asset.write_joint_velocity_to_sim_index(velocity=default_joint_vel, env_ids=env_ids)
        # reset joint targets if required
        if reset_joint_targets:
            articulation_asset.set_joint_position_target_index(target=default_joint_pos, env_ids=env_ids)
            articulation_asset.set_joint_velocity_target_index(target=default_joint_vel, env_ids=env_ids)
    # cable objects
    for cable_object in env.scene.cable_objects.values():
        segment_pose = cable_object.data.default_segment_pose_w.torch[env_ids].clone()
        segment_velocity = cable_object.data.default_segment_velocity_w.torch[env_ids].clone()
        cable_object.write_segment_pose_to_sim_index(segment_pose=segment_pose, env_ids=env_ids)
        cable_object.write_segment_velocity_to_sim_index(segment_velocity=segment_velocity, env_ids=env_ids)
    # deformable objects
    for deformable_object in env.scene.deformable_objects.values():
        # obtain default and set into the physics simulation
        nodal_state = deformable_object.data.default_nodal_state_w.torch[env_ids].clone()
        deformable_object.write_nodal_state_to_sim(nodal_state, env_ids=env_ids)


class randomize_visual_texture_material(ManagerTermBase):
    """Randomize USD mesh textures with Isaac Sim Replicator.

    Textures are projected onto matched prims and rotated by the sampled angle [rad].
    The default prim pattern is ``{asset_prim_path}/{body_name}/visuals``; assets
    without that layout use all descendants of the asset root.

    Requires Kit and ``InteractiveSceneCfg.replicate_physics=False``. All matched
    prims are updated; ``env_ids`` does not restrict the update.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv) -> None:
        """Select the Isaac Sim visual implementation.

        Args:
            cfg: Event configuration.
            env: Environment owning this term.
        """
        super().__init__(cfg, env)
        from isaaclab_physx.envs.mdp import events

        self._impl = events.randomize_visual_texture_material(cfg, env)

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | slice,
        event_name: str,
        asset_cfg: SceneEntityCfg,
        texture_paths: list[str],
        texture_rotation: tuple[float, float] = (0.0, 0.0),
    ) -> None:
        self._impl(env, env_ids, event_name, asset_cfg, texture_paths, texture_rotation)

    @property
    def material_prims(self) -> list[Usd.Prim]:
        """USD materials created by the term."""
        return self._impl.material_prims

    @material_prims.setter
    def material_prims(self, value: list[Usd.Prim]) -> None:
        self._impl.material_prims = value

    @property
    def texture_rng(self) -> ReplicatorRNG:
        """Replicator random number generator."""
        return self._impl.texture_rng

    @texture_rng.setter
    def texture_rng(self, value: ReplicatorRNG) -> None:
        self._impl.texture_rng = value


class randomize_visual_color(ManagerTermBase):
    """Randomize USD mesh colors with Isaac Sim Replicator.

    Colors accept RGB tuples or a dictionary of ``r``, ``g``, and ``b`` ranges.
    ``mesh_name`` selects a path relative to the asset root. Otherwise, the term
    matches the selected bodies' ``visuals`` prims, falling back to all descendants.

    Requires Kit and ``InteractiveSceneCfg.replicate_physics=False``. All matched
    prims are updated; ``env_ids`` does not restrict the update.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv) -> None:
        """Select the Isaac Sim visual implementation.

        Args:
            cfg: Event configuration.
            env: Environment owning this term.
        """
        super().__init__(cfg, env)
        from isaaclab_physx.envs.mdp import events

        self._impl = events.randomize_visual_color(cfg, env)

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | slice,
        event_name: str,
        asset_cfg: SceneEntityCfg,
        colors: list[tuple[float, float, float]] | dict[str, tuple[float, float]],
        mesh_name: str = "",
    ) -> None:
        self._impl(env, env_ids, event_name, asset_cfg, colors, mesh_name)

    @property
    def material_prims(self) -> list[Usd.Prim]:
        """USD materials created by the term."""
        return self._impl.material_prims

    @material_prims.setter
    def material_prims(self, value: list[Usd.Prim]) -> None:
        self._impl.material_prims = value

    @property
    def color_rng(self) -> ReplicatorRNG:
        """Replicator random number generator."""
        return self._impl.color_rng

    @color_rng.setter
    def color_rng(self, value: ReplicatorRNG) -> None:
        self._impl.color_rng = value


"""
Internal helper functions.
"""


def _randomize_prop_by_op(
    data: torch.Tensor,
    distribution_parameters: tuple[float | torch.Tensor, float | torch.Tensor],
    dim_0_ids: torch.Tensor | slice | None,
    dim_1_ids: torch.Tensor | slice,
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"],
) -> torch.Tensor:
    """Perform data randomization based on the given operation and distribution.

    Args:
        data: The data tensor to be randomized. Shape is (dim_0, dim_1).
        distribution_parameters: The parameters for the distribution to sample values from.
        dim_0_ids: The indices of the first dimension to randomize.
        dim_1_ids: The indices of the second dimension to randomize.
        operation: The operation to perform on the data. Options: 'add', 'scale', 'abs'.
        distribution: The distribution to sample the random values from. Options: 'uniform', 'log_uniform'.

    Returns:
        The data tensor after randomization. Shape is (dim_0, dim_1).

    Raises:
        NotImplementedError: If the operation or distribution is not supported.
    """
    if dim_0_ids is None:
        dim_0_ids = slice(None)
    n_dim_0 = len(range(data.shape[0])[dim_0_ids]) if isinstance(dim_0_ids, slice) else len(dim_0_ids)
    n_dim_1 = len(range(data.shape[1])[dim_1_ids]) if isinstance(dim_1_ids, slice) else len(dim_1_ids)
    if not isinstance(dim_0_ids, slice) and not isinstance(dim_1_ids, slice):
        dim_0_ids = dim_0_ids[:, None]

    # resolve the distribution
    if distribution == "uniform":
        dist_fn = math_utils.sample_uniform
    elif distribution == "log_uniform":
        dist_fn = math_utils.sample_log_uniform
    elif distribution == "gaussian":
        dist_fn = math_utils.sample_gaussian
    else:
        raise NotImplementedError(
            f"Unknown distribution: '{distribution}' for joint properties randomization."
            " Please use 'uniform', 'log_uniform', 'gaussian'."
        )
    # perform the operation
    if operation == "add":
        data[dim_0_ids, dim_1_ids] += dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    elif operation == "scale":
        data[dim_0_ids, dim_1_ids] *= dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    elif operation == "abs":
        data[dim_0_ids, dim_1_ids] = dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
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
    """
    Validates a (low, high) tuple used in scale-based randomization.

    This function ensures the tuple follows expected rules when applying a 'scale'
    operation. It performs type and value checks, optionally allowing negative or
    zero lower bounds.

    Args:
        params (tuple[float, float] | None): The (low, high) range to validate. If None,
            validation is skipped.
        name (str): The name of the parameter being validated, used for error messages.
        allow_negative (bool, optional): If True, allows the lower bound to be negative.
            Defaults to False.
        allow_zero (bool, optional): If True, allows the lower bound to be zero.
            Defaults to True.

    Raises:
        TypeError: If `params` is not a tuple of two numbers.
        ValueError: If the lower bound is negative or zero when not allowed.
        ValueError: If the upper bound is less than the lower bound.

    Example:
        _validate_scale_range((0.5, 1.5), "mass_scale")
    """
    if params is None:  # caller didn’t request randomisation for this field
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


def _get_backend_events(env: ManagerBasedEnv) -> ModuleType:
    """Select event implementations from the simulation's resolved physics configuration."""
    physics_cfg = env.sim.cfg.physics
    # Task-specific subclasses may live outside the backend package.
    physics_cfg_modules = {cls.__module__.split(".")[0] for cls in type(physics_cfg).__mro__}
    if "isaaclab_newton" in physics_cfg_modules:
        from isaaclab_newton.envs.mdp import events
        from isaaclab_newton.physics import NewtonCfg

        expected_cfg_type = NewtonCfg
    elif "isaaclab_ov" in physics_cfg_modules:
        from isaaclab_ov.envs.mdp import events
        from isaaclab_ov.physics import OvPhysxCfg

        expected_cfg_type = OvPhysxCfg
    elif "isaaclab_physx" in physics_cfg_modules:
        from isaaclab_physx.envs.mdp import events
        from isaaclab_physx.physics import PhysxCfg

        expected_cfg_type = PhysxCfg
    else:
        raise NotImplementedError(f"Physics randomization is unsupported for {type(physics_cfg).__name__}.")
    if not isinstance(physics_cfg, expected_cfg_type):
        raise NotImplementedError(f"Physics randomization is unsupported for {type(physics_cfg).__name__}.")
    return events


class _GravityRandomization(ManagerTermBase):
    """Cache distribution bounds and sample gravity values."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv, device: str) -> None:
        super().__init__(cfg, env)
        self._sampling_device = device
        self._distribution = cfg.params.get("distribution", "uniform")
        if self._distribution not in ("uniform", "log_uniform", "gaussian"):
            raise NotImplementedError(f"Unknown gravity distribution: {self._distribution!r}.")
        if cfg.params["operation"] not in ("add", "scale", "abs"):
            raise NotImplementedError(f"Unknown gravity operation: {cfg.params['operation']!r}.")
        self._last_gravity_params = None

    def _sample_gravity(
        self,
        gravity: torch.Tensor,
        params: tuple[list[float], list[float]],
        operation: Literal["add", "scale", "abs"],
    ) -> torch.Tensor:
        # Curricula can change bounds without rebuilding the term.
        key = (tuple(params[0]), tuple(params[1]))
        if key != self._last_gravity_params:
            self._last_gravity_params = key
            self._bounds = torch.tensor(params, device=self._sampling_device, dtype=torch.float32)
        return _randomize_prop_by_op(
            gravity.clone(),
            (self._bounds[0], self._bounds[1]),
            None,
            slice(None),
            operation=operation,
            distribution=self._distribution,
        )
