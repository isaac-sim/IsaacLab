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

import importlib
import logging
import warnings
from typing import TYPE_CHECKING, Literal

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedEnv

# import logger
logger = logging.getLogger(__name__)


def _resolve_backend_event(impl_name: str, *args, **kwargs):
    """Instantiate the active physics backend's implementation of an event term.

    Resolves ``isaaclab_<backend>.envs.mdp.<impl_name>`` for the active physics manager and
    returns ``impl(*args, **kwargs)``. Each backend package owns every term it supports (OVPhysX
    re-exports the PhysX implementations it reuses), so resolution is uniform across backends.

    Args:
        impl_name: Name of the backend implementation class to resolve.
        args: Positional arguments forwarded to the implementation's constructor.
        kwargs: Keyword arguments forwarded to the implementation's constructor.

    Returns:
        An instance of the resolved backend implementation.

    Raises:
        ValueError: If the physics manager is unknown or the backend lacks the implementation.
    """
    from isaaclab.sim.simulation_context import SimulationContext

    manager_name = SimulationContext.instance().physics_manager.__name__.lower()
    # substring matching (matches the original events.py dispatch) so backend variants whose
    # class name embeds the backend (e.g. ``NewtonMJWarpManager``) still resolve. Order matters:
    # "ovphysx" contains "physx", so it must be tested before "physx".
    if "newton" in manager_name:
        backend = "newton"
    elif "ovphysx" in manager_name:
        backend = "ovphysx"
    elif "physx" in manager_name:
        backend = "physx"
    else:
        raise ValueError(f"Unknown physics manager: {manager_name!r}")

    module = importlib.import_module(f"isaaclab_{backend}.envs.mdp")
    try:
        impl_cls = getattr(module, impl_name)
    except AttributeError as exc:
        raise ValueError(
            f"The {backend!r} backend does not implement event term {impl_name!r} (looked in '{module.__name__}')."
        ) from exc
    return impl_cls(*args, **kwargs)


def randomize_rigid_body_scale(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    scale_range: tuple[float, float] | dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
    relative_child_path: str | None = None,
):
    """Randomize the scale of a rigid body asset in the USD stage.

    .. deprecated::
        This term is deprecated. Prefer generating separate USD files for each scale variant and
        using multi-asset spawning via :class:`~isaaclab.sim.MultiAssetSpawnerCfg`.

    This function modifies the "xformOp:scale" property of all the prims corresponding to the asset.
    The scale is authored directly on the USD stage, so it is backend-agnostic (works on every
    physics backend) but must run before the simulation starts playing (event mode ``"usd"``).

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
    import isaaclab.sim as sim_utils

    warnings.warn(
        "'randomize_rigid_body_scale' is deprecated. Prefer multi-asset spawning"
        " (see :class:`~isaaclab.sim.MultiAssetSpawnerCfg`) with per-scale USD variants instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    # check if sim is running
    if env.sim.is_playing():
        raise RuntimeError(
            "Randomizing scale while simulation is running leads to unpredictable behaviors."
            " Please ensure that the event term is called before the simulation starts by using the 'usd' mode."
        )

    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    if any(cls.__name__ == "Articulation" for cls in type(asset).__mro__):
        raise ValueError(
            "Scaling an articulation randomly is not supported, as it affects joint attributes and can cause"
            " unexpected behavior. To achieve different scales, we recommend generating separate USD files for"
            " each version of the articulation and using multi-asset spawning. For more details, refer to:"
            " https://isaac-sim.github.io/IsaacLab/main/source/how-to/multi_asset_spawning.html"
        )

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # acquire stage
    stage = env.sim.stage
    # resolve prim paths for spawning and cloning
    prim_paths = sim_utils.find_matching_prim_paths(asset.cfg.prim_path)

    # sample scale values
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

    # the scale is authored on the USD stage (backend-agnostic); ``pxr`` is only available with Kit
    from pxr import Gf, Sdf, UsdGeom, Vt  # noqa: PLC0415

    # use sdf changeblock for faster processing of USD properties
    with Sdf.ChangeBlock():
        for i, env_id in enumerate(env_ids):
            # path to prim to randomize
            prim_path = prim_paths[env_id] + relative_child_path
            # spawn single instance
            prim_spec = Sdf.CreatePrimInLayer(stage.GetRootLayer(), prim_path)

            # get the attribute to randomize
            scale_spec = prim_spec.GetAttributeAtPath(prim_path + ".xformOp:scale")
            # if the scale attribute does not exist, create it
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

    Automatically detects the active physics backend (PhysX or Newton) and delegates to
    the appropriate backend-specific implementation:

    - **PhysX**: Uses the 3-tuple material format (static_friction, dynamic_friction, restitution)
      with bucket-based assignment (limited to 64000 unique materials). Applied via the PhysX
      tensor API (``root_view.set_material_properties``).
    - **Newton**: Samples friction (mu) and restitution continuously per shape (no bucket
      limitation). Newton uses a single friction coefficient, so ``dynamic_friction_range``
      and ``num_buckets`` are ignored. Applied directly to Newton's view-level bindings.

    If the flag ``make_consistent`` is set to ``True``, the dynamic friction is set to be less than or equal to
    the static friction (PhysX only). This obeys the physics constraint on friction values.

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

        # extract the used quantities (to enable type-hinting)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: RigidObject | Articulation = env.scene[self.asset_cfg.name]

        if not isinstance(self.asset, (BaseRigidObject, BaseArticulation)):
            raise ValueError(
                f"Randomization term 'randomize_rigid_body_material' not supported for asset: '{self.asset_cfg.name}'"
                f" with type: '{type(self.asset)}'."
            )

        # construct the backend-specific implementation via the converter
        self._impl = _resolve_backend_event("RandomizeRigidBodyMaterial", cfg, env, self.asset, self.asset_cfg)

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

        # extract the used quantities (to enable type-hinting)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: RigidObject | Articulation = env.scene[self.asset_cfg.name]
        # check for valid operation
        if cfg.params["operation"] == "scale":
            if "mass_distribution_params" in cfg.params:
                validate_scale_range(
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
        env_ids: torch.Tensor | None,
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
            env_ids = torch.arange(env.scene.num_envs, device=self.asset.device, dtype=torch.int32)
        else:
            env_ids = env_ids.to(self.asset.device)

        # resolve body indices
        if self.asset_cfg.body_ids == slice(None):
            body_ids = torch.arange(self.asset.num_bodies, dtype=torch.int32, device=self.asset.device)
        else:
            body_ids = torch.tensor(self.asset_cfg.body_ids, dtype=torch.int32, device=self.asset.device)

        # get the current masses of the bodies (num_assets, num_bodies)
        masses = self.asset.data.body_mass.torch.clone()

        # apply randomization on default values
        # this is to make sure when calling the function multiple times, the randomization is applied on the
        # default values and not the previously randomized values
        masses[env_ids[:, None], body_ids] = self.default_mass[env_ids[:, None], body_ids].clone()

        # sample from the given range
        # note: we modify the masses in-place for all environments
        #   however, the setter takes care that only the masses of the specified environments are modified
        masses = randomize_prop_by_op(
            masses, mass_distribution_params, env_ids, body_ids, operation=operation, distribution=distribution
        )
        masses = torch.clamp(masses, min=min_mass)  # ensure masses are positive

        # set the mass into the physics simulation
        # note: backends expect partial data of shape (len(env_ids), len(body_ids))
        self.asset.set_masses_index(masses=masses[env_ids[:, None], body_ids], body_ids=body_ids, env_ids=env_ids)

        # recompute inertia tensors if needed
        if recompute_inertia:
            # compute the ratios of the new masses to the initial masses
            ratios = masses[env_ids[:, None], body_ids] / self.default_mass[env_ids[:, None], body_ids]
            # scale the inertia tensors by the the ratios
            # since mass randomization is done on default values, we can use the default inertia tensors
            inertias = self.asset.data.body_inertia.torch.clone()
            # inertia has shape: (num_envs, num_bodies, 9) for all assets
            inertias[env_ids[:, None], body_ids] = self.default_inertia[env_ids[:, None], body_ids] * ratios[..., None]
            # set the inertia tensors into the physics simulation
            # note: backends expect partial data of shape (len(env_ids), len(body_ids), 9)
            self.asset.set_inertias_index(
                inertias=inertias[env_ids[:, None], body_ids], body_ids=body_ids, env_ids=env_ids
            )


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

        # extract the used quantities (to enable type-hinting)
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
                validate_scale_range(
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
        # store default inertia on first call for repeatable randomization
        if self.default_inertia is None:
            self.default_inertia = self.asset.data.body_inertia.torch.clone()

        # resolve environment ids
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device=self.asset.device, dtype=torch.int32)
        else:
            env_ids = env_ids.to(self.asset.device)

        # resolve body indices
        if self.asset_cfg.body_ids == slice(None):
            body_ids = torch.arange(self.asset.num_bodies, dtype=torch.int32, device=self.asset.device)
        else:
            body_ids = torch.tensor(self.asset_cfg.body_ids, dtype=torch.int32, device=self.asset.device)

        # get default inertias for affected envs/bodies (advanced indexing creates a copy)
        # shape: (len(env_ids), len(body_ids), 9)
        inertias = self.default_inertia[env_ids[:, None], body_ids]

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
        random_values = dist_fn(*inertia_distribution_params, (len(env_ids), len(body_ids)), device=self.asset.device)

        # apply the operation with the SAME random value per body
        if operation == "add":
            inertias[:, :, self._inertia_idx] += random_values[..., None]
        elif operation == "scale":
            inertias[:, :, self._inertia_idx] *= random_values[..., None]
        elif operation == "abs":
            inertias[:, :, self._inertia_idx] = random_values[..., None]

        # set the inertia tensors into the physics simulation
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

        # construct the backend-specific implementation via the converter
        self._impl = _resolve_backend_event("RandomizeRigidBodyCom", self.asset, self.asset_cfg)

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        com_range: dict[str, tuple[float, float]],
        asset_cfg: SceneEntityCfg,
    ):
        self._impl(env, env_ids, com_range, asset_cfg)


class randomize_rigid_body_collider_offsets(ManagerTermBase):
    """Randomize the collider parameters of rigid bodies by setting random values.

    This function allows randomizing the collider parameters of the asset, such as rest and contact offsets.
    These correspond to the physics engine collider properties that affect collision checking.

    Automatically detects the active physics backend (PhysX or Newton) and delegates to
    the appropriate backend-specific implementation:

    - **PhysX**: Uses rest offset and contact offset directly via the PhysX tensor API
      (``root_view.set_rest_offsets`` / ``root_view.set_contact_offsets``).
    - **Newton**: Maps PhysX concepts to Newton's geometry properties. PhysX ``rest_offset``
      maps to Newton ``shape_margin``, and PhysX ``contact_offset`` is converted to Newton
      ``shape_gap`` via ``gap = contact_offset - margin``.

    The function samples random values from the given distribution parameters and applies them
    as absolute values to the collider properties. If the distribution parameters are not
    provided for a particular property, the function does not modify it.

    .. tip::
        This function uses CPU tensors (PhysX) or GPU tensors (Newton) to assign the collision
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

        # construct the backend-specific implementation via the converter
        self._impl = _resolve_backend_event("RandomizeRigidBodyColliderOffsets", self.asset)

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

    Automatically detects the active physics backend (PhysX or Newton) and applies
    the appropriate gravity randomization strategy:

    - **PhysX**: samples a single gravity vector and sets it scene-wide via the PhysX
      simulation view.  All environments share the same gravity.
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

        # construct the backend-specific implementation via the converter
        self._impl = _resolve_backend_event("RandomizePhysicsSceneGravity", cfg, env)

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

        # extract the used quantities (to enable type-hinting)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: RigidObject | Articulation = env.scene[self.asset_cfg.name]

        # construct the backend-specific implementation via the converter
        self._impl = _resolve_backend_event("RandomizeActuatorGains", cfg, env, self.asset, self.asset_cfg)

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
        self._impl(
            env,
            env_ids,
            asset_cfg,
            stiffness_distribution_params,
            damping_distribution_params,
            operation,
            distribution,
        )


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

        # extract the used quantities (to enable type-hinting)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: Articulation = env.scene[self.asset_cfg.name]

        # construct the backend-specific implementation via the converter
        self._impl = _resolve_backend_event("RandomizeJointParameters", cfg, env, self.asset, self.asset_cfg)

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
        self._impl(
            env,
            env_ids,
            asset_cfg,
            friction_distribution_params,
            armature_distribution_params,
            lower_limit_distribution_params,
            upper_limit_distribution_params,
            operation,
            distribution,
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

        # extract the used quantities (to enable type-hinting)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: RigidObject | Articulation = env.scene[self.asset_cfg.name]

        # construct the backend-specific implementation via the converter
        self._impl = _resolve_backend_event("RandomizeFixedTendonParameters", cfg, env, self.asset, self.asset_cfg)

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
        self._impl(
            env,
            env_ids,
            asset_cfg,
            stiffness_distribution_params,
            damping_distribution_params,
            limit_stiffness_distribution_params,
            lower_limit_distribution_params,
            upper_limit_distribution_params,
            rest_length_distribution_params,
            offset_distribution_params,
            operation,
            distribution,
        )


def randomize_prop_by_op(
    data: torch.Tensor,
    distribution_parameters: tuple[float | torch.Tensor, float | torch.Tensor],
    dim_0_ids: torch.Tensor | None,
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
    # resolve shape
    # -- dim 0
    if dim_0_ids is None:
        n_dim_0 = data.shape[0]
        dim_0_ids = slice(None)
    else:
        n_dim_0 = len(dim_0_ids)
        if not isinstance(dim_1_ids, slice):
            dim_0_ids = dim_0_ids[:, None]
    # -- dim 1
    if isinstance(dim_1_ids, slice):
        n_dim_1 = data.shape[1]
    else:
        n_dim_1 = len(dim_1_ids)

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


def validate_scale_range(
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
        validate_scale_range((0.5, 1.5), "mass_scale")
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
