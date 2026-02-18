# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
import numpy as np
import re

import warp as wp
from newton import (
    Axis,
    BroadPhaseMode,
    CollisionPipelineUnified,
    Contacts,
    Control,
    Model,
    ModelBuilder,
    State,
    eval_fk,
)
from newton.geometry import SDFHydroelasticConfig
from newton.sensors import SensorContact as NewtonContactSensor
from newton.sensors import populate_contacts
from newton.solvers import SolverBase, SolverFeatherstone, SolverMuJoCo, SolverNotifyFlags, SolverXPBD

from isaaclab.sim._impl.newton_manager_cfg import HydroelasticCfg, NewtonCfg
from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.utils.timer import Timer

logger = logging.getLogger(__name__)


def flipped_match(x: str, y: str) -> re.Match | None:
    """Flipped match function.

    This function is used to match the contact partners' body/shape names with the body/shape names in the simulation.

    Args:
        x: The body/shape name in the simulation.
        y: The body/shape name in the contact view.

    Returns:
        The match object if the body/shape name is found in the contact view, otherwise None.
    """
    return re.match(y, x)


class NewtonManager:
    _builder: ModelBuilder = None
    _model: Model = None
    _device: str = "cuda:0"
    _dt: float = 1.0 / 200.0
    _solver_dt: float = 1.0 / 200.0
    _num_substeps: int = 1
    _solver = None
    _state_0: State = None
    _state_1: State = None
    _state_temp: State = None
    _control: Control = None
    _on_init_callbacks: list = []
    _on_start_callbacks: list = []
    _contacts: Contacts = None
    _needs_collision_pipeline: bool = False
    _collision_pipeline = None
    _newton_contact_sensor: NewtonContactSensor = None  # TODO: allow several contact sensors
    _report_contacts: bool = False
    _graph = None
    _newton_stage_path = None
    _sim_time = 0.0
    _usdrt_stage = None
    _newton_index_attr = "newton:index"
    _clone_physics_only = False
    _cfg: NewtonCfg = NewtonCfg()
    _solver_type: str = "mujoco_warp"
    _gravity_vector: tuple[float, float, float] = (0.0, 0.0, -9.81)
    _up_axis: str = "Z"
    _num_envs: int = None
    _model_changes: set[int] = set()

    @classmethod
    def clear(cls):
        NewtonManager._builder = None
        NewtonManager._model = None
        NewtonManager._solver = None
        NewtonManager._state_0 = None
        NewtonManager._state_1 = None
        NewtonManager._state_temp = None
        NewtonManager._control = None
        NewtonManager._contacts = None
        NewtonManager._needs_collision_pipeline = False
        NewtonManager._collision_pipeline = None
        NewtonManager._newton_contact_sensor = None
        NewtonManager._report_contacts = False
        NewtonManager._graph = None
        NewtonManager._newton_stage_path = None
        NewtonManager._sim_time = 0.0
        NewtonManager._on_init_callbacks = []
        NewtonManager._on_start_callbacks = []
        NewtonManager._usdrt_stage = None
        # Only create new config if not during Python shutdown
        try:
            NewtonManager._cfg = NewtonCfg()
        except (ImportError, AttributeError, TypeError):
            NewtonManager._cfg = None
        NewtonManager._up_axis = "Z"
        NewtonManager._first_call = True
        NewtonManager._model_changes = set()

    @classmethod
    def set_builder(cls, builder):
        NewtonManager._builder = builder

    @classmethod
    def add_on_init_callback(cls, callback) -> None:
        NewtonManager._on_init_callbacks.append(callback)

    @classmethod
    def add_on_start_callback(cls, callback) -> None:
        NewtonManager._on_start_callbacks.append(callback)

    @classmethod
    def add_model_change(cls, change: SolverNotifyFlags) -> None:
        NewtonManager._model_changes.add(change)

    @classmethod
    def start_simulation(cls) -> None:
        """Starts the simulation.

        This function finalizes the model and initializes the simulation state.
        """

        print(f"[INFO] Builder: {NewtonManager._builder}")
        if NewtonManager._builder is None:
            NewtonManager.instantiate_builder_from_stage()
        print("[INFO] Running on init callbacks")
        for callback in NewtonManager._on_init_callbacks:
            callback()
        print(f"[INFO] Finalizing model on device: {NewtonManager._device}")
        NewtonManager._builder.up_axis = Axis.from_string(NewtonManager._up_axis)
        with Timer(name="newton_finalize_builder", msg="Finalize builder took:", enable=True, format="ms"):
            NewtonManager._model = NewtonManager._builder.finalize(device=NewtonManager._device)
            NewtonManager._model.set_gravity(NewtonManager._gravity_vector)
            NewtonManager._model.num_envs = NewtonManager._num_envs
        NewtonManager._state_0 = NewtonManager._model.state()
        NewtonManager._state_1 = NewtonManager._model.state()
        NewtonManager._state_temp = NewtonManager._model.state()
        NewtonManager._control = NewtonManager._model.control()
        NewtonManager.forward_kinematics()
        # Contacts are initialized here with empty defaults.
        # The actual collision pipeline is created later in initialize_solver()
        # once _needs_collision_pipeline has been determined from the solver type.
        NewtonManager._contacts = Contacts(0, 0)
        print("[INFO] Running on start callbacks")
        for callback in NewtonManager._on_start_callbacks:
            callback()
        if not NewtonManager._clone_physics_only:
            import usdrt

            NewtonManager._usdrt_stage = get_current_stage(fabric=True)
            for i, prim_path in enumerate(NewtonManager._model.body_key):
                prim = NewtonManager._usdrt_stage.GetPrimAtPath(prim_path)
                prim.CreateAttribute(NewtonManager._newton_index_attr, usdrt.Sdf.ValueTypeNames.UInt, True)
                prim.GetAttribute(NewtonManager._newton_index_attr).Set(i)
                xformable_prim = usdrt.Rt.Xformable(prim)
                if not xformable_prim.HasWorldXform():
                    xformable_prim.SetWorldXformFromUsd()

    @classmethod
    def instantiate_builder_from_stage(cls):
        from pxr import UsdGeom

        stage = get_current_stage()
        up_axis = UsdGeom.GetStageUpAxis(stage)
        builder = ModelBuilder(up_axis=up_axis)
        builder.add_usd(stage)

        # Newton's import_usd does not propagate SDF settings from USD
        # (physics:approximation="sdf") to the ShapeConfig. We post-process
        # the builder to apply SDF parameters from NewtonCfg to all mesh shapes.
        cls._apply_sdf_config(builder)

        # Bridge PhysX disable_gravity -> MuJoCo gravcomp.
        # Newton's USD importer reads ``mjc:gravcomp`` but IsaacLab sets
        # ``physxRigidBody:disableGravity``.  Translate the latter into the
        # MuJoCo gravity-compensation attribute so bodies configured with
        # ``disable_gravity=True`` are not affected by gravity in the MuJoCo solver.
        cls._apply_gravity_compensation(builder, stage)

        NewtonManager.set_builder(builder)

    @classmethod
    def _apply_sdf_config(cls, builder: ModelBuilder):
        """Ensure matching bodies have collision mesh shapes with SDF and hydroelastic settings.

        Newton's ``import_usd`` may fail to load collision shapes for assets that use
        ``PhysxSDFMeshCollisionAPI`` (e.g. factory gear assets).  This method:

        1. Identifies bodies whose key matches ``sdf_shape_patterns``.
        2. For bodies that have a visual mesh but **no** collision shape, creates a
           collision shape from the visual mesh via ``builder.add_shape_mesh()``.
        3. Sets ``sdf_max_resolution`` / ``sdf_narrow_band_range`` on all matching
           collision mesh shapes (both pre-existing and newly created).
        4. If ``hydroelastic_cfg`` is set, enables the ``HYDROELASTIC`` flag and
           ``k_hydro`` stiffness on matching shapes.
        """
        from newton import GeoType, ShapeFlags  # noqa: PLC0415

        cfg = NewtonManager._cfg
        has_sdf = cfg.sdf_max_resolution is not None or cfg.sdf_target_voxel_size is not None

        if not has_sdf:
            if cfg.hydroelastic_cfg is not None:
                logger.warning(
                    "Hydroelastic contacts require SDF to be enabled (sdf_max_resolution or sdf_target_voxel_size). "
                    "Hydroelastic will be disabled."
                )
            return

        hydro_cfg = cfg.hydroelastic_cfg

        # Compile body-name filter patterns for SDF
        patterns = None
        if cfg.sdf_shape_patterns is not None:
            patterns = [re.compile(p) for p in cfg.sdf_shape_patterns]

        # Compile separate hydroelastic patterns (falls back to SDF patterns if not specified)
        hydro_patterns = patterns
        if hydro_cfg is not None and hydro_cfg.shape_patterns is not None:
            hydro_patterns = [re.compile(p) for p in hydro_cfg.shape_patterns]

        # Build a map: body_index -> {visual_shape_indices, collision_shape_indices}
        body_info: dict[int, dict] = {}
        for i in range(builder.shape_count):
            body_idx = builder.shape_body[i]
            if body_idx < 0:
                continue
            if body_idx not in body_info:
                body_info[body_idx] = {"visual": [], "collision": []}
            flags = builder.shape_flags[i]
            has_collision = bool(flags & ShapeFlags.COLLIDE_SHAPES)
            if has_collision:
                body_info[body_idx]["collision"].append(i)
            else:
                body_info[body_idx]["visual"].append(i)

        # Build collision ShapeConfig with SDF (and optionally hydroelastic) parameters
        shape_cfg_kwargs = dict(
            density=0.0,  # Don't add mass -- body mass comes from USD import
            has_shape_collision=True,
            has_particle_collision=True,
            is_visible=True,
            sdf_max_resolution=cfg.sdf_max_resolution,
            sdf_narrow_band_range=cfg.sdf_narrow_band_range,
            sdf_target_voxel_size=cfg.sdf_target_voxel_size,
            contact_margin=cfg.sdf_contact_margin,
        )
        if hydro_cfg is not None:
            shape_cfg_kwargs["is_hydroelastic"] = True
            shape_cfg_kwargs["k_hydro"] = hydro_cfg.k_hydro
        sdf_shape_cfg = ModelBuilder.ShapeConfig(**shape_cfg_kwargs)

        num_added = 0
        num_patched = 0
        num_hydro = 0

        for body_idx, info in body_info.items():
            body_key = builder.body_key[body_idx]
            # Check if this body matches the SDF pattern
            if patterns is not None:
                if not any(p.search(body_key) for p in patterns):
                    continue

            # Determine if this body should also get hydroelastic
            body_gets_hydro = False
            if hydro_cfg is not None:
                if hydro_patterns is None:
                    body_gets_hydro = True
                else:
                    body_gets_hydro = any(p.search(body_key) for p in hydro_patterns)

            # Case 1: Body has collision shapes already -> patch SDF settings
            if info["collision"]:
                for si in info["collision"]:
                    if builder.shape_type[si] == GeoType.MESH:
                        if cfg.sdf_max_resolution is not None:
                            builder.shape_sdf_max_resolution[si] = cfg.sdf_max_resolution
                        if cfg.sdf_target_voxel_size is not None:
                            builder.shape_sdf_target_voxel_size[si] = cfg.sdf_target_voxel_size
                        builder.shape_sdf_narrow_band_range[si] = cfg.sdf_narrow_band_range
                        if cfg.sdf_contact_margin is not None:
                            builder.shape_contact_margin[si] = cfg.sdf_contact_margin
                        if body_gets_hydro:
                            builder.shape_flags[si] |= ShapeFlags.HYDROELASTIC
                            builder.shape_material_k_hydro[si] = hydro_cfg.k_hydro
                            num_hydro += 1
                        num_patched += 1
                continue

            # Case 2: Body has NO collision shapes -> create from visual mesh
            visual_mesh_idx = None
            for si in info["visual"]:
                if builder.shape_type[si] == GeoType.MESH and builder.shape_source[si] is not None:
                    visual_mesh_idx = si
                    break

            if visual_mesh_idx is None:
                logger.warning(f"SDF: body '{body_key}' matched but has no visual mesh to create collision from.")
                continue

            # Reuse the visual mesh geometry and transform
            mesh = builder.shape_source[visual_mesh_idx]
            xform = builder.shape_transform[visual_mesh_idx]
            scale = builder.shape_scale[visual_mesh_idx]

            # For new shapes: hydroelastic is already in sdf_shape_cfg if hydro_cfg is set.
            # If this body shouldn't get hydro but hydro_cfg is set globally, use a non-hydro cfg.
            if hydro_cfg is not None and not body_gets_hydro:
                non_hydro_cfg = ModelBuilder.ShapeConfig(
                    density=0.0,
                    has_shape_collision=True,
                    has_particle_collision=True,
                    is_visible=True,
                    sdf_max_resolution=cfg.sdf_max_resolution,
                    sdf_narrow_band_range=cfg.sdf_narrow_band_range,
                    sdf_target_voxel_size=cfg.sdf_target_voxel_size,
                    contact_margin=cfg.sdf_contact_margin,
                )
                add_cfg = non_hydro_cfg
            else:
                add_cfg = sdf_shape_cfg
                if body_gets_hydro:
                    num_hydro += 1

            new_shape_id = builder.add_shape_mesh(
                body=body_idx,
                xform=xform,
                mesh=mesh,
                scale=scale,
                cfg=add_cfg,
                key=f"{body_key}/sdf_collision",
            )
            num_added += 1
            logger.info(f"SDF: added collision shape {new_shape_id} for body '{body_key}'")

        hydro_msg = f", {num_hydro} hydroelastic shape(s)" if hydro_cfg is not None else ""
        logger.info(
            f"SDF config: {num_added} collision shape(s) added, {num_patched} existing shape(s) patched{hydro_msg}. "
            f"(max_resolution={cfg.sdf_max_resolution}, narrow_band={cfg.sdf_narrow_band_range})"
        )

    @classmethod
    def _apply_gravity_compensation(cls, builder: ModelBuilder, stage):
        """Translate ``physxRigidBody:disableGravity`` into MuJoCo ``gravcomp``.

        Newton's MuJoCo solver uses a per-body ``gravcomp`` attribute (0.0 = full
        gravity, 1.0 = fully compensated) whereas PhysX uses the boolean
        ``physxRigidBody:disableGravity``.  This method bridges the two by scanning
        the USD stage for prims that have ``disableGravity=True`` and setting
        ``gravcomp=1.0`` on the corresponding bodies in the builder.
        """

        # Register the mujoco:gravcomp custom attribute if not yet present.
        if not builder.has_custom_attribute("mujoco:gravcomp"):
            builder.add_custom_attribute(
                ModelBuilder.CustomAttribute(
                    name="gravcomp",
                    frequency=Model.AttributeFrequency.BODY,
                    assignment=Model.AttributeAssignment.MODEL,
                    dtype=wp.float32,
                    default=0.0,
                    namespace="mujoco",
                )
            )

        gravcomp_attr = builder.custom_attributes["mujoco:gravcomp"]

        # Build body_key -> index lookup
        body_key_to_idx = {key: idx for idx, key in enumerate(builder.body_key)}

        num_set = 0
        for body_key, body_idx in body_key_to_idx.items():
            prim = stage.GetPrimAtPath(body_key)
            if not prim.IsValid():
                continue
            attr = prim.GetAttribute("physxRigidBody:disableGravity")
            if attr.IsValid() and attr.Get():
                gravcomp_attr.values[body_idx] = 1.0
                num_set += 1

        if num_set > 0:
            logger.info(f"Gravity compensation: set gravcomp=1.0 on {num_set} bodies with disableGravity=True")

    @classmethod
    def _create_collision_pipeline(cls, model: Model) -> CollisionPipelineUnified:
        """Create a ``CollisionPipelineUnified`` with optional hydroelastic support.

        When ``hydroelastic_cfg`` is set on the active :class:`NewtonCfg`, the pipeline
        is created with an :class:`SDFHydroelasticConfig` so that shapes with the
        ``HYDROELASTIC`` flag use distributed surface contacts.  Otherwise the pipeline
        is created with default settings (point contacts only).

        Args:
            model: The finalized Newton model.

        Returns:
            A ``CollisionPipelineUnified`` instance.
        """
        hydro_cfg = NewtonManager._cfg.hydroelastic_cfg
        sdf_hydro_config = None

        if hydro_cfg is not None:
            # Build the Newton-level SDFHydroelasticConfig from our IsaacLab cfg
            sdf_hydro_config = SDFHydroelasticConfig(
                reduce_contacts=hydro_cfg.reduce_contacts,
                output_contact_surface=hydro_cfg.output_contact_surface,
                sticky_contacts=hydro_cfg.sticky_contacts,
                normal_matching=hydro_cfg.normal_matching,
                moment_matching=hydro_cfg.moment_matching,
                margin_contact_area=hydro_cfg.margin_contact_area,
                betas=hydro_cfg.betas,
                buffer_mult_broad=hydro_cfg.buffer_mult_broad,
                buffer_mult_iso=hydro_cfg.buffer_mult_iso,
                buffer_mult_contact=hydro_cfg.buffer_mult_contact,
                grid_size=hydro_cfg.grid_size,
            )
            logger.info(
                f"Hydroelastic contacts enabled (k_hydro={hydro_cfg.k_hydro}, "
                f"reduce_contacts={hydro_cfg.reduce_contacts})"
            )

        return CollisionPipelineUnified.from_model(
            model,
            broad_phase_mode=BroadPhaseMode.EXPLICIT,
            sdf_hydroelastic_config=sdf_hydro_config,
        )

    @classmethod
    def set_solver_settings(cls, newton_params: dict):
        # Reconstruct nested dataclass configs that were flattened to dicts by to_dict()
        hydro = newton_params.get("hydroelastic_cfg")
        if isinstance(hydro, dict):
            newton_params["hydroelastic_cfg"] = HydroelasticCfg(**hydro)
        NewtonManager._cfg = NewtonCfg(**newton_params)

    @classmethod
    def initialize_solver(cls):
        """Initializes the solver.

        This function initializes the solver based on the specified solver type. Currently, only XPBD and MuJoCoWarp
        are supported. The graphing of the simulation is performed in this function if the simulation is ran using
        a CUDA enabled device.

        .. warning::
            When using a CUDA enabled device, the simulation will be graphed. This means that this function steps the
            simulation once to capture the graph. Hence, this function should only be called after everything else in
            the simulation is initialized.
        """
        with Timer(name="newton_initialize_solver", msg="Initialize solver took:", enable=True, format="ms"):
            NewtonManager._num_substeps = NewtonManager._cfg.num_substeps
            NewtonManager._solver_dt = NewtonManager._dt / NewtonManager._num_substeps
            NewtonManager._solver = NewtonManager._get_solver(NewtonManager._model, NewtonManager._cfg.solver_cfg)
            if isinstance(NewtonManager._solver, SolverMuJoCo):
                NewtonManager._needs_collision_pipeline = not NewtonManager._cfg.solver_cfg.get(
                    "use_mujoco_contacts", False
                )
            else:
                NewtonManager._needs_collision_pipeline = True

        # Create the collision pipeline now that _needs_collision_pipeline is determined
        if NewtonManager._needs_collision_pipeline:
            with Timer(
                name="newton_create_collision_pipeline", msg="Create collision pipeline took:", enable=True, format="ms"
            ):
                NewtonManager._collision_pipeline = cls._create_collision_pipeline(NewtonManager._model)
                NewtonManager._contacts = NewtonManager._model.collide(
                    NewtonManager._state_0, collision_pipeline=NewtonManager._collision_pipeline
                )

        # Capture the graph if CUDA is enabled
        with Timer(name="newton_cuda_graph", msg="CUDA graph took:", enable=True, format="ms"):
            if NewtonManager._cfg.use_cuda_graph and NewtonManager._device.startswith("cuda"):
                with wp.ScopedCapture() as capture:
                    NewtonManager.simulate()
                NewtonManager._graph = capture.graph
            elif NewtonManager._cfg.use_cuda_graph and not NewtonManager._device.startswith("cuda"):
                logger.warning("CUDA graphs requested but device is CPU. Disabling CUDA graphs.")
                NewtonManager._cfg.use_cuda_graph = False

    @classmethod
    def simulate(cls) -> None:
        """Simulates the simulation.

        Performs one simulation step with the specified number of substeps. Depending on the solver type, this function
        may need to explicitly compute the collisions. This function also aggregates the contacts and evaluates the
        contact sensors. Finally, it performs the state swapping for Newton.
        """
        state_0_dict = NewtonManager._state_0.__dict__
        state_1_dict = NewtonManager._state_1.__dict__
        state_temp_dict = NewtonManager._state_temp.__dict__
        contacts = None

        # MJWarp computes its own collisions.
        if NewtonManager._needs_collision_pipeline:
            contacts = NewtonManager._model.collide(
                NewtonManager._state_0, collision_pipeline=NewtonManager._collision_pipeline
            )

        if NewtonManager._num_substeps % 2 == 0:
            for i in range(NewtonManager._num_substeps):
                NewtonManager._solver.step(
                    NewtonManager._state_0,
                    NewtonManager._state_1,
                    NewtonManager._control,
                    contacts,
                    NewtonManager._solver_dt,
                )
                NewtonManager._state_0, NewtonManager._state_1 = NewtonManager._state_1, NewtonManager._state_0
                NewtonManager._state_0.clear_forces()
        else:
            for i in range(NewtonManager._num_substeps):
                NewtonManager._solver.step(
                    NewtonManager._state_0,
                    NewtonManager._state_1,
                    NewtonManager._control,
                    contacts,
                    NewtonManager._solver_dt,
                )

                # FIXME: Ask Lukasz help to deal with non-even number of substeps. This should not be needed.
                if i < NewtonManager._num_substeps - 1 or not NewtonManager._cfg.use_cuda_graph:
                    # we can just swap the state references
                    NewtonManager._state_0, NewtonManager._state_1 = NewtonManager._state_1, NewtonManager._state_0
                elif NewtonManager._cfg.use_cuda_graph:
                    # swap states by actually copying the state arrays to make sure the graph capture works
                    for key, value in state_0_dict.items():
                        if isinstance(value, wp.array):
                            if key not in state_temp_dict:
                                state_temp_dict[key] = wp.empty_like(value)
                            state_temp_dict[key].assign(value)
                            state_0_dict[key].assign(state_1_dict[key])
                            state_1_dict[key].assign(state_temp_dict[key])
                NewtonManager._state_0.clear_forces()

        if NewtonManager._report_contacts:
            populate_contacts(NewtonManager._contacts, NewtonManager._solver)
            NewtonManager._newton_contact_sensor.eval(NewtonManager._contacts)

    @classmethod
    def set_device(cls, device: str) -> None:
        """Sets the device to use for the Newton simulation.

        Args:
            device (str): The device to use for the Newton simulation.
        """
        NewtonManager._device = device

    @classmethod
    def step(cls) -> None:
        """Steps the simulation.

        This function steps the simulation by the specified time step in the simulation configuration.
        """
        if NewtonManager._model_changes:
            for change in NewtonManager._model_changes:
                NewtonManager._solver.notify_model_changed(change)
            NewtonManager._model_changes = set()

        if NewtonManager._cfg.use_cuda_graph:
            wp.capture_launch(NewtonManager._graph)
        else:
            NewtonManager.simulate()

        if NewtonManager._cfg.debug_mode:
            convergence_data = NewtonManager.get_solver_convergence_steps()
            # print(f"solver niter: {convergence_data}")
            if convergence_data["max"] == NewtonManager._solver.mjw_model.opt.iterations:
                print("solver didn't converge!", convergence_data["max"])

        NewtonManager._sim_time += NewtonManager._solver_dt * NewtonManager._num_substeps

    @classmethod
    def get_solver_convergence_steps(cls) -> dict[str, float | int]:
        niter = NewtonManager._solver.mjw_data.solver_niter.numpy()
        max_niter = np.max(niter)
        mean_niter = np.mean(niter)
        min_niter = np.min(niter)
        std_niter = np.std(niter)
        return {"max": max_niter, "mean": mean_niter, "min": min_niter, "std": std_niter}

    @classmethod
    def set_simulation_dt(cls, dt: float) -> None:
        """Sets the simulation time step and the number of substeps.

        Args:
            dt (float): The simulation time step.
        """
        NewtonManager._dt = dt

    @classmethod
    def get_model(cls):
        return NewtonManager._model

    @classmethod
    def get_state_0(cls):
        return NewtonManager._state_0

    @classmethod
    def get_state_1(cls):
        return NewtonManager._state_1

    @classmethod
    def get_control(cls):
        return NewtonManager._control

    @classmethod
    def get_dt(cls):
        return NewtonManager._dt

    @classmethod
    def get_solver_dt(cls):
        return NewtonManager._solver_dt

    @classmethod
    def forward_kinematics(cls, mask: wp.array | None = None) -> None:
        """Evaluates the forward kinematics for the selected articulations.

        This function evaluates the forward kinematics for the selected articulations.
        """
        eval_fk(
            NewtonManager._model,
            NewtonManager._state_0.joint_q,
            NewtonManager._state_0.joint_qd,
            NewtonManager._state_0,
            None,
        )

    @classmethod
    def _get_solver(cls, model: Model, solver_cfg: dict) -> SolverBase:
        NewtonManager._solver_type = solver_cfg.pop("solver_type")

        if NewtonManager._solver_type == "mujoco_warp":
            return SolverMuJoCo(model, **solver_cfg)
        elif NewtonManager._solver_type == "xpbd":
            return SolverXPBD(model, **solver_cfg)
        elif NewtonManager._solver_type == "featherstone":
            return SolverFeatherstone(model, **solver_cfg)
        else:
            raise ValueError(f"Invalid solver type: {NewtonManager._solver_type}")

    @classmethod
    def add_contact_sensor(
        cls,
        body_names_expr: str | list[str] | None = None,
        shape_names_expr: str | list[str] | None = None,
        contact_partners_body_expr: str | list[str] | None = None,
        contact_partners_shape_expr: str | list[str] | None = None,
        prune_noncolliding: bool = False,
        verbose: bool = False,
    ):
        """Adds a contact view.

        Adds a contact view to the simulation allowing to report contacts between the specified bodies/shapes and the
        contact partners. As of now, only one body/shape name expression can be provided. Similarly, only one contact
        partner body/shape expression can be provided. If no contact partner expression is provided, the contact view
        will report contacts with all bodies/shapes.

        Note that we make an explicit difference between a body and a shape. A body is a rigid body, while a shape
        is a collision shape. A body can have multiple shapes. The shape option allows a more fine-grained control
        over the contact reporting.

        Args:
            body_names_expr (str | None): The expression for the body names.
            shape_names_expr (str | None): The expression for the shape names.
            contact_partners_body_expr (str | None): The expression for the contact partners' body names.
            contact_partners_shape_expr (str | None): The expression for the contact partners' shape names.
            prune_noncolliding (bool): Make the force matrix sparse using the collision pairs in the model.
            verbose (bool): Whether to print verbose information.
        """
        if body_names_expr is None and shape_names_expr is None:
            raise ValueError("At least one of body_names_expr or shape_names_expr must be provided")
        if body_names_expr is not None and shape_names_expr is not None:
            raise ValueError("Only one of body_names_expr or shape_names_expr must be provided")
        if contact_partners_body_expr is not None and contact_partners_shape_expr is not None:
            raise ValueError("Only one of contact_partners_body_expr or contact_partners_shape_expr must be provided")
        if contact_partners_body_expr is None and contact_partners_shape_expr is None:
            print(f"[INFO] Adding contact view for {body_names_expr}. It will report contacts with all bodies/shapes.")
        else:
            if body_names_expr is not None:
                if contact_partners_body_expr is not None:
                    print(f"[INFO] Adding contact view for {body_names_expr} with filter {contact_partners_body_expr}.")
                else:
                    print(f"[INFO] Adding contact view for {body_names_expr} with filter {shape_names_expr}.")
            else:
                if contact_partners_body_expr is not None:
                    print(
                        f"[INFO] Adding contact view for {shape_names_expr} with filter {contact_partners_body_expr}."
                    )
                else:
                    print(
                        f"[INFO] Adding contact view for {shape_names_expr} with filter {contact_partners_shape_expr}."
                    )

        NewtonManager._newton_contact_sensor = NewtonContactSensor(
            NewtonManager._model,
            sensing_obj_bodies=body_names_expr,
            sensing_obj_shapes=shape_names_expr,
            counterpart_bodies=contact_partners_body_expr,
            counterpart_shapes=contact_partners_shape_expr,
            match_fn=flipped_match,
            include_total=True,
            prune_noncolliding=prune_noncolliding,
            verbose=verbose,
        )
        NewtonManager._report_contacts = True
