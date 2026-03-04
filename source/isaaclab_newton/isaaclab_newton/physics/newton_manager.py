# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton physics manager for Isaac Lab."""

from __future__ import annotations

import logging
import numpy as np
import re
from typing import TYPE_CHECKING

import warp as wp
from newton import Axis, BroadPhaseMode, CollisionPipeline, Contacts, Control, Model, ModelBuilder, State, eval_fk
from newton.geometry import HydroelasticSDF
from newton.sensors import SensorContact as NewtonContactSensor
from newton.solvers import SolverBase, SolverFeatherstone, SolverMuJoCo, SolverNotifyFlags, SolverXPBD

from isaaclab.physics import PhysicsEvent, PhysicsManager
from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.utils.timer import Timer

if TYPE_CHECKING:
    from isaaclab.sim.simulation_context import SimulationContext

logger = logging.getLogger(__name__)


def flipped_match(x: str, y: str) -> bool:
    """Flipped match function for contact partner matching.

    Args:
        x: The body/shape name in the simulation.
        y: The body/shape name in the contact view.

    Returns:
        True if the body/shape name matches the contact view pattern.
    """
    return re.match(y, x) is not None


class NewtonManager(PhysicsManager):
    """Newton physics manager for Isaac Lab.

    This is a class-level (singleton-like) manager for the Newton simulation.
    It handles solver configuration, physics stepping, and reset.

    Lifecycle: initialize() -> reset() -> step() (repeated) -> close()
    """

    _solver_dt: float = 1.0 / 200.0
    _num_substeps: int = 1
    _num_envs: int | None = None

    # Newton model and state
    _builder: ModelBuilder = None
    _model: Model = None
    _solver: SolverBase = None
    _solver_type: str = "mujoco_warp"
    _use_single_state: bool | None = None
    """Use only one state for both input and output for solver stepping. Requires solver support."""
    _state_0: State = None
    _state_1: State = None
    _control: Control = None

    # Physics settings
    _gravity_vector: tuple[float, float, float] = (0.0, 0.0, -9.81)
    _up_axis: str = "Z"

    # Collision and contacts
    _contacts: Contacts | None = None
    _needs_collision_pipeline: bool = False
    _collision_pipeline = None
    _newton_contact_sensors: dict = {}  # Maps sensor_key to NewtonContactSensor
    _report_contacts: bool = False

    # CUDA graphing
    _graph = None

    # USD/Fabric sync
    _newton_stage_path = None
    _usdrt_stage = None
    _newton_index_attr = "newton:index"
    _clone_physics_only = False

    # Model changes (callbacks use unified system from PhysicsManager)
    _model_changes: set[int] = set()

    # Views list for assets to register their views
    _views: list = []

    @classmethod
    def initialize(cls, sim_context: SimulationContext) -> None:
        """Initialize the manager with simulation context.

        Args:
            sim_context: Parent simulation context.
        """
        super().initialize(sim_context)

        # Newton-specific setup: get gravity from SimulationCfg (not physics manager cfg)
        sim = PhysicsManager._sim
        if sim is not None:
            cls._gravity_vector = sim.cfg.gravity  # type: ignore[union-attr]

            # USD fabric sync only needed for OV rendering
            viz_str = sim.get_setting("/isaaclab/visualizer") or ""
            requested = [v.strip() for v in viz_str.split(",") if v.strip()]
            cls._clone_physics_only = "omniverse" not in requested

    @classmethod
    def reset(cls, soft: bool = False) -> None:
        """Reset physics simulation.

        Args:
            soft: If True, skip full reinitialization.
        """
        if not soft:
            cls.start_simulation()
            cls.initialize_solver()

    @classmethod
    def forward(cls) -> None:
        """Update articulation kinematics without stepping physics."""
        eval_fk(cls._model, cls._state_0.joint_q, cls._state_0.joint_qd, cls._state_0, None)

    @classmethod
    def step(cls) -> None:
        """Step the physics simulation."""
        sim = PhysicsManager._sim
        if sim is None or not sim.is_playing():
            return

        # Notify solver of model changes
        if cls._model_changes:
            for change in cls._model_changes:
                cls._solver.notify_model_changed(change)
            cls._model_changes = set()

        # Step simulation (graphed or not)
        cfg = PhysicsManager._cfg
        if cfg is not None and cfg.use_cuda_graph:  # type: ignore[union-attr]
            wp.capture_launch(cls._graph)  # type: ignore[arg-type]
        else:
            cls._simulate()

        # Debug convergence info
        if cfg is not None and cfg.debug_mode:  # type: ignore[union-attr]
            convergence_data = cls.get_solver_convergence_steps()
            if convergence_data["max"] == cls._solver.mjw_model.opt.iterations:
                logger.warning(f"Solver didn't converge! max_iter={convergence_data['max']}")

        PhysicsManager._sim_time += cls._solver_dt * cls._num_substeps

    @classmethod
    def close(cls) -> None:
        """Clean up Newton physics resources."""
        cls.clear()
        super().close()

    @classmethod
    def get_physics_sim_view(cls) -> list:
        """Get the list of registered views.

        Assets can append their views to this list, and sensors can access them.
        Returns a list that callers can append to.

        Returns:
            List of registered views (e.g., NewtonArticulationView instances).
        """
        return cls._views

    @classmethod
    def is_fabric_enabled(cls) -> bool:
        """Check if fabric interface is enabled (not applicable for Newton)."""
        return False

    @classmethod
    def clear(cls):
        """Clear all Newton-specific state (callbacks cleared by super().close())."""
        cls._builder = None
        cls._model = None
        cls._solver = None
        cls._use_single_state = None
        cls._state_0 = None
        cls._state_1 = None
        cls._control = None
        cls._contacts = None
        cls._needs_collision_pipeline = False
        cls._collision_pipeline = None
        cls._newton_contact_sensors = {}
        cls._report_contacts = False
        cls._graph = None
        cls._newton_stage_path = None
        cls._usdrt_stage = None
        cls._up_axis = "Z"
        cls._model_changes = set()
        cls._views = []

    @classmethod
    def set_builder(cls, builder: ModelBuilder) -> None:
        """Set the Newton model builder."""
        cls._builder = builder

    @classmethod
    def add_model_change(cls, change: SolverNotifyFlags) -> None:
        """Register a model change to notify the solver."""
        cls._model_changes.add(change)

    @classmethod
    def start_simulation(cls) -> None:
        """Start simulation by finalizing model and initializing state.

        This function finalizes the model and initializes the simulation state.
        Note: Collision pipeline is initialized later in initialize_solver() after
        we determine whether the solver needs external collision detection.
        """
        logger.debug(f"Builder: {cls._builder}")

        # Create builder from USD stage if not provided
        if cls._builder is None:
            cls.instantiate_builder_from_stage()
        else:
            # Builder was set externally (e.g. by newton_replicate) — still apply SDF and gravity comp
            from pxr import UsdGeom
            stage = get_current_stage()
            cls._apply_sdf_config(cls._builder)
            cls._apply_gravity_compensation(cls._builder, stage)

        logger.info("Dispatching MODEL_INIT callbacks")
        cls.dispatch_event(PhysicsEvent.MODEL_INIT)

        device = PhysicsManager._device
        logger.info(f"Finalizing model on device: {device}")
        cls._builder.up_axis = Axis.from_string(cls._up_axis)
        # Set smaller contact margin for manipulation examples (default 10cm is too large)
        cls._builder.default_shape_cfg.contact_margin = 0.01
        with Timer(name="newton_finalize_builder", msg="Finalize builder took:", enable=True, format="ms"):
            cls._model = cls._builder.finalize(device=device)
            cls._model.set_gravity(cls._gravity_vector)
            cls._model.num_envs = cls._num_envs

        cls._state_0 = cls._model.state()
        cls._state_1 = cls._model.state()
        cls._control = cls._model.control()
        eval_fk(cls._model, cls._state_0.joint_q, cls._state_0.joint_qd, cls._state_0, None)

        logger.info("Dispatching PHYSICS_READY callbacks")
        cls.dispatch_event(PhysicsEvent.PHYSICS_READY)

        # Setup USD/Fabric sync for Omniverse rendering
        if not cls._clone_physics_only:
            import usdrt

            cls._usdrt_stage = get_current_stage(fabric=True)
            for i, prim_path in enumerate(cls._model.body_key):
                prim = cls._usdrt_stage.GetPrimAtPath(prim_path)
                prim.CreateAttribute(cls._newton_index_attr, usdrt.Sdf.ValueTypeNames.UInt, True)
                prim.GetAttribute(cls._newton_index_attr).Set(i)
                xformable_prim = usdrt.Rt.Xformable(prim)
                if not xformable_prim.HasWorldXform():
                    xformable_prim.SetWorldXformFromUsd()

    @classmethod
    def instantiate_builder_from_stage(cls):
        """Create builder from USD stage."""
        from pxr import UsdGeom

        stage = get_current_stage()
        up_axis = UsdGeom.GetStageUpAxis(stage)
        builder = ModelBuilder(up_axis=up_axis)
        builder.add_usd(stage)
        cls._apply_sdf_config(builder)
        cls._apply_gravity_compensation(builder, stage)
        cls.set_builder(builder)

    @classmethod
    def _apply_sdf_config(cls, builder: ModelBuilder):
        """Ensure matching bodies have collision mesh shapes with SDF and hydroelastic settings."""
        from newton import GeoType, ShapeFlags

        cfg = PhysicsManager._cfg
        has_sdf = getattr(cfg, "sdf_max_resolution", None) is not None or getattr(cfg, "sdf_target_voxel_size", None) is not None

        if not has_sdf:
            if getattr(cfg, "hydroelastic_cfg", None) is not None:
                logger.warning(
                    "Hydroelastic contacts require SDF to be enabled (sdf_max_resolution or sdf_target_voxel_size). "
                    "Hydroelastic will be disabled."
                )
            return

        hydro_cfg = cfg.hydroelastic_cfg

        patterns = None
        if cfg.sdf_shape_patterns is not None:
            patterns = [re.compile(p) for p in cfg.sdf_shape_patterns]

        hydro_patterns = patterns
        if hydro_cfg is not None and hydro_cfg.shape_patterns is not None:
            hydro_patterns = [re.compile(p) for p in hydro_cfg.shape_patterns]

        # Per-pattern resolution overrides
        res_overrides = None
        if getattr(cfg, "sdf_pattern_resolutions", None) is not None:
            res_overrides = [(re.compile(p), r) for p, r in cfg.sdf_pattern_resolutions.items()]

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

        # ShapeConfig for new collision shapes — SDF must be built on the mesh directly
        # via mesh.build_sdf() before adding the shape.
        shape_cfg_kwargs = dict(
            density=0.0,
            has_shape_collision=True,
            has_particle_collision=True,
            is_visible=True,
            contact_margin=cfg.sdf_contact_margin,
        )
        if hydro_cfg is not None:
            shape_cfg_kwargs["is_hydroelastic"] = True
            shape_cfg_kwargs["kh"] = hydro_cfg.k_hydro
        sdf_shape_cfg = ModelBuilder.ShapeConfig(**shape_cfg_kwargs)

        num_added = 0
        num_patched = 0
        num_hydro = 0

        for body_idx, info in body_info.items():
            body_key = builder.body_key[body_idx]

            # Collect all keys for this body (body key + shape keys) for pattern matching
            all_shape_keys = []
            for si in info["collision"] + info["visual"]:
                sk = builder.shape_key[si]
                if sk:
                    all_shape_keys.append(sk)

            if patterns is not None:
                # Match if body key OR any shape key under this body matches
                body_matches = any(p.search(body_key) for p in patterns)
                shape_matches = any(p.search(sk) for p in patterns for sk in all_shape_keys)
                if not body_matches and not shape_matches:
                    continue

            body_gets_hydro = False
            if hydro_cfg is not None:
                if hydro_patterns is None:
                    body_gets_hydro = True
                else:
                    body_gets_hydro = any(
                        p.search(body_key) or any(p.search(sk) for sk in all_shape_keys)
                        for p in hydro_patterns
                    )

            if info["collision"]:
                for si in info["collision"]:
                    if builder.shape_type[si] == GeoType.MESH:
                        # Skip shapes that don't match any pattern (shape-level filtering)
                        if patterns is not None:
                            shape_key = builder.shape_key[si] or ""
                            shape_match = any(p.search(body_key) or p.search(shape_key) for p in patterns)
                            if not shape_match:
                                continue
                        # Build SDF on the mesh (new Newton API requires mesh.sdf)
                        mesh = builder.shape_source[si]
                        if mesh is not None:
                            if mesh.sdf is not None:
                                mesh.clear_sdf()
                            # Resolve per-pattern resolution override (check shape key then body key)
                            shape_resolution = cfg.sdf_max_resolution
                            if res_overrides is not None:
                                shape_key = builder.shape_key[si] or body_key
                                for pat, res in res_overrides:
                                    if pat.search(shape_key) or pat.search(body_key):
                                        shape_resolution = res
                                        break
                            sdf_kwargs = dict(narrow_band_range=cfg.sdf_narrow_band_range)
                            if shape_resolution is not None:
                                sdf_kwargs["max_resolution"] = shape_resolution
                            if cfg.sdf_target_voxel_size is not None:
                                sdf_kwargs["target_voxel_size"] = cfg.sdf_target_voxel_size
                            mesh.build_sdf(**sdf_kwargs)
                        if cfg.sdf_contact_margin is not None:
                            builder.shape_contact_margin[si] = cfg.sdf_contact_margin
                        if body_gets_hydro:
                            builder.shape_flags[si] |= ShapeFlags.HYDROELASTIC
                            builder.shape_material_kh[si] = hydro_cfg.k_hydro
                            num_hydro += 1
                        num_patched += 1
                continue

            visual_mesh_idx = None
            for si in info["visual"]:
                if builder.shape_type[si] == GeoType.MESH and builder.shape_source[si] is not None:
                    visual_mesh_idx = si
                    break

            if visual_mesh_idx is None:
                logger.warning(f"SDF: body '{body_key}' matched but has no visual mesh to create collision from.")
                continue

            mesh = builder.shape_source[visual_mesh_idx]
            xform = builder.shape_transform[visual_mesh_idx]
            scale = builder.shape_scale[visual_mesh_idx]

            if hydro_cfg is not None and not body_gets_hydro:
                non_hydro_cfg = ModelBuilder.ShapeConfig(
                    density=0.0,
                    has_shape_collision=True,
                    has_particle_collision=True,
                    is_visible=True,
                    contact_margin=cfg.sdf_contact_margin,
                )
                add_cfg = non_hydro_cfg
            else:
                add_cfg = sdf_shape_cfg
                if body_gets_hydro:
                    num_hydro += 1

            # Build SDF on the mesh before adding the shape
            if mesh.sdf is not None:
                mesh.clear_sdf()
            # Resolve per-pattern resolution override (check shape key then body key)
            shape_resolution = cfg.sdf_max_resolution
            if res_overrides is not None:
                shape_key = builder.shape_key[visual_mesh_idx] or body_key
                for pat, res in res_overrides:
                    if pat.search(shape_key) or pat.search(body_key):
                        shape_resolution = res
                        break
            sdf_kwargs = dict(narrow_band_range=cfg.sdf_narrow_band_range)
            if shape_resolution is not None:
                sdf_kwargs["max_resolution"] = shape_resolution
            if cfg.sdf_target_voxel_size is not None:
                sdf_kwargs["target_voxel_size"] = cfg.sdf_target_voxel_size
            mesh.build_sdf(**sdf_kwargs)

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
        """Translate ``physxRigidBody:disableGravity`` into MuJoCo ``gravcomp``."""
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
    def _create_collision_pipeline(cls, model: Model):
        """Create a collision pipeline with optional hydroelastic support.

        When ``hydroelastic_cfg`` is set on the active :class:`NewtonCfg`, the pipeline
        is created with a :class:`HydroelasticSDF.Config` so that shapes with the
        ``HYDROELASTIC`` flag use distributed surface contacts. Otherwise the pipeline
        is created with default settings (point contacts only).
        """
        cfg = PhysicsManager._cfg
        hydro_cfg = getattr(cfg, "hydroelastic_cfg", None)

        if hydro_cfg is not None:
            sdf_hydro_config = HydroelasticSDF.Config(
                reduce_contacts=hydro_cfg.reduce_contacts,
                output_contact_surface=hydro_cfg.output_contact_surface,
                normal_matching=hydro_cfg.normal_matching,
                moment_matching=hydro_cfg.moment_matching,
                margin_contact_area=hydro_cfg.margin_contact_area,
                buffer_mult_broad=hydro_cfg.buffer_mult_broad,
                buffer_mult_iso=hydro_cfg.buffer_mult_iso,
                buffer_mult_contact=hydro_cfg.buffer_mult_contact,
                grid_size=hydro_cfg.grid_size,
            )
            logger.info(
                f"Hydroelastic contacts enabled (k_hydro={hydro_cfg.k_hydro}, "
                f"reduce_contacts={hydro_cfg.reduce_contacts})"
            )
            pipeline = CollisionPipeline(
                model,
                broad_phase="explicit",
                sdf_hydroelastic_config=sdf_hydro_config,
            )
            return pipeline

        return CollisionPipeline(
            model,
            broad_phase="explicit",
        )

    @classmethod
    def _initialize_contacts(cls) -> None:
        """Unified method to initialize contacts and collision pipeline.

        This method handles both Newton collision pipeline and MuJoCo contact modes.
        It ensures contacts are properly initialized with force attributes if sensors are registered.
        """
        if cls._needs_collision_pipeline:
            # Newton collision pipeline: create pipeline and generate contacts
            if cls._collision_pipeline is None:
                with Timer(
                    name="newton_create_collision_pipeline",
                    msg="Create collision pipeline took:",
                    enable=True,
                    format="ms",
                ):
                    cls._collision_pipeline = cls._create_collision_pipeline(cls._model)
            if cls._contacts is None:
                cls._contacts = cls._collision_pipeline.contacts()

        elif cls._solver is not None and isinstance(cls._solver, SolverMuJoCo):
            # MuJoCo contacts mode: create properly sized Contacts object
            # The solver's update_contacts() will populate this from MuJoCo data
            rigid_contact_max = cls._solver.get_max_contact_count()
            cls._contacts = Contacts(
                rigid_contact_max=rigid_contact_max,
                soft_contact_max=0,
                device=PhysicsManager._device,
                requested_attributes=cls._model.get_requested_contact_attributes(),
            )

    @classmethod
    def initialize_solver(cls) -> None:
        """Initialize the solver and collision pipeline.

        This function initializes the solver based on the specified solver type. Currently, only XPBD and MuJoCoWarp
        are supported. If the solver requires external collision detection (i.e., not using MuJoCo's internal
        contacts), a unified collision pipeline is created.

        The graphing of the simulation is performed in this function if the simulation is ran using
        a CUDA enabled device.

        .. warning::
            When using a CUDA enabled device, the simulation will be graphed. This means that this function steps the
            simulation once to capture the graph. Hence, this function should only be called after everything else in
            the simulation is initialized.
        """
        cfg = PhysicsManager._cfg
        if cfg is None:
            return

        with Timer(name="newton_initialize_solver", msg="Initialize solver took:", enable=True, format="ms"):
            cls._num_substeps = cfg.num_substeps  # type: ignore[union-attr]
            cls._solver_dt = cls.get_physics_dt() / cls._num_substeps

            # Create solver from config
            solver_cfg = cfg.solver_cfg  # type: ignore[union-attr]
            cfg_dict = solver_cfg.to_dict() if hasattr(solver_cfg, "to_dict") else {}
            cls._solver_type = cfg_dict.pop("solver_type", "mujoco_warp")

            if cls._solver_type == "mujoco_warp":
                # SolverMuJoCo does not require distinct input & output states
                cls._use_single_state = True
                cls._solver = SolverMuJoCo(cls._model, **cfg_dict)
            elif cls._solver_type == "xpbd":
                cls._use_single_state = False
                cls._solver = SolverXPBD(cls._model, **cfg_dict)
            elif cls._solver_type == "featherstone":
                cls._use_single_state = False
                cls._solver = SolverFeatherstone(cls._model, **cfg_dict)
            else:
                raise ValueError(f"Invalid solver type: {cls._solver_type}")

            # Determine if we need external collision detection
            # - SolverMuJoCo with use_mujoco_contacts=True: uses internal MuJoCo collision detection
            # - SolverMuJoCo with use_mujoco_contacts=False: needs Newton's unified collision pipeline
            # - Other solvers (XPBD, Featherstone): always need Newton's unified collision pipeline
            if isinstance(cls._solver, SolverMuJoCo):
                # Handle both dict and object configs
                if hasattr(solver_cfg, "use_mujoco_contacts"):
                    use_mujoco_contacts = solver_cfg.use_mujoco_contacts
                elif isinstance(solver_cfg, dict):
                    use_mujoco_contacts = solver_cfg.get("use_mujoco_contacts", False)
                else:
                    use_mujoco_contacts = getattr(solver_cfg, "use_mujoco_contacts", False)
                cls._needs_collision_pipeline = not use_mujoco_contacts
            else:
                cls._needs_collision_pipeline = True

            # Force Newton pipeline when SDF is enabled
            has_sdf = getattr(cfg, "sdf_max_resolution", None) is not None or getattr(cfg, "sdf_target_voxel_size", None) is not None
            if has_sdf and not cls._needs_collision_pipeline:
                logger.warning("SDF collision requires Newton collision pipeline. Overriding use_mujoco_contacts.")
                cls._needs_collision_pipeline = True

            # Initialize contacts and collision pipeline
            cls._initialize_contacts()

        # Ensure we are using a CUDA enabled device
        device = PhysicsManager._device
        assert device.startswith("cuda"), "NewtonManager only supports CUDA enabled devices"

        with Timer(name="newton_cuda_graph", msg="CUDA graph took:", enable=True, format="ms"):
            if cfg.use_cuda_graph:  # type: ignore[union-attr]
                with wp.ScopedCapture() as capture:
                    cls._simulate()
                cls._graph = capture.graph

    @classmethod
    def _simulate(cls) -> None:
        """Run one simulation step with substeps."""

        # MJWarp can use its internal collision pipeline.
        if cls._needs_collision_pipeline:
            cls._collision_pipeline.collide(cls._state_0, cls._contacts)
            contacts = cls._contacts
        else:
            contacts = None

        def step_fn(state_0, state_1):
            cls._solver.step(state_0, state_1, cls._control, contacts, cls._solver_dt)

        if cls._use_single_state:
            for i in range(cls._num_substeps):
                step_fn(cls._state_0, cls._state_0)
                cls._state_0.clear_forces()
        else:
            cfg = PhysicsManager._cfg
            need_copy_on_last_substep = (cfg is not None and cfg.use_cuda_graph) and cls._num_substeps % 2 == 1  # type: ignore[union-attr]

            for i in range(cls._num_substeps):
                step_fn(cls._state_0, cls._state_1)
                if need_copy_on_last_substep and i == cls._num_substeps - 1:
                    cls._state_0.assign(cls._state_1)
                else:
                    cls._state_0, cls._state_1 = cls._state_1, cls._state_0
                cls._state_0.clear_forces()

        # Populate contacts for contact sensors
        if cls._report_contacts:
            # For newton_contacts (unified pipeline): use locally computed contacts
            # For mujoco_contacts: use class-level _contacts, solver populates it from MuJoCo data
            eval_contacts = contacts if contacts is not None else cls._contacts
            cls._solver.update_contacts(eval_contacts, cls._state_0)
            for sensor in cls._newton_contact_sensors.values():
                sensor.eval(eval_contacts)


    @classmethod
    def get_solver_convergence_steps(cls) -> dict[str, float | int]:
        """Get solver convergence statistics."""
        niter = cls._solver.mjw_data.solver_niter.numpy()
        return {
            "max": np.max(niter),
            "mean": np.mean(niter),
            "min": np.min(niter),
            "std": np.std(niter),
        }

    # State accessors (used extensively by articulation/rigid object data)
    @classmethod
    def get_model(cls) -> Model:
        """Get the Newton model."""
        return cls._model

    @classmethod
    def get_state_0(cls) -> State:
        """Get the current state."""
        return cls._state_0

    @classmethod
    def get_state_1(cls) -> State:
        """Get the next state."""
        return cls._state_1

    @classmethod
    def get_control(cls) -> Control:
        """Get the control object."""
        return cls._control

    @classmethod
    def get_dt(cls) -> float:
        """Get the physics timestep. Alias for get_physics_dt()."""
        return cls.get_physics_dt()

    @classmethod
    def get_solver_dt(cls) -> float:
        """Get the solver substep timestep."""
        return cls._solver_dt

    @classmethod
    def add_contact_sensor(
        cls,
        body_names_expr: str | list[str] | None = None,
        shape_names_expr: str | list[str] | None = None,
        contact_partners_body_expr: str | list[str] | None = None,
        contact_partners_shape_expr: str | list[str] | None = None,
        prune_noncolliding: bool = True,
        verbose: bool = False,
    ) -> None:
        """Add a contact sensor for reporting contacts between bodies/shapes.

        Note: Only one contact sensor can be active at a time.

        Args:
            body_names_expr: Expression for body names to sense.
            shape_names_expr: Expression for shape names to sense.
            contact_partners_body_expr: Expression for contact partner body names.
            contact_partners_shape_expr: Expression for contact partner shape names.
            prune_noncolliding: Make force matrix sparse using collision pairs.
            verbose: Print verbose information.
        """
        # Validate inputs
        if body_names_expr is None and shape_names_expr is None:
            raise ValueError("At least one of body_names_expr or shape_names_expr must be provided")
        if body_names_expr is not None and shape_names_expr is not None:
            raise ValueError("Only one of body_names_expr or shape_names_expr must be provided")
        if contact_partners_body_expr is not None and contact_partners_shape_expr is not None:
            raise ValueError("Only one of contact_partners_body_expr or contact_partners_shape_expr must be provided")

        # Log sensor configuration
        sensor_target = body_names_expr or shape_names_expr
        partner_filter = contact_partners_body_expr or contact_partners_shape_expr or "all bodies/shapes"
        logger.info(f"Adding contact sensor for {sensor_target} with filter {partner_filter}")

        # Create unique key for this sensor
        sensor_key = (body_names_expr, shape_names_expr, contact_partners_body_expr, contact_partners_shape_expr)

        # Create and store the sensor
        # Note: SensorContact constructor requests 'force' attribute from the model
        newton_sensor = NewtonContactSensor(
            cls._model,
            sensing_obj_bodies=body_names_expr,
            sensing_obj_shapes=shape_names_expr,
            counterpart_bodies=contact_partners_body_expr,
            counterpart_shapes=contact_partners_shape_expr,
            match_fn=flipped_match,
            include_total=True,
            prune_noncolliding=prune_noncolliding,
            verbose=verbose,
        )
        cls._newton_contact_sensors[sensor_key] = newton_sensor
        cls._report_contacts = True

        # Regenerate contacts only if they were already created without force attribute
        # If solver is not initialized, contacts will be created with force in initialize_solver()
        if cls._solver is not None and cls._contacts is not None:
            # Only regenerate if contacts don't have force attribute (sensor.eval() requires it)
            if cls._contacts.force is None:
                cls._initialize_contacts()

        return sensor_key
