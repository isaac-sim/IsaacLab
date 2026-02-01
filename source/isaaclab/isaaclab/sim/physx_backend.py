# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PhysX physics backend for SimulationContext."""

from __future__ import annotations

import glob
import logging
import os
import re
import time
import torch
from datetime import datetime
from typing import TYPE_CHECKING

import carb
import omni.kit.app
import omni.physx
import omni.physics.tensors
from pxr import Usd, Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.sim.simulation_manager import SimulationManager

if TYPE_CHECKING:
    from isaaclab.sim.simulation_context import SimulationContext

logger = logging.getLogger(__name__)


class AnimationRecorder:
    """Handles animation recording for the simulation.

    This class manages the recording of physics animations using the PhysX PVD
    (Physics Visual Debugger) interface. It handles the setup, update, and
    finalization of animation recordings.
    """

    def __init__(self, carb_settings: carb.settings.ISettings, app_iface: omni.kit.app.IApp):
        """Initialize the animation recorder.

        Args:
            carb_settings: The Carbonite settings interface.
            app_iface: The Omniverse Kit application interface.
        """
        self._carb_settings = carb_settings
        self._app_iface = app_iface
        self._enabled = False
        self._start_time: float = 0.0
        self._stop_time: float = 0.0
        self._started_timestamp: float | None = None
        self._output_dir: str = ""
        self._timestamp: str = ""
        self._physx_pvd_interface = None

        self._setup()

    @property
    def enabled(self) -> bool:
        """Whether animation recording is enabled."""
        return self._enabled

    def _setup(self) -> None:
        """Sets up animation recording settings and initializes the recording."""
        self._enabled = bool(self._carb_settings.get("/isaaclab/anim_recording/enabled"))
        if not self._enabled:
            return

        # Import omni.physx.pvd.bindings here since it is not available by default
        from omni.physxpvd.bindings import _physxPvd

        # Init anim recording settings
        self._start_time = self._carb_settings.get("/isaaclab/anim_recording/start_time")
        self._stop_time = self._carb_settings.get("/isaaclab/anim_recording/stop_time")
        self._started_timestamp = None

        # Make output path relative to repo path
        repo_path = os.path.join(carb.tokens.get_tokens_interface().resolve("${app}"), "..")
        self._timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
        self._output_dir = (
            os.path.join(repo_path, "anim_recordings", self._timestamp).replace("\\", "/").rstrip("/") + "/"
        )
        os.makedirs(self._output_dir, exist_ok=True)

        # Acquire physx pvd interface and set output directory
        self._physx_pvd_interface = _physxPvd.acquire_physx_pvd_interface()

        # Set carb settings for the output path and enabling pvd recording
        self._carb_settings.set_string("/persistent/physics/omniPvdOvdRecordingDirectory", self._output_dir)
        self._carb_settings.set_bool("/physics/omniPvdOutputEnabled", True)

    def update(self) -> bool:
        """Tracks timestamps and triggers finish if total time has elapsed.

        Returns:
            True if animation recording has finished, False otherwise.
        """
        if not self._enabled:
            return False

        if self._started_timestamp is None:
            self._started_timestamp = time.time()

        total_time = time.time() - self._started_timestamp
        if total_time > self._stop_time:
            self._finish()
            return True
        return False

    def _finish(self) -> bool:
        """Finishes the animation recording and outputs the baked animation recording.

        Returns:
            True if the recording was successfully finished.
        """
        logger.warning(
            "[INFO][SimulationContext]: Finishing animation recording. Stage must be saved. Might take a few minutes."
        )

        # Detaching the stage will also close it and force the serialization of the OVD file
        physx = omni.physx.get_physx_simulation_interface()
        physx.detach_stage()

        # Save stage to disk
        stage_path = os.path.join(self._output_dir, "stage_simulation.usdc")
        sim_utils.save_stage(stage_path, save_and_reload_in_place=False)

        # Find the latest ovd file not named tmp.ovd
        ovd_files = [f for f in glob.glob(os.path.join(self._output_dir, "*.ovd")) if not f.endswith("tmp.ovd")]
        input_ovd_path = max(ovd_files, key=os.path.getctime)

        # Invoke pvd interface to create recording
        stage_filename = "baked_animation_recording.usda"
        result = self._physx_pvd_interface.ovd_to_usd_over_with_layer_creation(
            input_ovd_path,
            stage_path,
            self._output_dir,
            stage_filename,
            self._start_time,
            self._stop_time,
            True,  # True: ASCII layers / False : USDC layers
            False,  # True: verify over layer
        )

        # Workaround for manually setting the truncated start time in the baked animation recording
        self._update_usda_start_time(os.path.join(self._output_dir, stage_filename), self._start_time)

        # Disable recording
        self._carb_settings.set_bool("/physics/omniPvdOutputEnabled", False)

        return result

    @staticmethod
    def _update_usda_start_time(file_path: str, start_time: float) -> None:
        """Updates the start time of the USDA baked animation recording file.

        Args:
            file_path: Path to the USDA file.
            start_time: The new start time to set.
        """
        # Read the USDA file
        with open(file_path) as file:
            content = file.read()

        # Extract the timeCodesPerSecond value
        time_code_match = re.search(r"timeCodesPerSecond\s*=\s*(\d+)", content)
        if not time_code_match:
            raise ValueError("timeCodesPerSecond not found in the file.")
        time_codes_per_second = int(time_code_match.group(1))

        # Compute the new start time code
        new_start_time_code = int(start_time * time_codes_per_second)

        # Replace the startTimeCode in the file
        content = re.sub(r"startTimeCode\s*=\s*\d+", f"startTimeCode = {new_start_time_code}", content)

        # Write the updated content back to the file
        with open(file_path, "w") as file:
            file.write(content)


class PhysXBackend:
    """PhysX physics backend.

    This class manages the PhysX physics simulation, including:
    - Physics scene creation and configuration
    - Device settings (CPU/GPU)
    - Timestep and solver configuration
    - Fabric interface for fast data access
    - Physics stepping and reset

    Lifecycle: __init__() -> reset() -> step() (repeated) -> close()
    """

    def __init__(self, sim_context: "SimulationContext"):
        """Initialize the PhysX backend.

        Args:
            sim_context: Parent simulation context.
        """
        self._sim = sim_context

        # acquire physics interfaces
        self._physx_iface = omni.physx.get_physx_interface()
        self._physx_sim_iface = omni.physx.get_physx_simulation_interface()

        # Initialize physics device (will be set in _apply_physics_settings)
        self._physics_device: str = "cpu"

        # Fabric interface (will be set in _load_fabric_interface)
        self._fabric_iface = None
        self._update_fabric = None

        # Physics scene references (will be set in _init_physics_scene)
        self._physics_scene = None
        self._physx_scene_api = None

        # Initialize physics
        self._init_physics_scene()
        self._configure_simulation_dt()
        self._apply_physics_settings()
        SimulationManager.initialize()

        # a stage update here is needed for the case when physics_dt != rendering_dt, otherwise the app crashes
        # when in headless mode
        self._sim.set_setting("/app/player/playSimulations", False)
        self._sim.app.update()
        self._sim.set_setting("/app/player/playSimulations", True)

        # load flatcache/fabric interface
        self._load_fabric_interface()

        # initialize animation recorder
        self._anim_recorder = AnimationRecorder(self._sim.carb_settings, self._sim.app)

    @property
    def device(self) -> str:
        """Device used for physics simulation."""
        return self._physics_device

    @property
    def physics_dt(self) -> float:
        """Physics timestep."""
        return self._sim.cfg.dt

    @property
    def physics_sim_view(self) -> "omni.physics.tensors.SimulationView":
        """Physics simulation view with torch backend."""
        return SimulationManager.get_physics_sim_view()

    def reset(self, soft: bool = False) -> None:
        """Reset the physics simulation.

        Args:
            soft: If True, skip full reinitialization.
        """
        if not soft:
            # initialize the physics simulation
            if SimulationManager.get_physics_sim_view() is None:
                SimulationManager.initialize_physics()

        # app.update() may be changing the cuda device in reset, so we force it back to our desired device here
        if "cuda" in self._physics_device:
            torch.cuda.set_device(self._physics_device)

        # enable kinematic rendering with fabric
        physics_sim_view = SimulationManager.get_physics_sim_view()
        if physics_sim_view is not None:
            physics_sim_view._backend.initialize_kinematic_bodies()

    def forward(self) -> None:
        """Update articulation kinematics and fabric for rendering."""
        if self._fabric_iface is not None:
            physics_sim_view = SimulationManager.get_physics_sim_view()
            if physics_sim_view is not None and self._sim.is_playing():
                # Update the articulations' link's poses before rendering
                physics_sim_view.update_articulations_kinematic()
            self._update_fabric(0.0, 0.0)

    def step(self, render: bool = True) -> bool:
        """Step the physics simulation.

        Args:
            render: If True, step via app.update() which includes rendering.
                   If False, step physics only without rendering.

        Returns:
            True if animation recording finished and app should shutdown, False otherwise.
        """
        # update animation recorder if enabled
        if self._anim_recorder.enabled and self._anim_recorder.update():
            logger.warning("Animation recording finished. Closing app.")
            self._sim.app.shutdown()
            return True

        if render:
            self._sim.app.update()
        else:
            self._physx_sim_iface.simulate(self._sim.cfg.dt, 0.0)
            self._physx_sim_iface.fetch_results()

        # app.update() may be changing the cuda device in step, so we force it back to our desired device here
        if "cuda" in self._physics_device:
            torch.cuda.set_device(self._physics_device)

        return False

    def close(self) -> None:
        """Clean up physics resources."""
        # clear the simulation manager state (notifies assets to cleanup)
        SimulationManager.clear()
        # detach the stage from physx
        if self._physx_sim_iface is not None:
            self._physx_sim_iface.detach_stage()

    # -------------------------
    # Private initialization methods
    # -------------------------

    def _init_physics_scene(self) -> None:
        """Initialize the USD physics scene."""
        stage = self._sim.stage
        cfg = self._sim.cfg

        with sim_utils.use_stage(stage):
            # correct conventions for metric units
            UsdGeom.SetStageUpAxis(stage, "Z")
            UsdGeom.SetStageMetersPerUnit(stage, 1.0)
            UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)

            # find if any physics prim already exists and delete it
            for prim in stage.Traverse():
                if prim.HasAPI(PhysxSchema.PhysxSceneAPI) or prim.GetTypeName() == "PhysicsScene":
                    sim_utils.delete_prim(prim.GetPath().pathString, stage=stage)

            # create a new physics scene
            if stage.GetPrimAtPath(cfg.physics_prim_path).IsValid():
                raise RuntimeError(f"A prim already exists at path '{cfg.physics_prim_path}'.")

            self._physics_scene = UsdPhysics.Scene.Define(stage, cfg.physics_prim_path)
            self._physx_scene_api = PhysxSchema.PhysxSceneAPI.Apply(self._physics_scene.GetPrim())

    def _configure_simulation_dt(self):
        """Configures the simulation step size based on the physics and rendering step sizes."""
        cfg = self._sim.cfg
        stage = self._sim.stage
        carb_settings = self._sim.carb_settings
        timeline_iface = self._sim._timeline_iface

        # if rendering is called the substeps term is used to determine
        # how many physics steps to perform per rendering step.
        # it is not used if step(render=False).
        render_interval = max(cfg.render_interval, 1)

        # set simulation step per second
        steps_per_second = int(1.0 / cfg.dt)
        self._physx_scene_api.CreateTimeStepsPerSecondAttr(steps_per_second)
        # set minimum number of steps per frame
        min_steps = int(steps_per_second / render_interval)
        carb_settings.set_int("/persistent/simulation/minFrameRate", min_steps)

        # compute rendering frequency
        rendering_hz = int(1.0 / (cfg.dt * render_interval))

        # If rate limiting is enabled, set the rendering rate to the specified value
        # Otherwise run the app as fast as possible and do not specify the target rate
        if carb_settings.get("/app/runLoops/main/rateLimitEnabled"):
            carb_settings.set_int("/app/runLoops/main/rateLimitFrequency", rendering_hz)
            timeline_iface.set_target_framerate(rendering_hz)
        with Usd.EditContext(stage, stage.GetRootLayer()):
            stage.SetTimeCodesPerSecond(rendering_hz)
        timeline_iface.set_time_codes_per_second(rendering_hz)
        # The isaac sim loop runner is enabled by default in isaac sim apps,
        # but in case we are in an app with the kit loop runner, protect against this
        try:
            import omni.kit.loop._loop as omni_loop

            _loop_runner = omni_loop.acquire_loop_interface()
            _loop_runner.set_manual_step_size(cfg.dt * render_interval)
            _loop_runner.set_manual_mode(True)
        except Exception:
            self._sim.logger.warning(
                "Isaac Sim loop runner not found, enabling rate limiting to support rendering at specified rate"
            )
            carb_settings.set_bool("/app/runLoops/main/rateLimitEnabled", True)
            carb_settings.set_int("/app/runLoops/main/rateLimitFrequency", rendering_hz)
            timeline_iface.set_target_framerate(rendering_hz)

    def _apply_physics_settings(self):
        """Sets various carb physics settings."""
        cfg = self._sim.cfg
        carb_settings = self._sim.carb_settings

        # --------------------------
        # Carb Physics API settings
        # --------------------------

        # enable hydra scene-graph instancing
        # note: this allows rendering of instanceable assets on the GUI
        carb_settings.set_bool("/persistent/omnihydra/useSceneGraphInstancing", True)
        # change dispatcher to use the default dispatcher in PhysX SDK instead of carb tasking
        # note: dispatcher handles how threads are launched for multi-threaded physics
        carb_settings.set_bool("/physics/physxDispatcher", True)
        # disable contact processing in omni.physx
        # note: we disable it by default to avoid the overhead of contact processing when it isn't needed.
        #   The physics flag gets enabled when a contact sensor is created.
        if hasattr(cfg, "disable_contact_processing"):
            self._sim.logger.warning(
                "The `disable_contact_processing` attribute is deprecated and always set to True"
                " to avoid unnecessary overhead. Contact processing is automatically enabled when"
                " a contact sensor is created, so manual configuration is no longer required."
            )
        # FIXME: From investigation, it seems this flag only affects CPU physics. For GPU physics, contacts
        #  are always processed. The issue is reported to the PhysX team by @mmittal.
        carb_settings.set_bool("/physics/disableContactProcessing", True)
        # disable custom geometry for cylinder and cone collision shapes to allow contact reporting for them
        # reason: cylinders and cones aren't natively supported by PhysX so we need to use custom geometry flags
        # reference: https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/Geometry.html?highlight=capsule#geometry
        carb_settings.set_bool("/physics/collisionConeCustomGeometry", False)
        carb_settings.set_bool("/physics/collisionCylinderCustomGeometry", False)
        # hide the Simulation Settings window
        carb_settings.set_bool("/physics/autoPopupSimulationOutputWindow", False)

        # handle device settings
        if "cuda" in cfg.device:
            parsed_device = cfg.device.split(":")
            if len(parsed_device) == 1:
                # if users only specified "cuda", we check if carb settings provide a valid device id
                # otherwise, we default to device id 0
                device_id = carb_settings.get_as_int("/physics/cudaDevice")
                if device_id < 0:
                    carb_settings.set_int("/physics/cudaDevice", 0)
                    device_id = 0
            else:
                # if users specified "cuda:N", we use the provided device id
                device_id = int(parsed_device[1])
                carb_settings.set_int("/physics/cudaDevice", device_id)
            # suppress readback for GPU physics
            carb_settings.set_bool("/physics/suppressReadback", True)
            # save the device
            self._physics_device = f"cuda:{device_id}"
        else:
            # enable USD read/write operations for CPU physics
            carb_settings.set_int("/physics/cudaDevice", -1)
            carb_settings.set_bool("/physics/suppressReadback", False)
            # save the device
            self._physics_device = "cpu"

        # Configure simulation manager backend
        # Isaac Lab always uses torch tensors for consistency, even on CPU
        SimulationManager.set_backend("torch")
        SimulationManager.set_physics_sim_device(self._physics_device)

        # --------------------------
        # USDPhysics API settings
        # --------------------------

        # set gravity
        gravity = self._sim._gravity_tensor
        gravity_magnitude = torch.norm(gravity).item()
        # avoid division by zero
        if gravity_magnitude == 0.0:
            gravity_magnitude = 1.0
        gravity_direction = gravity / gravity_magnitude
        gravity_direction = gravity_direction.tolist()

        self._physics_scene.CreateGravityDirectionAttr(Gf.Vec3f(*gravity_direction))
        self._physics_scene.CreateGravityMagnitudeAttr(gravity_magnitude)

        # create the default physics material
        # this material is used when no material is specified for a primitive
        material_path = f"{cfg.physics_prim_path}/defaultMaterial"
        cfg.physics_material.func(material_path, cfg.physics_material)
        # bind the physics material to the scene
        sim_utils.bind_physics_material(cfg.physics_prim_path, material_path)

        # --------------------------
        # PhysX API settings
        # --------------------------

        # set broadphase type
        broadphase_type = "GPU" if "cuda" in cfg.device else "MBP"
        self._physx_scene_api.CreateBroadphaseTypeAttr(broadphase_type)
        # set gpu dynamics
        enable_gpu_dynamics = "cuda" in cfg.device
        self._physx_scene_api.CreateEnableGPUDynamicsAttr(enable_gpu_dynamics)

        # GPU-dynamics does not support CCD, so we disable it if it is enabled.
        if enable_gpu_dynamics and cfg.physx.enable_ccd:
            cfg.physx.enable_ccd = False
            self._sim.logger.warning(
                "CCD is disabled when GPU dynamics is enabled. Please disable CCD in the PhysxCfg config to avoid this"
                " warning."
            )
        self._physx_scene_api.CreateEnableCCDAttr(cfg.physx.enable_ccd)

        # set solver type
        solver_type = "PGS" if cfg.physx.solver_type == 0 else "TGS"
        self._physx_scene_api.CreateSolverTypeAttr(solver_type)

        # set solve articulation contact last
        attr = self._physx_scene_api.GetPrim().CreateAttribute(
            "physxScene:solveArticulationContactLast", Sdf.ValueTypeNames.Bool
        )
        attr.Set(cfg.physx.solve_articulation_contact_last)

        # iterate over all the settings and set them
        for key, value in cfg.physx.to_dict().items():  # type: ignore
            if key in ["solver_type", "enable_ccd", "solve_articulation_contact_last"]:
                continue
            if key == "bounce_threshold_velocity":
                key = "bounce_threshold"
            sim_utils.safe_set_attribute_on_usd_schema(self._physx_scene_api, key, value, camel_case=True)

        # throw warnings for helpful guidance
        if cfg.physx.solver_type == 1 and not cfg.physx.enable_external_forces_every_iteration:
            logger.warning(
                "The `enable_external_forces_every_iteration` parameter in the PhysxCfg is set to False. If you are"
                " experiencing noisy velocities, consider enabling this flag. You may need to slightly increase the"
                " number of velocity iterations (setting it to 1 or 2 rather than 0), together with this flag, to"
                " improve the accuracy of velocity updates."
            )

        if not cfg.physx.enable_stabilization and cfg.dt > 0.0333:
            self._sim.logger.warning(
                "Large simulation step size (> 0.0333 seconds) is not recommended without enabling stabilization."
                " Consider setting the `enable_stabilization` flag to True in the PhysxCfg, or reducing the"
                " simulation step size if you run into physics issues."
            )

    def _load_fabric_interface(self):
        """Loads the fabric interface if enabled."""
        import omni.kit.app

        cfg = self._sim.cfg
        carb_settings = self._sim.carb_settings

        extension_manager = omni.kit.app.get_app().get_extension_manager()
        fabric_enabled = extension_manager.is_extension_enabled("omni.physx.fabric")

        if cfg.use_fabric:
            if not fabric_enabled:
                extension_manager.set_extension_enabled_immediate("omni.physx.fabric", True)

            # load fabric interface
            from omni.physxfabric import get_physx_fabric_interface

            # acquire fabric interface
            self._fabric_iface = get_physx_fabric_interface()
            if hasattr(self._fabric_iface, "force_update"):
                # The update method in the fabric interface only performs an update if a physics step has occurred.
                # However, for rendering, we need to force an update since any element of the scene might have been
                # modified in a reset (which occurs after the physics step) and we want the renderer to be aware of
                # these changes.
                self._update_fabric = self._fabric_iface.force_update
            else:
                # Needed for backward compatibility with older Isaac Sim versions
                self._update_fabric = self._fabric_iface.update
        else:
            if fabric_enabled:
                extension_manager.set_extension_enabled_immediate("omni.physx.fabric", False)
            # set fabric interface to None
            self._fabric_iface = None

        # set carb settings for fabric
        carb_settings.set_bool("/physics/fabricEnabled", cfg.use_fabric)
        carb_settings.set_bool("/physics/updateToUsd", not cfg.use_fabric)
        carb_settings.set_bool("/physics/updateParticlesToUsd", not cfg.use_fabric)
        carb_settings.set_bool("/physics/updateVelocitiesToUsd", not cfg.use_fabric)
        carb_settings.set_bool("/physics/updateForceSensorsToUsd", not cfg.use_fabric)
        carb_settings.set_bool("/physics/updateResidualsToUsd", not cfg.use_fabric)
        # disable simulation output window visibility
        carb_settings.set_bool("/physics/visualizationDisplaySimulationOutput", False)

    def is_fabric_enabled(self) -> bool:
        """Returns whether the fabric interface is enabled."""
        return self._fabric_iface is not None
