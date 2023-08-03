# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The superclass for Isaac Sim based environments."""


import abc
import enum
import gym
import numpy as np
import torch
from typing import Any, ClassVar, Dict, Iterable, List, Optional, Tuple, Union

import carb
import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.core.utils.stage as stage_utils
import omni.isaac.core.utils.torch as torch_utils
import omni.usd
from omni.isaac.cloner import GridCloner
from omni.isaac.core.utils.viewports import set_camera_view

from omni.isaac.orbit.sensors import *  # noqa: F401, F403
from omni.isaac.orbit.sensors.sensor_base import SensorBase
from omni.isaac.orbit.sim import SimulationContext

from .isaac_env_cfg import IsaacEnvCfg

# Define type aliases here to avoid circular import
VecEnvIndices = Union[int, Iterable[int]]
"""Indices of the sub-environments. Used when we want to access one or more environments."""

VecEnvObs = Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]
"""Observation returned by the environment. It contains the observation for each sub-environment.

The observations are stored in a dictionary. The keys are the group to which the observations belong.
This is useful for various learning setups beyond vanilla reinforcement learning, such as asymmetric
actor-critic, multi-agent, or hierarchical reinforcement learning.

For example, for asymmetric actor-critic, the observation for the actor and the critic can be accessed
using the keys ``"policy"`` and ``"critic"`` respectively.

Within each group, the observations can be stored either as a dictionary with keys as the names of each
observation term in the group, or a single tensor obtained from concatenating all the observation terms.
"""

VecEnvStepReturn = Tuple[VecEnvObs, torch.Tensor, torch.Tensor, Dict]
"""The environment signals processed at the end of each step. It contains the observation, reward, termination
signal and additional information for each sub-environment."""


class RenderMode(enum.Enum):
    """Different UI-based rendering modes."""

    HEADLESS = -1
    """Headless mode."""
    FULL_RENDERING = 0
    """Full rendering."""
    NO_RENDERING = 1
    """No rendering."""


class IsaacEnv(gym.Env):
    """The superclass for Isaac Sim based environments.

    It encapsulates an environment using Omniverse Isaac Sim for under-the-hood
    dynamics and rendering. An environment can be partially or fully-observed.

    Each observation from the environment is a batch of observations for each sub-
    environments. The method :meth:`step` is also expected to receive a batch of actions
    for each sub-environment.

    While the environment itself is implemented as a vectorized environment, we do not
    inherit from :class:`gym.vector.VectorEnv`. This is mainly because the class adds
    various methods (for wait and asynchronous updates) which are not required.
    Additionally, each RL library typically has its own definition for a vectorized
    environment. Thus, to reduce complexity, we directly use the :class:`gym.Env` over
    here and leave it up to library-defined wrappers to take care of wrapping this
    environment for their agents.
    """

    is_vector_env: ClassVar[bool] = True
    """Whether the environment is a vectorized environment."""
    metadata: ClassVar[Dict[str, Any]] = {"render.modes": ["human", "rgb_array"]}
    """Metadata for the environment."""

    def __init__(self, cfg: IsaacEnvCfg, render: bool = False, viewport: bool = False, **kwargs):
        """Initialize the environment.

        We currently support only PyTorch backend for the environment. In the future, we plan to extend this to use
        other backends such as Warp.

        If render is True, then viewport is True, and the full rendering process will take place with
        an interactive GUI available via local machine or streaming. If :obj:`render` is False and :obj:`viewport`
        is True, then only the lighter-weight viewport elements will be rendered in the GUI. However, either of these
        modes will significantly slow down the simulation compared to a non-rendering configuration. Thus, it
        is recommended to set both :obj:`render` and :obj:`viewport` to False when training an environment (unless
        it uses perception sensors).

        Args:
            cfg (IsaacEnvCfg): Instance of the environment configuration.
            render (bool, optional): Whether to render at every simulation step. Defaults to False.
            viewport (bool, optional): Whether to enable the viewport/camera rendering. If True, then the viewport
                will be rendered even if the GUI is disabled or :obj:`render` is False. Defaults to False.

        Raises:
            RuntimeError: No stage is found in the simulation.
        """
        # store inputs to class
        self.cfg = cfg
        self.enable_render = render
        self.enable_viewport = viewport or self.enable_render
        # extract commonly used parameters
        self.num_envs = self.cfg.env.num_envs
        self.device = self.cfg.sim.device
        self.physics_dt = self.cfg.sim.dt
        self.rendering_dt = self.cfg.sim.dt * self.cfg.sim.substeps

        # print useful information
        print("[INFO]: Isaac Orbit environment:")
        print(f"\t\t Number of instances : {self.num_envs}")
        print(f"\t\t Environment device  : {self.device}")
        print(f"\t\t Physics step-size   : {self.physics_dt}")
        print(f"\t\t Rendering step-size : {self.rendering_dt}")

        # check that simulation is running
        if stage_utils.get_current_stage() is None:
            raise RuntimeError("The stage has not been created. Did you run the simulator?")
        # create a simulation context to control the simulator
        self.sim = SimulationContext(self.cfg.sim)
        # create renderer and set camera view
        self._create_viewport_render_product()
        # add flag for checking closing status
        self._is_closed = False

        # we build the GUI only if we are not headless
        if self.enable_render:
            # need to import here to wait for the GUI extension to be loaded
            from omni.kit.viewport.utility import get_active_viewport

            # acquire viewport context
            self._viewport_context = get_active_viewport()
            self._viewport_context.updates_enabled = True
            # build GUI
            self._build_ui()
            # default rendering mode to full
            self.render_mode = RenderMode.FULL_RENDERING
            # counter for periodic rendering
            self._ui_throttle_counter = 0
            # rendering frequency in terms of environment steps
            # TODO: Make this configurable.
            self._ui_throttle_period = 5
            # disable rendering at every step (we render at environment step frequency)
            # TODO: Fix the name of this variable to be more intuitive.
            self.enable_render = False
        else:
            # set viewport context to None
            self._viewport_context = None
            # default rendering mode to no rendering
            self.render_mode = RenderMode.HEADLESS
            # no window made
            self._orbit_window = None
            self._viewport_window = None
        # add timeline event to close the environment
        self.sim.add_timeline_callback("close_env_on_stop", self._stop_simulation_callback)

        # initialize common environment buffers
        self.reward_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        # allocate dictionary to store metrics
        self.extras = {}

        # create a dictionary to store the sensors
        self.sensors = dict()
        # create cloner for duplicating the scenes
        cloner = GridCloner(spacing=self.cfg.env.env_spacing)
        cloner.define_base_env(self.env_ns)
        # create the xform prim to hold the template environment
        if not prim_utils.is_prim_path_valid(self.template_env_ns):
            prim_utils.define_prim(self.template_env_ns)
        # setup single scene
        global_prim_paths = self._design_scene()
        # check if any global prim paths are defined
        if global_prim_paths is None:
            global_prim_paths = list()
        # clone the scenes into the namespace "/World/envs" based on template namespace
        self.envs_prim_paths = cloner.generate_paths(self.env_ns + "/env", self.num_envs)
        self.envs_positions = cloner.clone(
            source_prim_path=self.template_env_ns,
            prim_paths=self.envs_prim_paths,
            replicate_physics=self.cfg.env.replicate_physics,
        )
        # convert environment positions to torch tensor
        # self.envs_positions = torch.tensor(self.envs_positions, dtype=torch.float, device=self.device)
        # TODO Move to right position
        self.envs_positions = self.terrain_importer.env_origins
        # filter collisions within each environment instance
        physics_scene_path = self.sim.get_physics_context().prim_path
        cloner.filter_collisions(
            physics_scene_path, "/World/collisions", prim_paths=self.envs_prim_paths, global_paths=global_prim_paths
        )

    def __del__(self):
        """Close the environment."""
        self.close()

    """
    Properties
    """

    @property
    def env_ns(self) -> str:
        """The namespace ``/World/envs`` in which all environments created.

        The environments are present w.r.t. this namespace under "env_{N}" prim,
        where N is a natural number.
        """
        return "/World/envs"

    @property
    def template_env_ns(self) -> str:
        """The namespace ``/World/envs/env_0`` used for the template environment.

        All prims under this namespace are cloned during construction.
        """
        return self.env_ns + "/env_0"

    """
    Operations - MDP
    """

    @staticmethod
    def seed(seed: int = -1):
        """Set the seed for the environment.

        Args:
            seed (int, optional): The seed for random generator. Defaults to -1.
        """
        import omni.replicator.core as rep

        rep.set_global_seed(seed)
        return torch_utils.set_seed(seed)

    def reset(self) -> VecEnvObs:
        """Flags all environments for reset.

        Note:
            Once the buffers for resetting the environments are set, a simulation step is
            taken to forward all other buffers.

        Returns:
            VecEnvObs: Observations from the environment.
        """
        # reset state of scene
        indices = torch.arange(self.num_envs, dtype=torch.int64, device=self.device)
        self._reset_idx(indices)
        # perform one step to have buffers into action
        self.sim.step(render=False)  # TODO why do we need this ?
        # compute common quantities
        self._cache_common_quantities()
        # return observations
        return self._get_observations()

    def step(self, actions: torch.Tensor) -> VecEnvStepReturn:
        """Reset any terminated environments and apply actions on the environment.

        This function deals with various timeline events (play, pause and stop) for clean execution.
        When the simulation is stopped all the physics handles expire and we cannot perform any read or
        write operations. The timeline event is only detected after every `sim.step()` call. Hence, at
        every call we need to check the status of the simulator. The logic is as follows:

        1. If the simulation is stopped, the environment is closed and the simulator is shutdown.
        2. If the simulation is paused, we step the simulator until it is playing.
        3. If the simulation is playing, we set the actions and step the simulator.

        Args:
            actions (torch.Tensor): Actions to apply on the simulator.

        Returns:
            VecEnvStepReturn: A tuple containing:
                - (VecEnvObs) observations from the environment
                - (torch.Tensor) reward from the environment
                - (torch.Tensor) whether the current episode is completed or not
                - (dict) misc information
        """
        # check if the simulation timeline is paused. in that case keep stepping until it is playing
        if not self.sim.is_playing():
            # step the simulator (but not the physics) to have UI still active
            while not self.sim.is_playing():
                self.sim.render()
                # meantime if someone stops, break out of the loop
                if self.sim.is_stopped():
                    break
            # need to do one step to refresh the app
            # reason: physics has to parse the scene again and inform other extensions like hydra-delegate.
            #   without this the app becomes unresponsive.
            # FIXME: This steps physics as well, which we is not good in general.
            self.sim.app.update()

        # perform the stepping of simulation
        self._step_impl(actions)

        # periodically update the UI to keep it responsive
        if self.render_mode == RenderMode.NO_RENDERING:
            self._ui_throttle_counter += 1
            if self._ui_throttle_counter % self._ui_throttle_period == 0:
                self._ui_throttle_counter = 0
                # here we don't render viewport so don't need to flush flatcache
                self.sim.render(flush=False)
        elif self.render_mode == RenderMode.FULL_RENDERING:
            # perform debug visualization
            if self.cfg.viewer.debug_vis:
                self._debug_vis()
            # render the scene
            self.sim.render()

        # return observations, rewards, resets and extras
        return self._get_observations(), self.reward_buf, self.reset_buf, self.extras

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """Run rendering without stepping through the physics.

        By convention, if mode is:

        - **human**: render to the current display and return nothing. Usually for human consumption.
        - **rgb_array**: Return an numpy.ndarray with shape (x, y, 3), representing RGB values for an
          x-by-y pixel image, suitable for turning into a video.

        Args:
            mode (str, optional): The mode to render with. Defaults to "human".
        """
        # render the scene only if rendering at every step is disabled
        # this is because we do not want to render the scene twice
        if not self.enable_render:
            # render the scene
            self.sim.render()
        # decide the rendering mode
        if mode == "human":
            return None
        elif mode == "rgb_array":
            # check if viewport is enabled -- if not, then complain because we won't get any data
            if not self.enable_viewport:
                raise RuntimeError(
                    f"Cannot render '{mode}' when enable viewport is False. Please check the provided"
                    "arguments to the environment class at initialization."
                )
            # obtain the rgb data
            rgb_data = self._rgb_annotator.get_data()
            # convert to numpy array
            rgb_data = np.frombuffer(rgb_data, dtype=np.uint8).reshape(*rgb_data.shape)
            # return the rgb data
            return rgb_data[:, :, :3]
        else:
            raise NotImplementedError(
                f"Render mode '{mode}' is not supported. Please use: {self.metadata['render.modes']}."
            )

    def close(self):
        """Cleanup for the environment."""
        if not self._is_closed:
            # stop physics simulation (precautionary)
            self.sim.stop()
            # update closing status
            self._is_closed = True

    """
    Operations - Scene
    """

    def enable_sensor(self, sensor_name: str):
        """Adds a sensor to the environment.

        Args:
            sensor_name (str): Name of the sensor to add.

        """
        sensor_cfg = self.cfg.sensors.__getattribute__(sensor_name)
        if sensor_name not in self.sensors.keys():
            sensor: SensorBase = eval(sensor_cfg.cls_name)(sensor_cfg)
            # sensor.spawn("")  # TODO: do we need to call spawn earlier?
            sensor.initialize(self.env_ns + "/.*")
            self.sensors[sensor_name] = sensor

    """
    Implementation specifics.
    """

    @abc.abstractmethod
    def _design_scene(self) -> Optional[List[str]]:
        """Creates the template environment scene.

        All prims under the *template namespace* will be duplicated across the
        stage and collisions between the duplicates will be filtered out. In case,
        there are any prims which need to be a common collider across all the
        environments, they should be returned as a list of prim paths. These could
        be prims like the ground plane, walls, etc.

        Returns:
            Optional[List[str]]: List of prim paths which are common across all the
                environments and need to be considered for common collision filtering.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _reset_idx(self, env_ids: VecEnvIndices):
        """Resets the MDP for given environment instances.

        Args:
            env_ids (VecEnvIndices): Indices of environment instances to reset.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _step_impl(self, actions: torch.Tensor):
        """Apply actions on the environment, computes MDP signals and perform resets.

        This function sets the simulation buffers for actions to apply, perform
        stepping of the simulation, computes the MDP signals for reward and
        termination, and perform resetting of the environment based on their termination.

        Note:
            The environment specific implementation of this function is responsible for also
            stepping the simulation. To have a clean exit when the timeline is stopped
            through the UI, the implementation should check the simulation status
            after stepping the simulator and return if the simulation is stopped.

            .. code-block:: python

                # simulate
                self.sim.step(render=self.enable_render)
                # check that simulation is playing
                if self.sim.is_stopped():
                    return

        Args:
            actions (torch.Tensor): Actions to apply on the environment.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _get_observations(self) -> VecEnvObs:
        """Grabs observations from the environment.

        The observations are stored in a dictionary. The keys are the group to which the observations belong.
        This is useful for various learning setups beyond vanilla reinforcement learning, such as asymmetric
        actor-critic, multi-agent, or hierarchical reinforcement learning.

        By default, most learning frameworks deal with default and privileged observations in different ways.
        This handling must be taken care of by the wrapper around the :class:`IsaacEnv` instance.

        Note:
            For included frameworks (RSL-RL, RL-Games, skrl), the observations must have the key "policy". In case,
            the key "critic" is also present, then the critic observations are taken from the "critic" group.
            Otherwise, they are the same as the "policy" group.

        Returns:
            VecEnvObs: Observations from the environment.
        """
        raise NotImplementedError

    def _debug_vis(self):
        """Visualize the environment for debugging purposes.

        This function can be overridden by the environment to perform any additional
        visualizations such as markers for frames, goals, etc.

        Note:
            This is called only when the viewport is enabled, i.e, the render mode is FULL_RENDERING.
        """
        pass

    """
    Helper functions - MDP.
    """

    def _cache_common_quantities(self) -> None:
        """Cache common quantities from simulator used for computing MDP signals.

        Implementing this function is optional. It is mostly useful in scenarios where
        shared quantities between observations, rewards and termination signals need to be
        computed only once.

        Note:
            The function should be called after performing simulation stepping and before
            computing any MDP signals.
        """
        pass

    """
    Helper functions - Simulation.
    """

    def _create_viewport_render_product(self):
        """Create a render product of the viewport for rendering."""
        # set camera view for "/OmniverseKit_Persp" camera
        set_camera_view(eye=self.cfg.viewer.eye, target=self.cfg.viewer.lookat)

        # check if viewport is enabled before creating render product
        if self.enable_viewport:
            import omni.replicator.core as rep

            # create render product
            self._render_product = rep.create.render_product("/OmniverseKit_Persp", self.cfg.viewer.resolution)
            # create rgb annotator -- used to read data from the render product
            self._rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
            self._rgb_annotator.attach([self._render_product])
        else:
            carb.log_info("Viewport is disabled. Skipping creation of render product.")

    def _stop_simulation_callback(self, event: carb.events.IEvent):
        """Callback for when the simulation is stopped."""
        # check if the simulation is stopped
        if event.type == int(omni.timeline.TimelineEventType.STOP):
            carb.log_warn("Simulation is stopped. Closing the environment. This might take a few seconds.")
            # close the environment
            # we do this so that wrappers can clean up
            self.close()
            # shutdown the simulator
            self.sim.app.shutdown()

    """
    Helper functions - GUI.
    """

    def _build_ui(self):
        """Constructs the GUI for the environment."""
        # need to import here to wait for the GUI extension to be loaded
        import omni.isaac.ui.ui_utils as ui_utils
        import omni.ui as ui

        # do a sim update to finish loading
        for _ in range(10):
            self.sim.app.update()
        # acquire viewport window
        self._viewport_window = ui.Workspace.get_window("Viewport")
        # create window for UI
        self._orbit_window = omni.ui.Window(
            "Orbit", width=400, height=500, visible=True, dock_preference=ui.DockPreference.RIGHT_TOP
        )
        # dock next to properties window
        property_window = ui.Workspace.get_window("Property")
        self._orbit_window.dock_in(property_window, ui.DockPosition.SAME, 1.0)
        self._orbit_window.focus()
        # do a sim update to finish loading
        for _ in range(10):
            self.sim.app.update()

        # keep a dictionary of stacks so that child environments can add their own UI elements
        # this can be done by using the `with` context manager
        self._orbit_window_elements = dict()
        # create main frame
        self._orbit_window_elements["main_frame"] = self._orbit_window.frame
        with self._orbit_window_elements["main_frame"]:
            # create main stack
            self._orbit_window_elements["main_vstack"] = ui.VStack(spacing=5, height=0)
            with self._orbit_window_elements["main_vstack"]:
                # create collapsable frame for controls
                self._orbit_window_elements["control_frame"] = ui.CollapsableFrame(
                    title="Controls",
                    width=ui.Fraction(1),
                    height=0,
                    collapsed=False,
                    style=ui_utils.get_style(),
                    horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
                    vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
                )
                with self._orbit_window_elements["control_frame"]:
                    # create stack for controls
                    self._orbit_window_elements["controls_vstack"] = ui.VStack(spacing=5, height=0)
                    with self._orbit_window_elements["controls_vstack"]:
                        # create rendering mode dropdown
                        render_mode_cfg = {
                            "label": "Rendering Mode",
                            "type": "dropdown",
                            "default_val": 0,
                            "items": [member.name for member in RenderMode if member.value >= 0],
                            "tooltip": "Select a rendering mode",
                            "on_clicked_fn": self._on_render_mode_select,
                        }
                        self._orbit_window_elements["render_dropdown"] = ui_utils.dropdown_builder(**render_mode_cfg)
                        # create debug visualization checkbox
                        debug_vis_checkbox = {
                            "label": "Debug Visualization",
                            "type": "checkbox",
                            "default_val": self.cfg.viewer.debug_vis,
                            "tooltip": "Toggle environment debug visualization",
                            "on_clicked_fn": self._toggle_debug_visualization_flag,
                        }
                        self._orbit_window_elements["debug_checkbox"] = ui_utils.cb_builder(**debug_vis_checkbox)

    def _on_render_mode_select(self, value: str):
        """Callback for when the rendering mode is selected."""
        if value == RenderMode.FULL_RENDERING.name:
            self._viewport_context.updates_enabled = True
            self._viewport_window.visible = True
            # update flags for rendering
            self.render_mode = RenderMode.FULL_RENDERING
        elif value == RenderMode.NO_RENDERING.name:
            self._viewport_context.updates_enabled = False
            self._viewport_window.visible = False  # hide viewport
            # update flags for rendering
            self.render_mode = RenderMode.NO_RENDERING
            self._ui_throttle_counter = 0
        else:
            carb.log_error(f"Unknown rendering mode selected: {value}. Please select a valid rendering mode.")

    def _toggle_debug_visualization_flag(self, value: bool):
        """Toggle environment debug visualization flag."""
        self.cfg.viewer.debug_vis = value
