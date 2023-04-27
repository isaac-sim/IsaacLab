# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The superclass for Isaac Sim based environments."""


import abc
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
from omni.isaac.core.simulation_context import SimulationContext
from omni.isaac.core.utils.extensions import enable_extension
from omni.isaac.core.utils.viewports import set_camera_view

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

    def __init__(self, cfg: IsaacEnvCfg, headless: bool = False, viewport: bool = False, **kwargs):
        """Initialize the environment.

        We currently support only PyTorch backend for the environment. In the future, we plan to extend this to use
        other backends such as Warp.

        If the environment is not headless and viewport is enabled, then the viewport will be rendered in the GUI.
        This allows us to render the environment even in the headless mode. However, it will significantly slow
        down the simulation. Thus, it is recommended to set both ``headless`` and ``viewport``
        to ``False`` when training an environment (unless it uses perception sensors).

        Args:
            cfg (IsaacEnvCfg): Instance of the environment configuration.
            headless (bool, optional): Whether to render at every simulation step. Defaults to False.
            viewport (bool, optional): Whether to enable the GUI viewport. If True, then the viewport
                will be rendered in the GUI (even in the headless mode). Defaults to False.

        Raises:
            RuntimeError: No stage is found in the simulation.
        """
        # store inputs to class
        self.cfg = cfg
        self.enable_render = not headless
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
        # flatten out the simulation dictionary
        sim_params = self.cfg.sim.to_dict()
        if sim_params is not None:
            if "physx" in sim_params:
                physx_params = sim_params.pop("physx")
                sim_params.update(physx_params)
        # set flags for simulator
        self._configure_simulation_flags(sim_params)
        # create a simulation context to control the simulator
        self.sim = SimulationContext(
            stage_units_in_meters=1.0,
            physics_dt=self.physics_dt,
            rendering_dt=self.rendering_dt,
            backend="torch",
            sim_params=sim_params,
            physics_prim_path="/physicsScene",
            device=self.device,
        )
        # create renderer and set camera view
        self._create_viewport_render_product()
        # add flag for checking closing status
        self._is_closed = False

        # initialize common environment buffers
        self.reward_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        # allocate dictionary to store metrics
        self.extras = {}
        # create dictionary for storing last observations
        # note: Only used for the corner case of when in the UI, the stopped button is pressed. Then the
        #   physics handles become invalid. So it is not possible to call :meth:`_get_observations()`
        self._last_obs_buf: VecEnvObs = None

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
            replicate_physics=self.cfg.sim.replicate_physics,
        )
        # convert environment positions to torch tensor
        self.envs_positions = torch.tensor(self.envs_positions, dtype=torch.float, device=self.device)
        # filter collisions within each environment instance
        physics_scene_path = self.sim.get_physics_context().prim_path
        cloner.filter_collisions(
            physics_scene_path, "/World/collisions", prim_paths=self.envs_prim_paths, global_paths=global_prim_paths
        )

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
        self.sim.step(render=False)
        # compute common quantities
        self._cache_common_quantities()
        # compute observations
        self._last_obs_buf = self._get_observations()
        # return observations
        return self._last_obs_buf

    def step(self, actions: torch.Tensor) -> VecEnvStepReturn:
        """Reset any terminated environments and apply actions on the environment.

        This function deals with various timeline events (play, pause and stop) for clean execution.
        When the simulation is stopped all the physics handles expire and we cannot perform any read or
        write operations. The timeline event is only detected after every `sim.step()` call. Hence, at
        every call we need to check the status of the simulator. The logic is as follows:

        1. If the simulation is stopped, we complain about it and return the previous buffers.
        2. If the simulation is paused, we do not set any actions, but step the simulator.
        3. If the simulation is playing, we set the actions and step the simulator.

        Args:
            actions (torch.Tensor): Actions to apply on the simulator.

        Note:
            We perform resetting of the terminated environments before simulation
            stepping. This is because the physics buffers are not forwarded until
            the physics step occurs.

        Returns:
            VecEnvStepReturn: A tuple containing:
                - (VecEnvObs) observations from the environment
                - (torch.Tensor) reward from the environment
                - (torch.Tensor) whether the current episode is completed or not
                - (dict) misc information
        """
        # check if the simulation timeline is playing
        # if stopped, we complain about it and return the previous mdp buffers
        if self.sim.is_stopped():
            carb.log_warn("Simulation is stopped. Please exit the simulator...")
        # if paused, we do not set any actions into the simulator, but step
        elif not self.sim.is_playing():
            # step the simulator (but not the physics) to have UI still active
            self.sim.render()
            # check if the simulation timeline is stopped, do not update buffers
            if not self.sim.is_stopped():
                self._last_obs_buf = self._get_observations()
            else:
                carb.log_warn("Simulation is stopped. Please exit the simulator...")
        # if playing, we set the actions into the simulator and step
        else:
            # reset environments that terminated
            reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
            if len(reset_env_ids) > 0:
                self._reset_idx(reset_env_ids)
            # increment the number of steps
            self.episode_length_buf += 1
            # perform the stepping of simulation
            self._step_impl(actions)
            # check if the simulation timeline is stopped, do not update buffers
            if not self.sim.is_stopped():
                self._last_obs_buf = self._get_observations()
            else:
                carb.log_warn("Simulation is stopped. Please exit the simulator...")
        # return observations, rewards, resets and extras
        return self._last_obs_buf, self.reward_buf, self.reset_buf, self.extras

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
            # manually flush the flatcache data to update Hydra textures
            if self.sim.get_physics_context().use_flatcache:
                self._flatcache_iface.update(0.0, 0.0)
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
            # cleanup the scene and callbacks
            self.sim.clear_all_callbacks()
            self.sim.clear()
            # fix warnings at stage close
            omni.usd.get_context().get_stage().GetRootLayer().Clear()
            # update closing status
            self._is_closed = True

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
        """Apply actions on the environment and computes MDP signals.

        This function sets the simulation buffers for actions to apply, perform
        stepping of the simulation, and compute the MDP signals for reward and
        termination.

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

    def _configure_simulation_flags(self, sim_params: dict = None):
        """Configure simulation flags and extensions at load and run time."""
        # acquire settings interface
        carb_settings_iface = carb.settings.get_settings()
        # enable hydra scene-graph instancing
        # note: this allows rendering of instanceable assets on the GUI
        carb_settings_iface.set_bool("/persistent/omnihydra/useSceneGraphInstancing", True)
        # change dispatcher to use the default dispatcher in PhysX SDK instead of carb tasking
        # note: dispatcher handles how threads are launched for multi-threaded physics
        carb_settings_iface.set_bool("/physics/physxDispatcher", True)
        # disable contact processing in omni.physx if requested
        # note: helpful when creating contact reporting over limited number of objects in the scene
        if sim_params["disable_contact_processing"]:
            carb_settings_iface.set_bool("/physics/disableContactProcessing", True)

        # set flags based on whether rendering is enabled or not
        # note: enabling extensions is order-sensitive. please do not change the order.
        if self.enable_render or self.enable_viewport:
            # enable scene querying if rendering is enabled
            # this is needed for some GUI features
            sim_params["enable_scene_query_support"] = True
            # load extra viewport extensions if requested
            if self.enable_viewport:
                # extension to enable UI buttons (otherwise we get attribute errors)
                enable_extension("omni.kit.window.toolbar")
                # extension to make RTX realtime and path-traced renderers
                enable_extension("omni.kit.viewport.rtx")
                # extension to make HydraDelegate renderers
                enable_extension("omni.kit.viewport.pxr")
            # enable viewport extension if not running in headless mode
            enable_extension("omni.kit.viewport.bundle")
            # load extra render extensions if requested
            if self.enable_viewport:
                # extension for window status bar
                enable_extension("omni.kit.window.status_bar")
        # enable isaac replicator extension
        # note: moved here since it requires to have the viewport extension to be enabled first.
        enable_extension("omni.replicator.isaac")

    def _create_viewport_render_product(self):
        """Create a render product of the viewport for rendering."""
        # set camera view for "/OmniverseKit_Persp" camera
        set_camera_view(eye=self.cfg.viewer.eye, target=self.cfg.viewer.lookat)

        # check if flatcache is enabled
        # this is needed to flush the flatcache data into Hydra manually when calling `env.render()`
        # ref: https://docs.omniverse.nvidia.com/prod_extensions/prod_extensions/ext_physics.html
        if not self.enable_render and self.sim.get_physics_context().use_flatcache:
            from omni.physxflatcache import get_physx_flatcache_interface

            # acquire flatcache interface
            self._flatcache_iface = get_physx_flatcache_interface()

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
