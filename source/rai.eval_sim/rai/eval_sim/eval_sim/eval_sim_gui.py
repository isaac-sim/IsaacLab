# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import asyncio
import copy

import carb
import isaacsim.core.utils.stage as stage_utils
import omni
import omni.physx
import torch
from carb.events import IEvent
from isaaclab.envs import ManagerBasedRLEnv, ViewerCfg
from isaaclab.envs.ui import ViewportCameraController
from isaaclab.sim import SimulationContext
from omni import usd
from omni.timeline import TimelineEventType, get_timeline_interface
from omni.usd import StageEventType
from pxr import UsdUtils
from rai.eval_sim.eval_sim import EvalSim, EvalSimCfg
from rai.eval_sim.utils import random_actions, zero_actions
from rai.eval_sim.utils.video_recorder import VideoRecorder, VideoRecorderCfg


class EvalSimGUI(EvalSim):
    def __init__(self, cfg: EvalSimCfg) -> None:
        super().__init__(cfg)
        self.timeline = get_timeline_interface()
        self._physics_step_callbacks = []

        # event loop for async operations
        self._event_loop = asyncio.get_event_loop()

    """
    Core Functions
    """

    def step_sim(self):
        """Updates the simulation state over one step.
        This step (re)applies the most recently processed action.

        Note: If the robot's base is being fixed, then base velocities and pose are recorded before and reset after each step.
        """
        # save the root state of articulations before scene update
        if self._fix_robot_base:
            current_state = dict()
            for articulation in self.env.scene.articulations:
                current_state[articulation] = self.env.scene.articulations[articulation].data.root_state_w.clone()

        # update buffers
        self.env.scene.update(dt=self.physics_dt)

        # overwrite root pose with saved state and write zeros for velocity
        if self._fix_robot_base:
            for articulation in self.env.scene.articulations:
                self.env.scene.articulations[articulation].write_root_pose_to_sim(current_state[articulation][:, :7])
                self.env.scene.articulations[articulation].write_root_velocity_to_sim(
                    torch.zeros_like(current_state[articulation][:, 7:])
                )

    def step_controller(self):
        """Sends observations to and gets actions from the controller over one step.

        Returns:
            actions (torch.Tensor): Input actions to apply. Shape: (num_actions)

        Note:
        If ROS is not enabled, after computing observations:
         - If `self._random_actions is True`, then returns a random action.
         - If `self._random_actions is False`, then returns a zero action.
        """
        # compute observations.
        self.env.obs_buf = self.env.observation_manager.compute()
        self.env.recorder_manager.record_post_step()

        # get actions
        if self.ros_manager and self._ros_enabled:

            # no delay
            if self.cfg.control_delay == 0:
                self.ros_manager.publish(self.env.obs_buf)
                # spin, waiting for actions
                # we only exit this loop if we've received an action
                while True:
                    print("Waiting for actions...")
                    actions = self.ros_manager.subscribe()

                    if actions is not None:
                        print(f"Received actions: {actions}")
                        return actions

            # time delay
            elif self.cfg.control_delay > 0:
                self.time_delay_actions()
                self.ros_manager.publish(self.env.obs_buf)
                return None

        # Otherwise, Just return a random action.
        return random_actions(self.env) if self._random_actions else zero_actions(self.env)

    def step_deployment(self) -> None:
        """Updates the simulation for one step.

        Note: Actions from the controller are intermittently applied (determined by decimation)
        """
        # do not step if environment does not exist or the sim is stopped
        if self.env is None or self.env.sim.is_stopped():
            return

        self.step_sim()
        if self.env.sim.current_time_step_index % self.env_cfg.decimation == 0:
            # profiling at decimation rate
            self.log_wallclock_time()
            # step control
            actions = self.step_controller()
            if actions is not None:
                self.action_queue.append(actions)

            # pop first action in queue
            actions = self.action_queue.pop(0)
            self.env.action_manager.process_action(actions)
            self.env.recorder_manager.record_pre_step()

            # keep track of previous ros_enable state
            self._prev_ros_enable = self._ros_enabled
            self._prev_control_delay = self.cfg.control_delay

        # apply actions to scene
        self.env.action_manager.apply_action()
        self.env.scene.write_data_to_sim()

        # post-step: step interval event
        if "interval" in self.env.event_manager.available_modes:
            self.env.event_manager.apply(mode="interval", dt=self.physics_dt)

        # step video recorder
        self.video_recorder.step()

    """
    Configuration and Loading
    """

    async def load_env_async(self):
        """Loads the configuration and initializes the simulation context."""
        await stage_utils.create_new_stage_async()
        # note: this allows rendering of instanceable assets on the GUI
        carb_settings_iface = carb.settings.get_settings()
        carb_settings_iface.set_bool("/persistent/omnihydra/useSceneGraphInstancing", True)
        sim_context = SimulationContext(self.env_cfg.sim)

        # this method is intended to be used in the Isaac Sim's Extensions workflow where the Kit application has the
        # control over timing of physics and rendering steps
        await sim_context.initialize_simulation_context_async()

        # load the environment
        # extract and overwrite viewer to default viewer to handle errors that occur when viewer is referencing an asset that has not yet been initialized
        viewer_cfg = copy.deepcopy(self.env_cfg.viewer)
        self.env_cfg.viewer = ViewerCfg()
        self.load_env()

        # update the current USD stage asynchronously
        await stage_utils.update_stage_async()
        # reset the simulator
        # for the first time, we need to reset the simulator to initialize the scene and the agent.
        # after that, the buttons need to be used.
        # note: this plays the simulator which allows setting up all the physics handles.
        # NOTE: This line is the source of regex matching errors that occur when a different EnvCfg gets loaded for the
        # second time. There seems to be something fishy with clearing environments, somehow the previous
        # ArticulationCfg gets persisted to the second environment.
        # NOTE: The problem seems to be coming from an ISubscription function being set in for self._initialize_handle
        # in isaaclab.assets.asset_base.AssetBase. There is a weakref.proxy on this function and it seems
        # to be called even after ManagerBasedEnv.close(), which should take care of unsubscribing to this function.
        # When running in debug mode, this error does not appear
        await self.env.sim.reset_async(soft=False)

        # pause the simulator and wait for user to click play
        self.pause()

        # Initialize the camera viewport
        if self.env.sim.render_mode >= self.env.sim.RenderMode.PARTIAL_RENDERING:
            # set viewport back to desired setup now that assets are all guaranteed to be initialized
            self.env.viewport_camera_controller = ViewportCameraController(self.env, viewer_cfg)
        else:
            self.env.viewport_camera_controller = None

        # load managers
        self.env.load_managers()

        # only for RL tasks
        if isinstance(self.env, ManagerBasedRLEnv):
            self.env._configure_gym_env_spaces()
            self.env.event_manager.apply(mode="startup")

        # load the ros manager
        self.load_ros_manager()

        # initialize video recorder
        self.video_recorder = VideoRecorder(self.env.sim, VideoRecorderCfg())

        # reset timeline variables
        self.reset_timeline_variables()

        # step environment for visualizing initial position, rather than raw usd state
        await self.step_async()

        self.add_simulation_callbacks()

    def reload(self):
        """Reload EvalSim configs and the currently loaded environment."""
        self._event_loop.create_task(self.reload_async())

    async def reload_async(self):
        """Reload EvalSim configs and the environment."""
        self.pause()

        self.load_default_configs()

        self.set_env_cfg(self.cfg.env_cfg)

        # clear existing scene
        self.clear()

        await self.load_env_async()

        self.enable_ros()

        self.play()

    def add_simulation_callbacks(self):
        """Adds callbacks to the simulation.

        This method is intended to be used in the Isaac Sim's Extensions workflow where the Kit application has the
        control over timing of physics and rendering steps.
        """

        # add physics callback for stepping the agent and environment interaction
        self.env.sim.add_physics_callback("physics_callback", self._physics_step_callback)

        # add stage callback for clearing the stage on exit
        self.env.sim.add_stage_callback("clear_stage_on_exit", self._exit_stage_callback)

        # add timeline callback to reset buttons when the simulation is stopped
        self.env.sim.add_timeline_callback("set_buttons_state", self._on_timeline_state_change)

    def _physics_step_callback(self, dt: float):
        """Callback for stepping the simulation and updating the GUI.

        This callback is called at each physics step.
        """
        # update EvalSim
        self.step()

        # NOTE: This is a workaround due to not being able to add multiple physics callbacks via
        # isaaclab.sim.Simulation.add_physics_callback. So in EvalSimGUIExtension, we add a callback to the
        # simulation to update the GUI at each physics step.
        for callback in self._physics_step_callbacks:
            callback(dt)

    def _exit_stage_callback(self, e: IEvent) -> None:
        """Callback when the stage is exited."""
        if e.type == int(StageEventType.CLOSED):
            self.stop()

    def _on_timeline_state_change(self, e) -> None:
        """Callback when the timeline event changes (e.g. play/pause/stop buttons are pressed)."""
        if (e.type == int(TimelineEventType.STOP)) or (e.type == int(TimelineEventType.PLAY)):
            # reset time buffers as they will be out of sync
            self.wallclock_dt_buffer.clear()
            self.log_wallclock_time()

    def add_physics_step_callback(self, callback):
        """Add a callback to be called at each physics step.

        Callbacks are called in the order that they are added to the list via this method.
        """
        self._physics_step_callbacks.append(callback)

    """
    Timeline
    """

    async def step_async(self, nr_steps: int = 1):
        """Steps the simulation for a given number of steps.

        Args:
            nr_steps (int): Number of steps to take.
        """

        # Get simulation / fabric interfaces and add the stage_id
        stage_id = UsdUtils.StageCache.Get().Insert(usd.get_context().get_stage()).ToLongInt()
        iphysx_sim = omni.physx.get_physx_simulation_interface()
        iphysx_fc = omni.physxfabric.get_physx_fabric_interface()
        iphysx_fc.attach_stage(stage_id)

        # step environment
        if self.env is not None:
            for _ in range(nr_steps):
                # pause simulation
                if self.timeline.is_playing():
                    self.timeline.pause()

                # Asynchronous step function from omni.physxui.physxDebugView (stripped to include relevant parts).
                self.timeline.set_current_time(self.timeline.get_current_time() + self.env.physics_dt)

                # Update fabric
                iphysx_sim.simulate(self.env.physics_dt, 0.0)
                iphysx_sim.fetch_results()
                iphysx_fc.update(self.env.physics_dt, 0.0)

                # render simulation (required for visualizing multiple steps)
                await self.env.sim.render_async()

        # reset wallclock
        self.wallclock_dt_buffer.clear()
        self.log_wallclock_time()

    async def reset_async(self):
        """Resets the environment and pauses the simulation.

        Also, takes one simulation step so the reset state is visualized.
        """
        # reset environment
        self.reset()

        # pause
        self.pause()

        # take one sim step to visualize reset
        await self.step_async()
