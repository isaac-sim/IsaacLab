# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# general modules
import asyncio
from functools import partial

import isaacsim.core.utils.stage as stage_utils

# IsaacLab modules
import isaacsim.gui.components.ui_utils as ui_utils
import launch
import launch_ros

# Omni modules
import omni.ext
import omni.kit.widget.toolbar
import omni.ui as ui
from isaacsim.gui.components.element_wrappers import StateButton
from omni.kit.window.property.templates import LABEL_WIDTH
from omni.ui import (
    Alignment,
    DockPosition,
    Frame,
    HStack,
    IntDrag,
    Label,
    ScrollBarPolicy,
    ScrollingFrame,
    VStack,
    Window,
    Workspace,
)

# EvalSim modules
from rai.eval_sim.eval_sim import EvalSimGUI
from rai.eval_sim.utils import (
    GAMEPAD_CFG_PATH,
    launch_a_launch_file_async,
    log_error,
    log_info,
    log_warn,
)
from ros2launch.api import get_share_file_path_from_package

# gui related
from .eval_sim_gui_extension_cfg import EvalSimGUIExtensionCfg
from .utils import ESButton, ESFloatField, ESIntField, create_collapsable_frame

ROS2_ENABLED_BUTTON_TEXT = "ENABLED"
ROS2_DISABLED_BUTTON_TEXT = "DISABLED"


class EvalSimGUIExtension(omni.ext.IExt):
    """GUI extension for EvalSim."""

    def __init__(self) -> None:
        """Initialize the extension.

        NOTE: on_startup gets called asynchronously immediately after the extension is loaded, so we don't
        do anything here.
        """
        super().__init__()

    """
    Extension callbacks.
    """

    def on_startup(self, ext_id: str) -> None:
        """Callback when extension is loaded."""
        # initialize EvalSim
        self.cfg = EvalSimGUIExtensionCfg()
        self.eval_sim = EvalSimGUI(self.cfg.eval_sim)

        self.eval_sim.add_physics_step_callback(self._update_gui)

        # initialize runtime variables
        self._input_device_type = None
        self._teleop_config_path = None
        self._launch_service = None

        # log
        log_info(f"Starting {self.extension_name}...")
        # load the UI
        self._window: Window | None = None
        self._ui_buttons: dict[str, ESButton] = dict()
        self._ui_state_buttons: dict[str, StateButton] = dict()
        self._ui_dropdowns: dict[str, object] = dict()
        self._ui_frames: dict[str, Frame] = dict()
        self._ui_labels: dict[str, Label] = dict()
        self._ui_sliders: dict[str, ESFloatField | ESIntField | IntDrag] = dict()

        # load at startup if specified in config
        if self.eval_sim.cfg.auto_load_env:
            self._on_load()

        # create UI window
        self._build_ui()

        # hide the stop button from Isaac Sim GUI
        self._hide_stop_button()

    def on_shutdown(self) -> None:
        """Callback when extension is unloaded."""
        # save user settings
        # disable window
        self._window = None
        # destroy UI
        self._destroy_ui()
        # close EvalSim
        self.eval_sim.close()
        # log
        log_info(f"Closing {self.extension_name}...")

    @property
    def extension_name(self) -> str:
        """The extension class name"""
        return type(self).__name__

    """
    UI callbacks.
    """

    def _on_clear(self):
        print("Disabling scene buttons")
        self._disable_scene_buttons()
        self.eval_sim.clear()
        print("Enabling scene buttons")
        self._enable_scene_buttons()
        self._update_gui()

    def _on_set_env_cfg(self, value: str):
        """Called when the environment configuration is selected from dropdown."""
        if value == "":
            self.eval_sim.env_cfg = None
        else:
            parsed_value = value.split(" (")[1].split(")")[0]
            self.eval_sim.set_env_cfg(parsed_value)

    def _on_set_ros_manager_cfg(self, value: str):
        """Called when the ROS manager configuration is selected from dropdown."""
        if value == "":
            self.eval_sim.ros_manager_cfg = None
            self.eval_sim.disable_ros()
            self._ui_state_buttons["ros"].reset()
            self._ui_state_buttons["ros"].enabled = False
        else:
            parsed_value = value.split(" (")[1].split(")")[0]
            self.eval_sim.set_ros_manager_cfg(parsed_value)
            self._ui_state_buttons["ros"].enabled = True

    def _on_load(self) -> None:
        async def _on_load_async():
            # clear existing scene
            self.eval_sim.clear()

            await self.eval_sim.load_env_async()

            # need to set ROS button's state
            if self.eval_sim.ros_enabled:
                self._ui_state_buttons["ros"].state_button.text = ROS2_ENABLED_BUTTON_TEXT
            else:
                self._ui_state_buttons["ros"].state_button.text = ROS2_DISABLED_BUTTON_TEXT

            # reset
            self._on_reset()

        async def _on_load_async_gui():
            if self.eval_sim.env_cfg is None:
                log_error("No environment configuration selected. Please select a valid environment configuration.")
                return

            self._disable_scene_buttons()
            await _on_load_async()
            self._enable_scene_buttons()
            self._update_gui()
            self._hide_stop_button()

            # update simulation rate controls
            self._ui_sliders["physics_dt"].set_value(self.default_physics_dt)
            self._ui_sliders["render_substeps"].model.set_value(self.default_render_substeps)
            self._ui_sliders["control_substeps"].set_value(self.default_control_substeps)

        asyncio.ensure_future(_on_load_async_gui())

    def _on_reset(self) -> None:
        """This function resets the simulator by stopping it. After that, all the handles for the scene
        and the agent are reinitialized and reset.
        """
        asyncio.ensure_future(self.eval_sim.reset_async())

        # Need to add these back in as eval_sim.reset() will re-initialize the config dictionary
        self._update_gui()

    def _on_step(self) -> None:
        nr_steps = self._ui_sliders["step"].model.as_int
        asyncio.ensure_future(self.eval_sim.step_async(nr_steps))
        self._update_gui()

    def _on_teleop_input_dropdown(self, value: str) -> None:
        """Callback when the teleop input device dropdown is selected."""
        self._input_device_type = value

    def _on_teleop_start_button(self) -> None:
        async def _start_joystick_node_async():
            match self._input_device_type:
                case "Gamepad":
                    # load config from default path
                    self._teleop_config_path = str(GAMEPAD_CFG_PATH)
                    # load default file path
                    path = get_share_file_path_from_package(
                        package_name="teleop_twist_joy", file_name="teleop-launch.py"
                    )
                    await launch_a_launch_file_async(
                        launch_file_path=path,
                        launch_file_arguments=[f"config_filepath:={self._teleop_config_path}"],
                        debug=False,
                    )
                case "Keyboard":
                    # TODO: parse keyboard.yaml for configuration options to send to launch_ros(...)
                    self._teleop_config_path = str(GAMEPAD_CFG_PATH)
                    self._launch_service = launch.LaunchService()
                    # xterm is required to launch a new terminal window for capturing and printing keyboard inputs.
                    node = launch_ros.actions.Node(
                        executable="teleop_twist_keyboard",
                        package="teleop_twist_keyboard",
                        exec_name="teleop_twist_keyboard",
                        output="screen",
                        prefix="xterm -e",
                    )
                    self._launch_service.include_launch_description(node)
                    await self._launch_service.run_async()

                case _:
                    raise RuntimeError(
                        f'Invalid device type {self._input_device_type}. Valid options are "Gamepad" and "Keyboard".'
                    )

        asyncio.ensure_future(_start_joystick_node_async())

    def _on_set_physics_dt(self, dt: float):
        self.eval_sim.set_physics_dt(dt / 1000.0)
        self._update_gui()

    def _on_set_render_substeps(self, value: ui.SimpleIntModel) -> None:
        self.eval_sim.env_cfg.sim.render_interval = value.as_int
        self._update_hz_labels()

    def _on_set_control_substeps(self, value: int) -> None:
        self.eval_sim.env_cfg.decimation = value  # NOTE: control substeps is called decimation in Isaac Lab
        self._update_hz_labels()

    def _on_set_control_delay(self, value: int) -> None:
        self.eval_sim.cfg.control_delay = value
        self.eval_sim.apply_delay_to_action_queue()

    """
    UI frames.
    """

    def _build_ui(self) -> None:
        # create window
        self._window = Window(
            title=self.cfg.window.title,
            width=self.cfg.window.width,
            height=self.cfg.window.height,
            visible=self.cfg.window.visible,
        )

        # dock window
        asyncio.ensure_future(self._dock_window())

        # stack main frames
        with self._window.frame:
            with VStack(spacing=5, height=0):
                self._ui_frames["scene_setup"] = create_collapsable_frame("Scene Setup")
                self._ui_frames["teleop_controls"] = create_collapsable_frame("Teleop Controls")
                self._ui_frames["simulation_controls"] = create_collapsable_frame("Simulation Controls")

        # build frames
        self._build_scene_setup_frame(self._ui_frames["scene_setup"])
        self._build_teleop_controls_frame(self._ui_frames["teleop_controls"])
        self._build_simulation_controls_frame(self._ui_frames["simulation_controls"])

        # update user interface
        self._update_gui()

    def _build_scene_setup_frame(self, frame: Frame):
        # get user settings environment configuration
        env_cfg_default = self.eval_sim.cfg.env_cfg
        ros_manager_cfg_default = self.eval_sim.cfg.ros_manager_cfg

        print(f"env_cfg_default: {env_cfg_default}")
        print(f"ros_manager_cfg_default: {ros_manager_cfg_default}")

        with frame:
            with VStack(style=ui_utils.get_style(), spacing=5, height=0):
                # button: clear
                self._ui_buttons["clear"] = ESButton(
                    label="Clear Scene",
                    text="Clear",
                    tooltip="Clear the robot, terrain and controller.",
                    on_click_fn=self._on_clear,
                )

                # dropdown: envs
                if env_cfg_default == "" or env_cfg_default not in self.eval_sim.env_configs:
                    default_env_cfg_index = 0
                    if env_cfg_default not in self.eval_sim.env_configs:
                        log_warn(
                            f"Default environment configuration {env_cfg_default} not found."
                            " Please select a valid environment configuration."
                        )

                else:
                    # need to add 1 to index to account for the blank space
                    default_env_cfg_index = list(self.eval_sim.env_configs.keys()).index(env_cfg_default) + 1

                # we add a blank space to the list of env_cfgs so that the user can select no env_cfg
                # we want items to be displayed like:
                # AnymalDEnvCfg (rai.eval_sim.tasks.anymal.anymal_env_cfg.AnymalDEnvCfg)
                env_cfg_items = [""] + [f"{key.split('.')[-1]} ({key})" for key in self.eval_sim.env_configs.keys()]

                self._ui_dropdowns["env_cfgs"] = ui_utils.dropdown_builder(
                    label="EnvCfg(s)",
                    tooltip="Available premade environment config(s)",
                    default_val=default_env_cfg_index,
                    items=env_cfg_items,
                    on_clicked_fn=self._on_set_env_cfg,
                )

                # dropdown: ros manager
                if ros_manager_cfg_default == "" or ros_manager_cfg_default not in self.eval_sim.ros_manager_configs:
                    default_ros_manager_cfg_index = 0
                    if ros_manager_cfg_default not in self.eval_sim.ros_manager_configs:
                        log_warn(
                            f"Default ros manager configuration {ros_manager_cfg_default} not found."
                            " Please select a valid ROS manager configuration."
                        )
                else:
                    # need to add 1 to index to account for the blank space
                    default_ros_manager_cfg_index = (
                        list(self.eval_sim.ros_manager_configs.keys()).index(ros_manager_cfg_default) + 1
                    )

                # we add a blank space to the list of ros_manager_cfgs so that the user can select no ros_manager_cfg
                # we want items to be displayed like:
                # AnymalDRosManagerCfg (rai.eval_sim.tasks.anymal.anymal_ros_manager_cfg.AnymalDRosManagerCfg)
                ros_manager_cfg_items = [""] + [
                    f"{key.split('.')[-1]} ({key})" for key in self.eval_sim.ros_manager_configs.keys()
                ]

                self._ui_dropdowns["ros_manager_cfgs"] = ui_utils.dropdown_builder(
                    label="RosManagerCfg(s)",
                    tooltip="Available premade ros manager config(s)",
                    default_val=default_ros_manager_cfg_index,
                    items=ros_manager_cfg_items,
                    on_clicked_fn=self._on_set_ros_manager_cfg,
                )

                # button: load
                self._ui_buttons["load"] = ESButton(
                    label="Load Configs",
                    text="Load",
                    tooltip="Loads the selected EnvCfg and RosManagerCfg.",
                    on_click_fn=self._on_load,
                )

                # button: reset
                self._ui_buttons["reset"] = ESButton(
                    label="Reset Scene",
                    text="Reset",
                    tooltip="Reset with stopping the simulator.",
                    on_click_fn=self._on_reset,
                )

                with HStack():
                    # button: step
                    self._ui_buttons["step"] = ESButton(
                        label="Step",
                        text="Step",
                        tooltip="Apply nr of steps in simulation.",
                        on_click_fn=self._on_step,
                        use_line_rect_flourish=False,
                    )
                    ui.Spacer(width=5)

                    # slider: step
                    self._ui_sliders["step"] = IntDrag(
                        name="Field", height=ui_utils.LABEL_HEIGHT, min=1, max=100, alignment=ui.Alignment.LEFT_CENTER
                    )
                    self._ui_sliders["step"].model.set_value(1)
                    ui_utils.add_line_rect_flourish(False)

                # state-button: record
                self._ui_state_buttons["record"] = StateButton(
                    label="Video Recording",
                    a_text="Record",
                    b_text="Stop",
                    tooltip="Toggle simulation recording",
                    on_a_click_fn=self.eval_sim.start_recording,
                    on_b_click_fn=self.eval_sim.stop_recording,
                )

                # state-button: ros
                self._ui_state_buttons["ros"] = StateButton(
                    label="ROS",
                    a_text=ROS2_ENABLED_BUTTON_TEXT if self.eval_sim.ros_enabled else ROS2_DISABLED_BUTTON_TEXT,
                    b_text=ROS2_DISABLED_BUTTON_TEXT if self.eval_sim.ros_enabled else ROS2_ENABLED_BUTTON_TEXT,
                    tooltip="Toggle ROS on / off. The RosManagerCfg must be set to enable ROS.",
                    on_a_click_fn=self.eval_sim.disable_ros if self.eval_sim.ros_enabled else self.eval_sim.enable_ros,
                    on_b_click_fn=self.eval_sim.enable_ros if self.eval_sim.ros_enabled else self.eval_sim.disable_ros,
                )

                # state-button: default actions
                self._ui_state_buttons["default_actions"] = StateButton(
                    label="Default Actions",
                    a_text="Zeros",
                    b_text="Randomized",
                    tooltip="If ROS is disabled, the default actions will be set. Choose between zeros or randomized.",
                    on_a_click_fn=partial(self.eval_sim.set_random_actions, True),
                    on_b_click_fn=partial(self.eval_sim.set_random_actions, False),
                )

                # state-button: debug visualization
                self._ui_state_buttons["debug"] = StateButton(
                    label="Toggle Debug Visualization",
                    a_text="Disabled",
                    b_text="Enabled",
                    tooltip="Toggle debug visualization of sensors.",
                    on_a_click_fn=partial(self.eval_sim.set_debug_vis, True),
                    on_b_click_fn=partial(self.eval_sim.set_debug_vis, False),
                )

                # state-button: debug visualization
                self._ui_state_buttons["fix_base"] = StateButton(
                    label="Fix robot base",
                    a_text="Floating",
                    b_text="Fixed",
                    tooltip="Fix robot base.",
                    on_a_click_fn=partial(self.eval_sim.set_fix_robot_base, True),
                    on_b_click_fn=partial(self.eval_sim.set_fix_robot_base, False),
                )

    def _build_teleop_controls_frame(self, frame: Frame):
        with frame:
            with VStack(style=ui_utils.get_style(), spacing=5, height=0):
                # dropdown: select teleop peripheral
                self._ui_dropdowns["select_teleop"] = ui_utils.dropdown_builder(
                    label="Teleop Input Device Type",
                    default_val=0,
                    items=["", "Gamepad", "Keyboard"],
                    tooltip="Dropdown tooltip",
                    on_clicked_fn=self._on_teleop_input_dropdown,
                )

                # button: start teleop
                self._ui_buttons["start_teleop"] = ESButton(
                    label="Start Teleop Input Listener",
                    text="Start",
                    tooltip="Launch the teleop ROS Node.",
                    on_click_fn=self._on_teleop_start_button,
                )

    def _build_simulation_controls_frame(self, frame: Frame):
        with frame:
            with VStack(style=ui_utils.get_style(), spacing=5, height=0):
                # label: time per step
                self._ui_labels["time_per_step"] = Label(
                    "Average Compute Time / Step: ",
                    width=LABEL_WIDTH,
                    alignment=Alignment.LEFT_CENTER,
                    tooltip=(
                        "This label indicates the average real-world time taken to calculate a single step "
                        "in the simulation. It reflects the computational efficiency and how long, on average, "
                        "the system spends processing each simulation increment."
                    ),
                )

                # label: simulation speed
                self._ui_labels["simulation_speed"] = Label(
                    "Simulation Speed (% Real Time): ",
                    width=LABEL_WIDTH,
                    alignment=Alignment.LEFT_CENTER,
                    tooltip=(
                        "This label shows how fast the simulation runs compared to actual time. A value of"
                        "100% means the simulation is running in real-time; values over 100% indicate the "
                        "simulation is running faster than real life, while values under 100% suggest the "
                        "simulation is slower, with 200% being twice as fast as reality and 20% five times slower."
                    ),
                )

                # label: simulation time
                self._ui_labels["simulation_time"] = Label(
                    "Simulation Time [s]: ",
                    width=LABEL_WIDTH,
                    alignment=Alignment.LEFT_CENTER,
                    tooltip="This label shows the current simulation time. A value of",
                )

                with HStack():
                    # float field: physics dt
                    self._ui_sliders["physics_dt"] = ESFloatField(
                        label="Physics dt [ms]",
                        tooltip=(
                            "Set the physics dt. We advise pausing when changing the physics_dt."
                            " [Environment must be loaded]"
                        ),
                        default_value=self.default_physics_dt,
                        step=0.1,
                        format="%.2f",
                        lower_limit=1.0,
                        upper_limit=50.0,
                        on_value_changed_fn=self._on_set_physics_dt,
                        mouse_double_clicked_fn=self._pause,  # pause when changing the dt through keyboard interface
                    )
                    self._ui_labels["physics_hz"] = Label(
                        "Hz",
                        width=50,
                        alignment=Alignment.RIGHT,
                        tooltip="This label shows the sim physics rate in Hz",
                    )

                with HStack():
                    # button: render substeps
                    self._ui_buttons["render_substeps"] = ESButton(
                        label="Render Substeps",
                        text="Load",
                        tooltip="Set the render substeps n. For every n physics steps there is one rendering update.",
                        on_click_fn=self._on_load,
                        use_line_rect_flourish=False,
                    )
                    ui.Spacer(width=5)

                    # slider: render substeps
                    self._ui_sliders["render_substeps"] = IntDrag(
                        name="Field", height=ui_utils.LABEL_HEIGHT, min=1, max=100, alignment=ui.Alignment.LEFT_CENTER
                    )
                    self._ui_sliders["render_substeps"].model.set_value(self.default_render_substeps)
                    self._ui_sliders["render_substeps"].model.add_value_changed_fn(self._on_set_render_substeps)
                    ui_utils.add_line_rect_flourish(False)

                    self._ui_labels["render_hz"] = Label(
                        "Hz",
                        width=50,
                        alignment=Alignment.RIGHT,
                        tooltip="This label shows the sim render rate in Hz",
                    )

                with HStack():
                    # int field: control substeps
                    self._ui_sliders["control_substeps"] = ESIntField(
                        label="Control Substeps",
                        tooltip=(
                            "Set the control substeps n. For every n physics steps there is one control loop "
                            "executed. [Environment must be loaded]"
                        ),
                        default_value=self.default_control_substeps,
                        lower_limit=1,
                        upper_limit=50,
                        on_value_changed_fn=self._on_set_control_substeps,
                    )

                    self._ui_labels["control_hz"] = Label(
                        "Hz",
                        width=50,
                        alignment=Alignment.RIGHT,
                        tooltip="This label shows the sim control rate in Hz",
                    )

                # int field: control substeps
                self._ui_sliders["control_delay"] = ESIntField(
                    label="Control Delay",
                    tooltip=(
                        "Set the number of steps to delay applying the control inpputs to the simulation. \n           "
                        "              Delay of d will result in commands at control step i to be applied at \n        "
                        "                 simulation step i*n+d, where n is the number of control substeps. For \n     "
                        "                    example (d=0,n=1) results in commands appled at physics step i. If \n     "
                        "                    (d=1,n=1) commands are applied at physics step i+1."
                    ),
                    default_value=self.eval_sim.cfg.control_delay,
                    lower_limit=0,
                    upper_limit=100,
                    on_value_changed_fn=self._on_set_control_delay,
                )

    def _hide_stop_button(self):
        """Hide the stop button in the toolbar."""

        # grey out the stop button because we don't want to stop the simulation manually in standalone mode
        toolbar = omni.kit.widget.toolbar.get_instance()
        play_button_group = toolbar._builtin_tools._play_button_group  # type: ignore
        if play_button_group is not None and play_button_group._stop_button is not None:
            play_button_group._stop_button.visible = False  # type: ignore
            play_button_group._stop_button.enabled = False  # type: ignore
            play_button_group._stop_button = None  # type: ignore

    """
    GUI updating.
    """

    @property
    def default_physics_dt(self):
        """In case no env_cfg is loaded, we provide a default physics_dt value for the GUI."""
        return 1000 * self.eval_sim.env_cfg.sim.dt if self.eval_sim.env_cfg is not None else 2.0

    @property
    def default_render_substeps(self):
        """In case no env_cfg is loaded, we provide a default render_substep value for the GUI."""
        return self.eval_sim.env_cfg.sim.render_interval if self.eval_sim.env_cfg is not None else 16

    @property
    def default_control_substeps(self):
        """In case no env_cfg is loaded, we provide a default control_substep value for the GUI."""
        return self.eval_sim.env_cfg.decimation if self.eval_sim.env_cfg is not None else 1

    """
    Other.
    """

    def _pause(self, *args, **kwargs):
        self.eval_sim.pause()

    def _update_gui(self, dt=None):
        """Update the GUI for any changes in the simulation.

        Args:
            dt: The time delta. This argument is unused, but is needed as we add this function as a callback to the
                simulation's physics step so that the GUI is updated every physics step.
        """
        # update simulation controls frame
        time_per_step, time_per_step_hz, sim_speed = self.eval_sim.get_simulation_time_profile()
        self._ui_labels["time_per_step"].text = f"Average Compute Time / Step: {time_per_step} {time_per_step_hz}"
        self._ui_labels["simulation_speed"].text = f"Simulation Speed (% Real Time): {sim_speed}"

        env_set = self.eval_sim.env is not None

        # -- enable / disable buttons
        # loaded / unloaded environment
        self._ui_sliders["physics_dt"].enabled = env_set
        self._ui_sliders["render_substeps"].enabled = env_set
        self._ui_sliders["control_substeps"].enabled = env_set

        if env_set:
            self._ui_labels["simulation_time"].text = f"Simulation Time [sec]: {self.eval_sim.env.sim.current_time:.3f}"
            self._ui_sliders["physics_dt"].float_field.model.set_value(self.eval_sim.physics_dt * 1000)
            self._ui_sliders["render_substeps"].model.set_value(self.eval_sim.env_cfg.sim.render_interval)
            self._ui_sliders["control_substeps"].int_field.model.set_value(self.eval_sim.env_cfg.decimation)
            self._update_hz_labels()

    def _update_hz_labels(self):
        self._ui_labels["physics_hz"].text = f"({int(1.0 / self.eval_sim.physics_dt)} Hz)"
        self._ui_labels["render_hz"].text = (
            f"({int(1.0 / (self.eval_sim.physics_dt * self.eval_sim.env_cfg.sim.render_interval))} Hz)"
        )
        self._ui_labels["control_hz"].text = (
            f"({int(1.0 / (self.eval_sim.physics_dt * self.eval_sim.env_cfg.decimation))} Hz)"
        )

    def _destroy_ui(self) -> None:
        """Destroy the UI.

        This function is a complement to the `_build_ui` function. It destroys the UI and resets the buttons when
        the extension is unloaded.
        """
        # disable all buttons
        if self._ui_buttons:
            for button in self._ui_buttons.values():
                button.enabled = False

        # hack: set dummy functions for ESFloatFields, otherwise the functions persist and hold a reference to this class
        for slider in self._ui_sliders.values():
            slider.set_on_value_changed_fn(lambda x: x)

    async def _dock_window(self) -> None:
        """Dock window to the left of viewport"""
        # load any remaining stuff
        await stage_utils.update_stage_async()
        # find viewport window
        target = Workspace.get_window("Viewport")
        # dock to left of viewport
        if self._window and target:
            self._window.dock_in(window=target, dock_position=DockPosition.LEFT, ratio=0.33)
        # update stage with settings
        await stage_utils.update_stage_async()

    def _add_info_text(self, frame, title: str, msg: str, height: int = 0) -> None:
        """Add text to the info frame in the UI.

        This is partially taken from `omni.isaac.ui.utils.build_info_text` function but modified to add
        the text to a specific frame.
        """
        # this clears out everything else in that frame.
        with frame:
            default_val = msg
            # create stack with text
            with VStack(style=ui_utils.get_style(), spacing=5):
                Label(
                    title,
                    style_type_name_override="Label::label",
                    word_wrap=True,
                    alignment=Alignment.LEFT_TOP,
                )
                with ScrollingFrame(
                    height=height,
                    style_type_name_override="ScrollingFrame",
                    alignment=Alignment.LEFT_TOP,
                    horizontal_scrollbar_policy=ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
                    vertical_scrollbar_policy=ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
                ):
                    Label(
                        default_val,
                        style_type_name_override="Label::label",
                        word_wrap=False,
                        alignment=Alignment.LEFT_TOP,
                    )

    def _disable_scene_buttons(self):
        self._ui_buttons["clear"].enabled = False
        self._ui_buttons["load"].enabled = False
        self._ui_buttons["reset"].enabled = False
        self._ui_buttons["step"].enabled = False

    def _enable_scene_buttons(self):
        self._ui_buttons["clear"].enabled = True
        self._ui_buttons["load"].enabled = True
        self._ui_buttons["reset"].enabled = True
        self._ui_buttons["step"].enabled = True
