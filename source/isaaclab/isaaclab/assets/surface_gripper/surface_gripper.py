# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import warnings
from typing import TYPE_CHECKING

from isaacsim.core.utils.extensions import enable_extension
from isaacsim.core.version import get_version

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBase

if TYPE_CHECKING:
    from isaacsim.robot.surface_gripper import GripperView

from .surface_gripper_cfg import SurfaceGripperCfg


class SurfaceGripper(AssetBase):
    """A surface gripper actuator class.

    Surface grippers are actuators capable of grasping objects when in close proximity with them.

    Each surface gripper in the collection must be a `Isaac Sim SurfaceGripper` primitive.
    On playing the simulation, the physics engine will automatically register the surface grippers into a
    SurfaceGripperView object. This object can be accessed using the :attr:`gripper_view` attribute.

    To interact with the surface grippers, the user can use the :attr:`state` to get the current state of the grippers,
    :attr:`command` to get the current command sent to the grippers, and :func:`update_gripper_properties` to update the
    properties of the grippers at runtime. Finally, the :func:`set_grippers_command` function should be used to set the
    desired command for the grippers.

    Note:
        The :func:`set_grippers_command` function does not write to the simulation. The simulation automatically
         calls :func:`write_data_to_sim` function to write the command to the simulation. Similarly, the update
         function is called automatically for every simulation step, and does not need to be called by the user.

    Note:
        The SurfaceGripper is only supported on CPU for now. Please set the simulation backend to run on CPU.
        Use `--device cpu` to run the simulation on CPU.
    """

    def __init__(self, cfg: SurfaceGripperCfg):
        """Initialize the surface gripper.

        Args:
            cfg: A configuration instance.
        """
        # copy the configuration
        self._cfg = cfg.copy()

        isaac_sim_version = get_version()
        # checks for Isaac Sim v5.0 to ensure that the surface gripper is supported
        if int(isaac_sim_version[2]) < 5:
            raise Exception(
                "SurfaceGrippers are only supported by IsaacSim 5.0 and newer. Use IsaacSim 5.0 or newer to use this"
                " feature."
            )

        # flag for whether the sensor is initialized
        self._is_initialized = False
        self._debug_vis_handle = None

        # register various callback functions
        self._register_callbacks()

    """
    Properties
    """

    @property
    def data(self):
        raise NotImplementedError("SurfaceGripper does have a data interface.")

    @property
    def num_instances(self) -> int:
        """Number of instances of the gripper.

        This is equal to the total number of grippers (the view can only contain one gripper per environment).
        """
        return self._num_envs

    @property
    def state(self) -> torch.Tensor:
        """Returns the gripper state buffer.

        The gripper state is a list of integers:
        - -1 --> Open
        - 0 --> Closing
        - 1 --> Closed
        """
        return self._gripper_state

    @property
    def command(self) -> torch.Tensor:
        """Returns the gripper command buffer.

        The gripper command is a list of floats:
        - [-1, -0.3] --> Open
        - [-0.3, 0.3] --> Do nothing
        - [0.3, 1] --> Close
        """
        return self._gripper_command

    @property
    def gripper_view(self) -> GripperView:
        """Returns the gripper view object."""
        return self._gripper_view

    """
    Operations
    """

    def update_gripper_properties(
        self,
        max_grip_distance: torch.Tensor | None = None,
        coaxial_force_limit: torch.Tensor | None = None,
        shear_force_limit: torch.Tensor | None = None,
        retry_interval: torch.Tensor | None = None,
        indices: torch.Tensor | None = None,
    ) -> None:
        """Update the gripper properties.

        Args:
            max_grip_distance: The maximum grip distance of the gripper. Should be a tensor of shape (num_envs,).
            coaxial_force_limit: The coaxial force limit of the gripper. Should be a tensor of shape (num_envs,).
            shear_force_limit: The shear force limit of the gripper. Should be a tensor of shape (num_envs,).
            retry_interval: The retry interval of the gripper. Should be a tensor of shape (num_envs,).
            indices: The indices of the grippers to update the properties for. Can be a tensor of any shape.
        """

        if indices is None:
            indices = self._ALL_INDICES

        indices_as_list = indices.tolist()

        if max_grip_distance is not None:
            self._max_grip_distance[indices] = max_grip_distance
        if coaxial_force_limit is not None:
            self._coaxial_force_limit[indices] = coaxial_force_limit
        if shear_force_limit is not None:
            self._shear_force_limit[indices] = shear_force_limit
        if retry_interval is not None:
            self._retry_interval[indices] = retry_interval

        self._gripper_view.set_surface_gripper_properties(
            max_grip_distance=self._max_grip_distance.tolist(),
            coaxial_force_limit=self._coaxial_force_limit.tolist(),
            shear_force_limit=self._shear_force_limit.tolist(),
            retry_interval=self._retry_interval.tolist(),
            indices=indices_as_list,
        )

    def update(self, dt: float) -> None:
        """Update the gripper state using the SurfaceGripperView.

        This function is called every simulation step.
        The data fetched from the gripper view is a list of strings containing 3 possible states:
            - "Open"
            - "Closing"
            - "Closed"

        To make this more neural network friendly, we convert the list of strings to a list of floats:
            - "Open" --> -1.0
            - "Closing" --> 0.0
            - "Closed" --> 1.0

        Note:
            We need to do this conversion for every single step of the simulation because the gripper can lose contact
            with the object if some conditions are met: such as if a large force is applied to the gripped object.
        """
        state_list: list[str] = self._gripper_view.get_surface_gripper_status()
        state_list_as_int: list[float] = [
            -1.0 if state == "Open" else 1.0 if state == "Closed" else 0.0 for state in state_list
        ]
        self._gripper_state = torch.tensor(state_list_as_int, dtype=torch.float32, device=self._device)

    def write_data_to_sim(self) -> None:
        """Write the gripper command to the SurfaceGripperView.

        The gripper command is a list of integers that needs to be converted to a list of strings:
            - [-1, -0.3] --> Open
            - ]-0.3, 0.3[ --> Do nothing
            - [0.3, 1] --> Closed

        The Do nothing command is not applied, and is only used to indicate whether the gripper state has changed.
        """
        # Remove the SurfaceGripper indices that have a commanded value of 2
        indices = (
            torch.argwhere(torch.logical_or(self._gripper_command < -0.3, self._gripper_command > 0.3))
            .to(torch.int32)
            .tolist()
        )
        # Write to the SurfaceGripperView if there are any indices to write to
        if len(indices) > 0:
            self._gripper_view.apply_gripper_action(self._gripper_command.tolist(), indices)

    def set_grippers_command(self, states: torch.Tensor, indices: torch.Tensor | None = None) -> None:
        """Set the internal gripper command buffer. This function does not write to the simulation.

        Possible values for the gripper command are:
            - [-1, -0.3] --> Open
            - ]-0.3, 0.3[ --> Do nothing
            - [0.3, 1] --> Close

        Args:
            states: A tensor of integers representing the gripper command. Shape must match that of indices.
            indices: A tensor of integers representing the indices of the grippers to set the command for. Defaults
                     to None, in which case all grippers are set.
        """
        if indices is None:
            indices = self._ALL_INDICES

        self._gripper_command[indices] = states

    def reset(self, indices: torch.Tensor | None = None) -> None:
        """Reset the gripper command buffer.

        Args:
            indices: A tensor of integers representing the indices of the grippers to reset the command for. Defaults
                     to None, in which case all grippers are reset.
        """
        # Would normally set the buffer to 0, for now we won't do that
        if indices is None:
            indices = self._ALL_INDICES

        # Reset the selected grippers to an open status
        self._gripper_command[indices] = -1.0
        self.write_data_to_sim()
        # Sets the gripper last command to be 0.0 (do nothing)
        self._gripper_command[indices] = 0
        # Force set the state to open. It will read open in the next update call.
        self._gripper_state[indices] = -1.0

    """
    Initialization.
    """

    def _initialize_impl(self) -> None:
        """Initializes the gripper-related handles and internal buffers.

        Raises:
            ValueError: If the simulation backend is not CPU.
            RuntimeError: If the Simulation Context is not initialized.

        Note:
            The SurfaceGripper is only supported on CPU for now. Please set the simulation backend to run on CPU.
            Use `--device cpu` to run the simulation on CPU.
        """

        enable_extension("isaacsim.robot.surface_gripper")
        from isaacsim.robot.surface_gripper import GripperView

        # Check that we are using the CPU backend.
        if self._device != "cpu":
            raise Exception(
                "SurfaceGripper is only supported on CPU for now. Please set the simulation backend to run on CPU. Use"
                " `--device cpu` to run the simulation on CPU."
            )
        # Count number of environments
        self._prim_expr = self._cfg.prim_expr
        env_prim_path_expr = self._prim_expr.rsplit("/", 1)[0]
        self._parent_prims = sim_utils.find_matching_prims(env_prim_path_expr)
        self._num_envs = len(self._parent_prims)

        # Create buffers
        self._create_buffers()

        # Process the configuration
        self._process_cfg()

        # Initialize gripper view and set properties. Note we do not set the properties through the gripper view
        # to avoid having to convert them to list of floats here. Instead, we do it in the update_gripper_properties
        # function which does this conversion internally.
        self._gripper_view = GripperView(
            self._prim_expr,
        )
        self.update_gripper_properties(
            max_grip_distance=self._max_grip_distance.clone(),
            coaxial_force_limit=self._coaxial_force_limit.clone(),
            shear_force_limit=self._shear_force_limit.clone(),
            retry_interval=self._retry_interval.clone(),
        )

        # Reset grippers
        self.reset()

    def _create_buffers(self) -> None:
        """Create the buffers for storing the gripper state, command, and properties."""
        self._gripper_state = torch.zeros(self._num_envs, device=self._device, dtype=torch.float32)
        self._gripper_command = torch.zeros(self._num_envs, device=self._device, dtype=torch.float32)
        self._ALL_INDICES = torch.arange(self._num_envs, device=self._device, dtype=torch.long)

        self._max_grip_distance = torch.zeros(self._num_envs, device=self._device, dtype=torch.float32)
        self._coaxial_force_limit = torch.zeros(self._num_envs, device=self._device, dtype=torch.float32)
        self._shear_force_limit = torch.zeros(self._num_envs, device=self._device, dtype=torch.float32)
        self._retry_interval = torch.zeros(self._num_envs, device=self._device, dtype=torch.float32)

    def _process_cfg(self) -> None:
        """Process the configuration for the gripper properties."""
        # Get one of the grippers as defined in the default stage
        gripper_prim = self._parent_prims[0]
        try:
            max_grip_distance = gripper_prim.GetAttribute("isaac:maxGripDistance").Get()
        except Exception as e:
            warnings.warn(
                f"Failed to retrieve max_grip_distance from stage, defaulting to user provided cfg. Exception: {e}"
            )
            max_grip_distance = None

        try:
            coaxial_force_limit = gripper_prim.GetAttribute("isaac:coaxialForceLimit").Get()
        except Exception as e:
            warnings.warn(
                f"Failed to retrieve coaxial_force_limit from stage, defaulting to user provided cfg. Exception: {e}"
            )
            coaxial_force_limit = None

        try:
            shear_force_limit = gripper_prim.GetAttribute("isaac:shearForceLimit").Get()
        except Exception as e:
            warnings.warn(
                f"Failed to retrieve shear_force_limit from stage, defaulting to user provided cfg. Exception: {e}"
            )
            shear_force_limit = None

        try:
            retry_interval = gripper_prim.GetAttribute("isaac:retryInterval").Get()
        except Exception as e:
            warnings.warn(
                f"Failed to retrieve retry_interval from stage defaulting to user provided cfg. Exception: {e}"
            )
            retry_interval = None

        self._max_grip_distance = self.parse_gripper_parameter(self._cfg.max_grip_distance, max_grip_distance)
        self._coaxial_force_limit = self.parse_gripper_parameter(self._cfg.coaxial_force_limit, coaxial_force_limit)
        self._shear_force_limit = self.parse_gripper_parameter(self._cfg.shear_force_limit, shear_force_limit)
        self._retry_interval = self.parse_gripper_parameter(self._cfg.retry_interval, retry_interval)

    """
    Helper functions.
    """

    def parse_gripper_parameter(
        self, cfg_value: float | int | tuple | None, default_value: float | int | tuple | None, ndim: int = 0
    ) -> torch.Tensor:
        """Parse the gripper parameter.

        Args:
            cfg_value: The value to parse. Can be a float, int, tuple, or None.
            default_value: The default value to use if cfg_value is None. Can be a float, int, tuple, or None.
            ndim: The number of dimensions of the parameter. Defaults to 0.
        """
        # Adjust the buffer size based on the number of dimensions
        if ndim == 0:
            param = torch.zeros(self._num_envs, device=self._device)
        elif ndim == 3:
            param = torch.zeros(self._num_envs, 3, device=self._device)
        elif ndim == 4:
            param = torch.zeros(self._num_envs, 4, device=self._device)
        else:
            raise ValueError(f"Invalid number of dimensions: {ndim}")

        # Parse the parameter
        if cfg_value is not None:
            if isinstance(cfg_value, (float, int)):
                param[:] = float(cfg_value)
            elif isinstance(cfg_value, tuple):
                if len(cfg_value) == ndim:
                    param[:] = torch.tensor(cfg_value, dtype=torch.float, device=self._device)
                else:
                    raise ValueError(f"Invalid number of values for parameter. Got: {len(cfg_value)}\nExpected: {ndim}")
            else:
                raise TypeError(f"Invalid type for parameter value: {type(cfg_value)}. " + "Expected float or int.")
        elif default_value is not None:
            if isinstance(default_value, (float, int)):
                param[:] = float(default_value)
            elif isinstance(default_value, tuple):
                assert len(default_value) == ndim, f"Expected {ndim} values, got {len(default_value)}"
                param[:] = torch.tensor(default_value, dtype=torch.float, device=self._device)
            else:
                raise TypeError(
                    f"Invalid type for default value: {type(default_value)}. " + "Expected float or Tensor."
                )
        else:
            raise ValueError("The parameter value is None and no default value is provided.")

        return param
