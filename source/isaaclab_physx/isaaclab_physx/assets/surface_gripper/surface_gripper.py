# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING

import numpy as np
import torch
import warp as wp

from isaacsim.core.utils.extensions import enable_extension

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBase
from isaaclab.utils.version import get_isaac_sim_version, has_kit

if TYPE_CHECKING:
    from isaacsim.robot.surface_gripper import GripperView

    from .surface_gripper_cfg import SurfaceGripperCfg

# import logger
logger = logging.getLogger(__name__)

# -- Warp kernels --


@wp.kernel
def write_scalar_at_indices(
    data: wp.array(dtype=wp.float32),
    env_ids: wp.array(dtype=wp.int32),
    from_mask: bool,
    out: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    if from_mask:
        out[env_ids[i]] = data[env_ids[i]]
    else:
        out[env_ids[i]] = data[i]


@wp.kernel
def fill_scalar_at_indices(
    value: wp.float32,
    env_ids: wp.array(dtype=wp.int32),
    out: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    out[env_ids[i]] = value


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

    .. note::
        The :func:`set_grippers_command` function does not write to the simulation. The simulation automatically
         calls :func:`write_data_to_sim` function to write the command to the simulation. Similarly, the update
         function is called automatically for every simulation step, and does not need to be called by the user.

    .. note::
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

        # checks for Isaac Sim v5.0 to ensure that the surface gripper is supported
        if has_kit() and get_isaac_sim_version().major < 5:
            raise NotImplementedError(
                "SurfaceGrippers are only supported by IsaacSim 5.0 and newer. Current version is"
                f" '{get_isaac_sim_version()}'. Please update to IsaacSim 5.0 or newer to use this feature."
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
    def state(self) -> wp.array:
        """Returns the gripper state buffer.

        The gripper state is a list of integers:
        - -1 --> Open
        - 0 --> Closing
        - 1 --> Closed
        """
        return self._gripper_state

    @property
    def command(self) -> wp.array:
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
    Operations - _index / _mask API
    """

    def set_grippers_command_index(
        self, states: torch.Tensor | wp.array, env_ids: wp.array | None = None, full_data: bool = False
    ) -> None:
        """Set the internal gripper command buffer. This function does not write to the simulation.

        Possible values for the gripper command are:
            - [-1, -0.3] --> Open
            - ]-0.3, 0.3[ --> Do nothing
            - [0.3, 1] --> Close

        Args:
            states: A tensor/array of floats representing the gripper command.
            env_ids: Environment indices. Defaults to None (all environments).
            full_data: Whether ``states`` is indexed by ``env_ids`` (True) or is compact (False).
        """
        if env_ids is None:
            env_ids = self._ALL_INDICES

        # Convert torch input to warp
        if isinstance(states, torch.Tensor):
            states = wp.from_torch(states.to(torch.float32).contiguous(), dtype=wp.float32)

        wp.launch(
            write_scalar_at_indices,
            dim=env_ids.shape[0],
            inputs=[states, env_ids, full_data],
            outputs=[self._gripper_command],
            device=self._device,
        )

    def set_grippers_command_mask(self, states: torch.Tensor | wp.array, env_mask: torch.Tensor | None = None) -> None:
        """Set the internal gripper command buffer using a boolean mask.

        Args:
            states: A tensor/array of floats representing the gripper command (full size).
            env_mask: Boolean mask of shape (num_envs,). Defaults to None (all environments).
        """
        if env_mask is not None:
            env_ids = wp.from_torch(torch.argwhere(env_mask).view(-1).to(torch.int32), dtype=wp.int32)
        else:
            env_ids = self._ALL_INDICES
        self.set_grippers_command_index(states, env_ids, full_data=True)

    def set_grippers_command(self, states: torch.Tensor, indices: torch.Tensor | None = None) -> None:
        """Set the internal gripper command buffer. This function does not write to the simulation.

        .. deprecated:: v2.0.0
            Use :meth:`set_grippers_command_index` instead.

        Args:
            states: A tensor of integers representing the gripper command.
            indices: A tensor of integers representing the indices. Defaults to None.
        """
        env_ids = self._resolve_env_ids(indices)
        self.set_grippers_command_index(states, env_ids)

    def update_gripper_properties_index(
        self,
        max_grip_distance: wp.array | None = None,
        coaxial_force_limit: wp.array | None = None,
        shear_force_limit: wp.array | None = None,
        retry_interval: wp.array | None = None,
        env_ids: wp.array | None = None,
        full_data: bool = False,
    ) -> None:
        """Update the gripper properties.

        Args:
            max_grip_distance: The maximum grip distance. Shape (num_envs,) or (len(env_ids),).
            coaxial_force_limit: The coaxial force limit. Shape (num_envs,) or (len(env_ids),).
            shear_force_limit: The shear force limit. Shape (num_envs,) or (len(env_ids),).
            retry_interval: The retry interval. Shape (num_envs,) or (len(env_ids),).
            env_ids: Environment indices. Defaults to None (all environments).
            full_data: Whether input arrays are indexed by ``env_ids`` (True) or compact (False).
        """
        if env_ids is None:
            env_ids = self._ALL_INDICES

        for prop_data, prop_buf in [
            (max_grip_distance, self._max_grip_distance),
            (coaxial_force_limit, self._coaxial_force_limit),
            (shear_force_limit, self._shear_force_limit),
            (retry_interval, self._retry_interval),
        ]:
            if prop_data is not None:
                wp.launch(
                    write_scalar_at_indices,
                    dim=env_ids.shape[0],
                    inputs=[prop_data, env_ids, full_data],
                    outputs=[prop_buf],
                    device=self._device,
                )

        # Convert to list for the GripperView API
        indices_list = wp.to_torch(env_ids).tolist()
        self._gripper_view.set_surface_gripper_properties(
            max_grip_distance=wp.to_torch(self._max_grip_distance).tolist(),
            coaxial_force_limit=wp.to_torch(self._coaxial_force_limit).tolist(),
            shear_force_limit=wp.to_torch(self._shear_force_limit).tolist(),
            retry_interval=wp.to_torch(self._retry_interval).tolist(),
            indices=indices_list,
        )

    def update_gripper_properties_mask(
        self,
        max_grip_distance: wp.array | None = None,
        coaxial_force_limit: wp.array | None = None,
        shear_force_limit: wp.array | None = None,
        retry_interval: wp.array | None = None,
        env_mask: torch.Tensor | None = None,
    ) -> None:
        """Update the gripper properties using a boolean mask.

        Args:
            max_grip_distance: The maximum grip distance. Shape (num_envs,).
            coaxial_force_limit: The coaxial force limit. Shape (num_envs,).
            shear_force_limit: The shear force limit. Shape (num_envs,).
            retry_interval: The retry interval. Shape (num_envs,).
            env_mask: Boolean mask of shape (num_envs,). Defaults to None (all environments).
        """
        if env_mask is not None:
            env_ids = wp.from_torch(torch.argwhere(env_mask).view(-1).to(torch.int32), dtype=wp.int32)
        else:
            env_ids = self._ALL_INDICES
        self.update_gripper_properties_index(
            max_grip_distance=max_grip_distance,
            coaxial_force_limit=coaxial_force_limit,
            shear_force_limit=shear_force_limit,
            retry_interval=retry_interval,
            env_ids=env_ids,
            full_data=True,
        )

    def update_gripper_properties(
        self,
        max_grip_distance: torch.Tensor | None = None,
        coaxial_force_limit: torch.Tensor | None = None,
        shear_force_limit: torch.Tensor | None = None,
        retry_interval: torch.Tensor | None = None,
        indices: torch.Tensor | None = None,
    ) -> None:
        """Update the gripper properties.

        .. deprecated:: v2.0.0
            Use :meth:`update_gripper_properties_index` instead.

        Args:
            max_grip_distance: The maximum grip distance. Shape (num_envs,).
            coaxial_force_limit: The coaxial force limit. Shape (num_envs,).
            shear_force_limit: The shear force limit. Shape (num_envs,).
            retry_interval: The retry interval. Shape (num_envs,).
            indices: The indices of the grippers to update. Defaults to None.
        """
        env_ids = self._resolve_env_ids(indices)

        # Convert torch inputs to warp
        def _to_wp(t):
            if t is None:
                return None
            if isinstance(t, torch.Tensor):
                return wp.from_torch(t.to(torch.float32).contiguous(), dtype=wp.float32)
            return t

        self.update_gripper_properties_index(
            max_grip_distance=_to_wp(max_grip_distance),
            coaxial_force_limit=_to_wp(coaxial_force_limit),
            shear_force_limit=_to_wp(shear_force_limit),
            retry_interval=_to_wp(retry_interval),
            env_ids=env_ids,
        )

    def update(self, dt: float) -> None:
        """Update the gripper state using the SurfaceGripperView.

        This function is called every simulation step.
        The data fetched from the gripper view is a list of strings containing 3 possible states:

        - "Open" --> 0
        - "Closing" --> 1
        - "Closed" --> 2

        To make this more neural network friendly, we convert the list of strings to a list of floats:

        - "Open" --> -1.0
        - "Closing" --> 0.0
        - "Closed" --> 1.0

        .. note::
            We need to do this conversion for every single step of the simulation because the gripper can lose contact
            with the object if some conditions are met: such as if a large force is applied to the gripped object.
        """
        state_list: list[int] = self._gripper_view.get_surface_gripper_status()
        self._gripper_state = wp.array([float(s) - 1.0 for s in state_list], dtype=wp.float32, device=self._device)

    def write_data_to_sim(self) -> None:
        """Write the gripper command to the SurfaceGripperView.

        The gripper command is a list of integers that needs to be converted to a list of strings:
            - [-1, -0.3] --> Open
            - ]-0.3, 0.3[ --> Do nothing
            - [0.3, 1] --> Closed

        The Do nothing command is not applied, and is only used to indicate whether the gripper state has changed.
        """
        # Convert to torch at the GripperView boundary (zero-copy on CPU)
        command_torch = wp.to_torch(self._gripper_command)
        # Remove the SurfaceGripper indices that have a commanded value of 2
        indices = torch.argwhere(torch.logical_or(command_torch < -0.3, command_torch > 0.3)).to(torch.int32).tolist()
        # Write to the SurfaceGripperView if there are any indices to write to
        if len(indices) > 0:
            self._gripper_view.apply_gripper_action(command_torch.tolist(), indices)

    def reset_index(self, env_ids: wp.array | None = None) -> None:
        """Reset the gripper command buffer.

        Args:
            env_ids: Environment indices. Defaults to None (all environments).
        """
        if env_ids is None:
            env_ids = self._ALL_INDICES

        # Reset the selected grippers to an open status
        wp.launch(
            fill_scalar_at_indices,
            dim=env_ids.shape[0],
            inputs=[wp.float32(-1.0), env_ids],
            outputs=[self._gripper_command],
            device=self._device,
        )
        self.write_data_to_sim()
        # Sets the gripper last command to be 0.0 (do nothing)
        wp.launch(
            fill_scalar_at_indices,
            dim=env_ids.shape[0],
            inputs=[wp.float32(0.0), env_ids],
            outputs=[self._gripper_command],
            device=self._device,
        )
        # Force set the state to open. It will read open in the next update call.
        wp.launch(
            fill_scalar_at_indices,
            dim=env_ids.shape[0],
            inputs=[wp.float32(-1.0), env_ids],
            outputs=[self._gripper_state],
            device=self._device,
        )

    def reset_mask(self, env_mask: torch.Tensor | None = None) -> None:
        """Reset the gripper command buffer using a boolean mask.

        Args:
            env_mask: Boolean mask of shape (num_envs,). Defaults to None (all environments).
        """
        if env_mask is not None:
            env_ids = wp.from_torch(torch.argwhere(env_mask).view(-1).to(torch.int32), dtype=wp.int32)
        else:
            env_ids = self._ALL_INDICES
        self.reset_index(env_ids)

    def reset(self, indices: torch.Tensor | None = None) -> None:
        """Reset the gripper command buffer.

        .. deprecated:: v2.0.0
            Use :meth:`reset_index` instead.

        Args:
            indices: A tensor of integers representing the indices of the grippers to reset. Defaults to None.
        """
        env_ids = self._resolve_env_ids(indices)
        self.reset_index(env_ids)

    """
    Initialization.
    """

    def _initialize_impl(self) -> None:
        """Initializes the gripper-related handles and internal buffers.

        Raises:
            ValueError: If the simulation backend is not CPU.
            RuntimeError: If the Simulation Context is not initialized or if gripper prims are not found.

        .. note::
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

        # obtain the first prim in the regex expression (all others are assumed to be a copy of this)
        template_prim = sim_utils.find_first_matching_prim(self._cfg.prim_path)
        if template_prim is None:
            raise RuntimeError(f"Failed to find prim for expression: '{self._cfg.prim_path}'.")
        template_prim_path = template_prim.GetPath().pathString

        # find surface gripper prims
        gripper_prims = sim_utils.get_all_matching_child_prims(
            template_prim_path,
            predicate=lambda prim: prim.GetTypeName() == "IsaacSurfaceGripper",
            traverse_instance_prims=False,
        )
        if len(gripper_prims) == 0:
            raise RuntimeError(
                f"Failed to find a surface gripper when resolving '{self._cfg.prim_path}'."
                " Please ensure that the prim has type 'IsaacSurfaceGripper'."
            )
        if len(gripper_prims) > 1:
            raise RuntimeError(
                f"Failed to find a single surface gripper when resolving '{self._cfg.prim_path}'."
                f" Found multiple '{gripper_prims}' under '{template_prim_path}'."
                " Please ensure that there is only one surface gripper in the prim path tree."
            )

        # resolve gripper prim back into regex expression
        gripper_prim_path = gripper_prims[0].GetPath().pathString
        gripper_prim_path_expr = self._cfg.prim_path + gripper_prim_path[len(template_prim_path) :]

        # Count number of environments
        self._prim_expr = gripper_prim_path_expr
        env_prim_path_expr = self._prim_expr.rsplit("/", 1)[0]
        self._parent_prims = sim_utils.find_matching_prims(env_prim_path_expr)
        self._num_envs = len(self._parent_prims)

        # Create buffers
        self._create_buffers()

        # Process the configuration
        self._process_cfg()

        # Initialize gripper view and set properties.
        self._gripper_view = GripperView(
            self._prim_expr,
        )
        self.update_gripper_properties_index(
            max_grip_distance=wp.clone(self._max_grip_distance),
            coaxial_force_limit=wp.clone(self._coaxial_force_limit),
            shear_force_limit=wp.clone(self._shear_force_limit),
            retry_interval=wp.clone(self._retry_interval),
        )

        # log information about the surface gripper
        logger.info(f"Surface gripper initialized at: {self._cfg.prim_path} with root '{gripper_prim_path_expr}'.")
        logger.info(f"Number of instances: {self._num_envs}")

        # Reset grippers
        self.reset_index()

    def _create_buffers(self) -> None:
        """Create the buffers for storing the gripper state, command, and properties."""
        self._gripper_state = wp.zeros(self._num_envs, dtype=wp.float32, device=self._device)
        self._gripper_command = wp.zeros(self._num_envs, dtype=wp.float32, device=self._device)
        self._ALL_INDICES = wp.array(np.arange(self._num_envs, dtype=np.int32), device=self._device)

        self._max_grip_distance = wp.zeros(self._num_envs, dtype=wp.float32, device=self._device)
        self._coaxial_force_limit = wp.zeros(self._num_envs, dtype=wp.float32, device=self._device)
        self._shear_force_limit = wp.zeros(self._num_envs, dtype=wp.float32, device=self._device)
        self._retry_interval = wp.zeros(self._num_envs, dtype=wp.float32, device=self._device)

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

    def _resolve_env_ids(self, env_ids) -> wp.array:
        """Resolve environment indices to a warp array.

        Args:
            env_ids: Environment indices. Can be None, a slice, a list, or a torch.Tensor.

        Returns:
            A warp array of int32 indices.
        """
        if env_ids is None or env_ids == slice(None):
            return self._ALL_INDICES
        elif isinstance(env_ids, list):
            return wp.array(env_ids, dtype=wp.int32, device=self._device)
        elif isinstance(env_ids, torch.Tensor):
            return wp.from_torch(env_ids.to(torch.int32).contiguous(), dtype=wp.int32)
        return env_ids

    def parse_gripper_parameter(
        self, cfg_value: float | int | tuple | None, default_value: float | int | tuple | None, ndim: int = 0
    ) -> wp.array:
        """Parse the gripper parameter.

        Args:
            cfg_value: The value to parse. Can be a float, int, tuple, or None.
            default_value: The default value to use if cfg_value is None. Can be a float, int, tuple, or None.
            ndim: The number of dimensions of the parameter. Defaults to 0.

        Returns:
            A warp array of float32 values.
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

        # Convert to warp
        return wp.from_torch(param, dtype=wp.float32)
