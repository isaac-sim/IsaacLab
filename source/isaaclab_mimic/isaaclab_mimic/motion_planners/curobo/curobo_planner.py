# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import warp as wp

from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModelState
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.sphere_fit import SphereFitType
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.state import JointState
from curobo.util.logger import setup_curobo_logger
from curobo.util.usd_helper import UsdHelper
from curobo.util_file import load_yaml
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig

import isaaclab.utils.math as PoseUtils
from isaaclab.assets import Articulation
from isaaclab.envs.manager_based_env import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim.spawners.materials import PreviewSurfaceCfg
from isaaclab.sim.spawners.meshes import MeshSphereCfg, spawn_mesh_sphere

from isaaclab_mimic.motion_planners.curobo.curobo_planner_cfg import CuroboPlannerCfg
from isaaclab_mimic.motion_planners.motion_planner_base import MotionPlannerBase


class PlannerLogger:
    """Logger class for motion planner debugging and monitoring.

    This class provides standard logging functionality while maintaining isolation from
    the main application's logging configuration. The logger supports configurable verbosity
    levels and formats messages consistently for debugging motion planning operations,
    collision checking, and object manipulation.
    """

    def __init__(self, name: str, level: int = logging.INFO):
        """Initialize the logger with specified name and level.

        Args:
            name: Logger name for identification in log messages
            level: Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        self._name = name
        self._level = level
        self._logger = None

    @property
    def logger(self):
        """Get the underlying logger instance, initializing it if needed.

        Returns:
            Configured Python logger instance with stream handler and formatter
        """
        if self._logger is None:
            self._logger = logging.getLogger(self._name)
            if not self._logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
                handler.setFormatter(formatter)
                self._logger.addHandler(handler)
                self._logger.setLevel(self._level)
        return self._logger

    def debug(self, msg, *args, **kwargs):
        """Log debug-level message for detailed internal state information.

        Args:
            msg: Message string or format string
            *args: Positional arguments for message formatting
            **kwargs: Keyword arguments passed to underlying logger
        """
        self.logger.debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        """Log info-level message for important operational events.

        Args:
            msg: Message string or format string
            *args: Positional arguments for message formatting
            **kwargs: Keyword arguments passed to underlying logger
        """
        self.logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        """Log warning-level message for potentially problematic conditions.

        Args:
            msg: Message string or format string
            *args: Positional arguments for message formatting
            **kwargs: Keyword arguments passed to underlying logger
        """
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        """Log error-level message for serious problems and failures.

        Args:
            msg: Message string or format string
            *args: Positional arguments for message formatting
            **kwargs: Keyword arguments passed to underlying logger
        """
        self.logger.error(msg, *args, **kwargs)


@dataclass
class Attachment:
    """Stores object attachment information for robot manipulation.

    This dataclass tracks the relative pose between an attached object and its parent link,
    enabling the robot to maintain consistent object positioning during motion planning.
    """

    pose: Pose  # Relative pose from parent link to object
    parent: str  # Parent link name


class CuroboPlanner(MotionPlannerBase):
    """Motion planner for robot manipulation using cuRobo.

    This planner provides collision-aware motion planning capabilities for robotic manipulation tasks.
    It integrates with Isaac Lab environments to:

    - Update collision world from current stage state
    - Plan collision-free paths to target poses
    - Handle object attachment and detachment during manipulation
    - Execute planned motions with proper collision checking

    The planner uses cuRobo for fast motion generation and supports
    multi-phase planning for contact scenarios like grasping and placing objects.
    """

    def __init__(
        self,
        env: ManagerBasedEnv,
        robot: Articulation,
        config: CuroboPlannerCfg,
        task_name: str | None = None,
        env_id: int = 0,
        collision_checker: CollisionCheckerType = CollisionCheckerType.MESH,
        num_trajopt_seeds: int = 12,
        num_graph_seeds: int = 12,
        interpolation_dt: float = 0.05,
    ) -> None:
        """Initialize the motion planner for a specific environment.

        Sets up the cuRobo motion generator with collision checking, configures the robot model,
        and prepares visualization components if enabled. The planner is isolated to CUDA device
        regardless of Isaac Lab's device configuration.

        Args:
            env: The Isaac Lab environment instance containing the robot and scene
            robot: Robot articulation to plan motions for
            config: Configuration object containing planner parameters and settings
            task_name: Task name for auto-configuration
            env_id: Environment ID for multi-environment setups (0 to num_envs-1)
            collision_checker: Type of collision checker
            num_trajopt_seeds: Number of seeds for trajectory optimization
            num_graph_seeds: Number of seeds for graph search
            interpolation_dt: Time step for interpolating waypoints

        Raises:
            ValueError: If ``robot_config_file`` is not provided
        """
        # Initialize base class
        super().__init__(env=env, robot=robot, env_id=env_id, debug=config.debug_planner)

        # Initialize planner logger with debug level based on config
        log_level = logging.DEBUG if config.debug_planner else logging.INFO
        self.logger = PlannerLogger(f"CuroboPlanner_{env_id}", log_level)

        # Store instance variables
        self.config: CuroboPlannerCfg = config
        self.n_repeat: int | None = self.config.n_repeat
        self.step_size: float | None = self.config.motion_step_size
        self.visualize_plan: bool = self.config.visualize_plan
        self.visualize_spheres: bool = self.config.visualize_spheres

        # Log the config parameter values
        self.logger.info(f"Config parameter values: {self.config}")

        # Initialize plan visualizer if enabled
        if self.visualize_plan:
            from isaaclab_mimic.motion_planners.curobo.plan_visualizer import PlanVisualizer

            # Use env-local base translation for multi-env rendering consistency
            env_origin = self.env.scene.env_origins[env_id, :3]
            base_translation = (wp.to_torch(self.robot.data.root_pos_w)[env_id, :3] - env_origin).detach().cpu().numpy()
            self.plan_visualizer = PlanVisualizer(
                robot_name=self.config.robot_name,
                recording_id=f"curobo_plan_{env_id}",
                debug=config.debug_planner,
                base_translation=base_translation,
            )

        # Store attached objects as Attachment objects
        self.attached_objects: dict[str, Attachment] = {}  # object_name -> Attachment

        # Initialize cuRobo components - FORCE CUDA DEVICE FOR ISOLATION
        setup_curobo_logger("warn")

        # Force cuRobo to always use CUDA device regardless of Isaac Lab device
        # This isolates the motion planner from Isaac Lab's device configuration
        self.tensor_args: TensorDeviceType
        if torch.cuda.is_available():
            idx = self.config.cuda_device if self.config.cuda_device is not None else torch.cuda.current_device()
            self.tensor_args = TensorDeviceType(device=torch.device(f"cuda:{idx}"), dtype=torch.float32)
            self.logger.debug(f"cuRobo motion planner initialized on CUDA device {idx}")
        else:
            # Fallback to CPU if CUDA not available, but this may cause issues
            self.tensor_args = TensorDeviceType()
            self.logger.warning("CUDA not available, cuRobo using CPU - this may cause device compatibility issues")

        # Load robot configuration
        if self.config.robot_config_file is None:
            raise ValueError("robot_config_file is required")
        robot_cfg_file = self.config.robot_config_file
        robot_cfg: dict[str, Any] = load_yaml(robot_cfg_file)["robot_cfg"]
        self.logger.info(f"Loaded robot configuration from {robot_cfg_file}")

        # Configure collision spheres
        if self.config.collision_spheres_file:
            robot_cfg["kinematics"]["collision_spheres"] = self.config.collision_spheres_file

        # Configure extra collision spheres
        if self.config.extra_collision_spheres:
            robot_cfg["kinematics"]["extra_collision_spheres"] = self.config.extra_collision_spheres

        self.robot_cfg: dict[str, Any] = robot_cfg

        # Load world configuration using the config's method
        world_cfg: WorldConfig = self.config.get_world_config()

        # Create motion generator config with parameters from configuration
        motion_gen_config: MotionGenConfig = MotionGenConfig.load_from_robot_config(
            robot_cfg,
            world_cfg,
            tensor_args=self.tensor_args,
            collision_checker_type=self.config.collision_checker_type,
            num_trajopt_seeds=self.config.num_trajopt_seeds,
            num_graph_seeds=self.config.num_graph_seeds,
            interpolation_dt=self.config.interpolation_dt,
            collision_cache=self.config.collision_cache_size,
            trajopt_tsteps=self.config.trajopt_tsteps,
            collision_activation_distance=self.config.collision_activation_distance,
            position_threshold=self.config.position_threshold,
            rotation_threshold=self.config.rotation_threshold,
        )

        # Create motion generator
        self.motion_gen: MotionGen = MotionGen(motion_gen_config)

        # Set motion generator reference for plan visualizer if enabled
        if self.visualize_plan:
            self.plan_visualizer.set_motion_generator_reference(self.motion_gen)

        # Create plan config with parameters from configuration
        self.plan_config: MotionGenPlanConfig = MotionGenPlanConfig(
            enable_graph=self.config.enable_graph,
            enable_graph_attempt=self.config.enable_graph_attempt,
            max_attempts=self.config.max_planning_attempts,
            enable_finetune_trajopt=self.config.enable_finetune_trajopt,
            time_dilation_factor=self.config.time_dilation_factor,
        )

        # Create USD helper
        self.usd_helper: UsdHelper = UsdHelper()
        self.usd_helper.load_stage(env.scene.stage)

        # Initialize planning state
        self._current_plan: JointState | None = None
        self._plan_index: int = 0

        # Initialize visualization state
        self.frame_counter: int = 0
        self.spheres: list[tuple[str, float]] | None = None
        self.sphere_update_freq: int = self.config.sphere_update_freq

        # Warm up planner
        self.logger.info("Warming up motion planner...")
        self.motion_gen.warmup(enable_graph=True, warmup_js_trajopt=False)

        # Read static world geometry once
        self._initialize_static_world()

        # Defer object validation baseline until first update_world() call when scene is fully loaded
        self._expected_objects: set[str] | None = None

        # Define supported cuRobo primitive types for object discovery and pose synchronization
        self.primitive_types: list[str] = ["mesh", "cuboid", "sphere", "capsule", "cylinder", "voxel", "blox"]

        # Cache object mappings
        # Only recompute when objects are added/removed, not when poses change
        self._cached_object_mappings: dict[str, str] | None = None

    # =====================================================================================
    # DEVICE CONVERSION UTILITIES
    # =====================================================================================

    def _to_curobo_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Convert tensor to cuRobo device for isolated device management.

        Ensures all tensors used by cuRobo are on CUDA device, providing device isolation
        from Isaac Lab's potentially different device configuration. This prevents device
        mismatch errors and optimizes cuRobo performance.

        Args:
            tensor: Input tensor (may be on any device)

        Returns:
            Tensor converted to cuRobo's CUDA device with appropriate dtype
        """
        return tensor.to(device=self.tensor_args.device, dtype=self.tensor_args.dtype)

    def _to_env_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Convert tensor back to environment device for Isaac Lab compatibility.

        Converts cuRobo tensors back to the environment's device to ensure compatibility
        with Isaac Lab operations that expect tensors on the environment's configured device.

        Args:
            tensor: Input tensor from cuRobo operations (typically on CUDA)

        Returns:
            Tensor converted to environment's device while preserving dtype
        """
        return tensor.to(device=self.env.device, dtype=tensor.dtype)

    # =====================================================================================
    # INITIALIZATION AND CONFIGURATION
    # =====================================================================================

    def _initialize_static_world(self) -> None:
        """Initialize static world geometry from USD stage.

        Reads static environment geometry once during planner initialization to establish
        the base collision world. This includes walls, tables, bins, and other fixed obstacles
        that don't change during the simulation. Dynamic objects are synchronized separately
        in update_world() to maintain performance.
        """
        env_prim_path = f"/World/envs/env_{self.env_id}"
        robot_prim_path = self.config.robot_prim_path or f"{env_prim_path}/Robot"

        ignore_list = self.config.world_ignore_substrings or [
            f"{env_prim_path}/Robot",
            f"{env_prim_path}/target",
            "/World/defaultGroundPlane",
            "/curobo",
        ]

        self._static_world_config = self.usd_helper.get_obstacles_from_stage(
            only_paths=[env_prim_path],
            reference_prim_path=robot_prim_path,
            ignore_substring=ignore_list,
        )
        self._static_world_config = self._static_world_config.get_collision_check_world()

        # Initialize cuRobo world with static geometry
        self.motion_gen.update_world(self._static_world_config)

    # =====================================================================================
    # PROPERTIES AND BASIC GETTERS
    # =====================================================================================

    @property
    def attached_link(self) -> str:
        """Default link name for object attachment operations."""
        return self.config.attached_object_link_name

    @property
    def attachment_links(self) -> set[str]:
        """Set of parent link names that currently have attached objects."""
        return {attachment.parent for attachment in self.attached_objects.values()}

    @property
    def current_plan(self) -> JointState | None:
        """Current plan from cuRobo motion generator."""
        return self._current_plan

    # =====================================================================================
    # WORLD AND OBJECT MANAGEMENT, ATTACHMENT, AND DETACHMENT
    # =====================================================================================

    def get_object_pose(self, object_name: str) -> Pose | None:
        """Retrieve object pose from cuRobo's collision world model.

        Searches the collision world model for the specified object and returns its current
        pose. This is useful for attachment calculations and debugging collision world state.
        The method handles both mesh and cuboid object types automatically.

        Args:
            object_name: Short object name used in Isaac Lab scene (e.g., "cube_1")

        Returns:
            Object pose in cuRobo coordinate frame, or None if object not found
        """
        # Get cached object mappings
        object_mappings = self._get_object_mappings()
        world_model = self.motion_gen.world_coll_checker.world_model

        object_path = object_mappings.get(object_name)
        if not object_path:
            self.logger.debug(f"Object {object_name} not found in world model")
            return None

        # Search for object in world model
        for obj_list, _ in [
            (world_model.mesh, "mesh"),
            (world_model.cuboid, "cuboid"),
        ]:
            if not obj_list:
                continue

            for obj in obj_list:
                if obj.name and object_path in str(obj.name):
                    if obj.pose is not None:
                        return Pose.from_list(obj.pose, tensor_args=self.tensor_args)

        self.logger.debug(f"Object {object_name} found in mappings but pose not available")
        return None

    def get_attached_pose(self, link_name: str, joint_state: JointState | None = None) -> Pose:
        """Calculate pose of specified link using forward kinematics.

        Computes the world pose of any robot link at the given joint configuration.
        This is essential for attachment calculations where we need to know the exact
        pose of the parent link to compute relative object positions.

        Args:
            link_name: Name of the robot link to get pose for
            joint_state: Joint configuration to use for calculation, uses current state if None

        Returns:
            World pose of the specified link in cuRobo coordinate frame

        Raises:
            KeyError: If link_name is not found in the computed link poses
        """
        if joint_state is None:
            joint_state = self._get_current_joint_state_for_curobo()

        # Get all link states using the robot model
        link_state = self.motion_gen.kinematics.get_state(
            q=joint_state.position.detach().clone().to(device=self.tensor_args.device, dtype=self.tensor_args.dtype),
            calculate_jacobian=False,
        )

        # Extract all link poses
        link_poses = {}
        if link_state.links_position is not None and link_state.links_quaternion is not None:
            for i, link in enumerate(link_state.link_names):
                # cuRobo kinematics returns quaternions in (w, x, y, z) format
                link_poses[link] = self._make_pose(
                    position=link_state.links_position[..., i, :],
                    quaternion=link_state.links_quaternion[..., i, :],
                    name=link,
                    quat_is_xyzw=False,
                )

        # For attached object link, use ee_link from robot config as parent
        if link_name == self.config.attached_object_link_name:
            ee_link = self.config.ee_link_name or self.robot_cfg["kinematics"]["ee_link"]
            if ee_link in link_poses:
                self.logger.debug(f"Using {ee_link} for {link_name}")
                return link_poses[ee_link]

        # Return directly for other links
        if link_name in link_poses:
            return link_poses[link_name]
        raise KeyError(f"Link {link_name} not found in computed link poses")

    def create_attachment(
        self, object_name: str, link_name: str | None = None, joint_state: JointState | None = None
    ) -> Attachment:
        """Create attachment relationship between object and robot link.

        Computes the relative pose between an object and a robot link to enable the robot
        to carry the object consistently during motion planning. The attachment stores the transform
        from the parent link frame to the object frame, which remains constant while grasped.

        Args:
            object_name: Name of the object to attach
            link_name: Parent link for attachment, uses default attached_object_link if None
            joint_state: Robot configuration for calculation, uses current state if None

        Returns:
            Attachment object containing relative pose and parent link information
        """
        if link_name is None:
            link_name = self.attached_link
        if joint_state is None:
            joint_state = self._get_current_joint_state_for_curobo()

        # Get current link pose
        link_pose = self.get_attached_pose(link_name, joint_state)
        self.logger.info(f"Getting object pose for {object_name}")
        obj_pose = self.get_object_pose(object_name)

        # Compute relative pose
        attach_pose = link_pose.inverse().multiply(obj_pose)

        self.logger.debug(f"Creating attachment for {object_name} to {link_name}")
        self.logger.debug(f"Link pose: {link_pose.position}")
        self.logger.debug(f"Object pose (ACTUAL): {obj_pose.position}")
        self.logger.debug(f"Computed relative pose: {attach_pose.position}")

        return Attachment(attach_pose, link_name)

    def update_world(self) -> None:
        """Synchronize collision world with current Isaac Lab scene state.

        Updates all dynamic object poses in cuRobo's collision world to match their current
        positions in Isaac Lab. This ensures collision checking uses accurate object positions
        after simulation steps, resets, or manual object movements. Static world geometry
        is loaded once during initialization and not updated here for performance.

        The method validates that the set of objects hasn't changed at runtime, as cuRobo
        requires world model reinitialization when objects are added or removed.

        Raises:
            RuntimeError: If the set of objects has changed at runtime
        """

        # Establish validation baseline on first call, validate on subsequent calls
        if self._expected_objects is None:
            self._expected_objects = set(self._get_world_object_names())
            self.logger.debug(f"Established object validation baseline: {len(self._expected_objects)} objects")
        else:
            # Subsequent calls: validate no changes
            current_objects = set(self._get_world_object_names())
            if current_objects != self._expected_objects:
                added = current_objects - self._expected_objects
                removed = self._expected_objects - current_objects

                error_msg = "World objects changed at runtime!\n"
                if added:
                    error_msg += f"Added: {added}\n"
                if removed:
                    error_msg += f"Removed: {removed}\n"
                error_msg += "cuRobo world model must be reinitialized."

                # Invalidate cached mappings since object set changed
                self._cached_object_mappings = None

                raise RuntimeError(error_msg)

        # Sync object poses with Isaac Lab
        self._sync_object_poses_with_isaaclab()

        if self.visualize_spheres:
            self._update_sphere_visualization(force_update=True)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def _get_world_object_names(self) -> list[str]:
        """Extract all object names from cuRobo's collision world model.

        Iterates through all supported primitive types (mesh, cuboid, sphere, etc.) in the
        collision world and collects their names. This is used for world validation to detect
        when objects are added or removed at runtime.

        Returns:
            List of all object names currently in the collision world model
        """
        try:
            world_model = self.motion_gen.world_coll_checker.world_model

            # Handle case where world_model might be a list
            if isinstance(world_model, list):
                if len(world_model) <= self.env_id:
                    return []
                world_model = world_model[self.env_id]

            object_names = []

            # Get all primitive object names using the defined primitive types
            for primitive_type in self.primitive_types:
                if hasattr(world_model, primitive_type) and getattr(world_model, primitive_type):
                    primitive_list = getattr(world_model, primitive_type)
                    for primitive in primitive_list:
                        if primitive.name:
                            object_names.append(str(primitive.name))

            return object_names

        except Exception as e:
            self.logger.debug(f"ERROR getting world object names: {e}")
            return []

    def _sync_object_poses_with_isaaclab(self) -> None:
        """Synchronize cuRobo collision world with Isaac Lab object positions.

        Updates all dynamic object poses in cuRobo's world model to match their current
        positions in Isaac Lab. This ensures accurate collision checking after simulation
        steps or manual object movements. Static objects (bins, tables, walls) are skipped
        for performance as they shouldn't move during simulation.

        The method updates both the world model and the collision checker to ensure
        consistency across all cuRobo components.
        """
        # Get cached object mappings and world model
        object_mappings = self._get_object_mappings()
        world_model = self.motion_gen.world_coll_checker.world_model
        rigid_objects = self.env.scene.rigid_objects

        updated_count = 0

        for object_name, object_path in object_mappings.items():
            if object_name not in rigid_objects:
                continue

            # Skip static mesh objects - they should not be dynamically updated
            static_objects = getattr(self.config, "static_objects", [])
            if any(static_name in object_name.lower() for static_name in static_objects):
                self.logger.debug(f"SYNC: Skipping static object {object_name}")
                continue

            # Get current pose from Lab (may be on CPU or CUDA depending on --device flag)
            obj = rigid_objects[object_name]
            env_origin = self.env.scene.env_origins[self.env_id]
            current_pos_raw = wp.to_torch(obj.data.root_pos_w)[self.env_id] - env_origin
            current_quat_raw = wp.to_torch(obj.data.root_quat_w)[self.env_id]  # (x, y, z, w)

            # Convert to cuRobo device and extract float values for pose list
            current_pos = self._to_curobo_device(current_pos_raw)
            current_quat = self._to_curobo_device(current_quat_raw)

            # Convert to cuRobo pose format [pos_x, pos_y, pos_z, qw, qx, qy, qz]
            # Isaac Lab quaternion format: (x, y, z, w) -> cuRobo format: (w, x, y, z)
            pose_list = [
                float(current_pos[0].item()),
                float(current_pos[1].item()),
                float(current_pos[2].item()),
                float(current_quat[3].item()),  # w
                float(current_quat[0].item()),  # x
                float(current_quat[1].item()),  # y
                float(current_quat[2].item()),  # z
            ]

            # Update object pose in cuRobo's world model
            if self._update_object_in_world_model(world_model, object_name, object_path, pose_list):
                updated_count += 1

        self.logger.debug(f"SYNC: Updated {updated_count} object poses in cuRobo world model")

        # Sync object poses with collision checker
        if updated_count > 0:
            # Update individual obstacle poses in collision checker
            # This preserves static mesh objects unlike load_collision_model which rebuilds everything
            for object_name, object_path in object_mappings.items():
                if object_name not in rigid_objects:
                    continue

                # Skip static mesh objects - they should not be dynamically updated
                static_objects = getattr(self.config, "static_objects", [])
                if any(static_name in object_name.lower() for static_name in static_objects):
                    continue

                # Get current pose and update in collision checker
                obj = rigid_objects[object_name]
                env_origin = self.env.scene.env_origins[self.env_id]
                current_pos_raw = wp.to_torch(obj.data.root_pos_w)[self.env_id] - env_origin
                current_quat_raw = wp.to_torch(obj.data.root_quat_w)[self.env_id]

                current_pos = self._to_curobo_device(current_pos_raw)
                current_quat = self._to_curobo_device(current_quat_raw)

                # Create cuRobo pose and update collision checker directly
                curobo_pose = self._make_pose(position=current_pos, quaternion=current_quat)
                self.motion_gen.world_coll_checker.update_obstacle_pose(  # type: ignore
                    object_path, curobo_pose, update_cpu_reference=True
                )

            self.logger.debug(f"Updated {updated_count} object poses in collision checker")

    def _get_object_mappings(self) -> dict[str, str]:
        """Get object mappings with caching for performance optimization.

        Returns cached mappings if available, otherwise computes and caches them.
        Cache is invalidated when the object set changes.

        Returns:
            Dictionary mapping Isaac Lab object names to their corresponding USD paths
        """
        if self._cached_object_mappings is None:
            world_model = self.motion_gen.world_coll_checker.world_model
            rigid_objects = self.env.scene.rigid_objects
            self._cached_object_mappings = self._discover_object_mappings(world_model, rigid_objects)
            self.logger.debug(f"Computed and cached object mappings: {len(self._cached_object_mappings)} objects")

        return self._cached_object_mappings

    def _discover_object_mappings(self, world_model, rigid_objects) -> dict[str, str]:
        """Build mapping between Isaac Lab object names and cuRobo world paths.

        Automatically discovers the correspondence between Isaac Lab's rigid object names
        and their full USD paths in cuRobo's world model. This mapping is essential for
        pose synchronization and attachment operations, as cuRobo uses full USD paths
        while Isaac Lab uses short object names.

        Args:
            world_model: cuRobo's collision world model containing primitive objects
            rigid_objects: Isaac Lab's rigid objects dictionary

        Returns:
            Dictionary mapping Isaac Lab object names to their corresponding USD paths
        """
        mappings = {}
        env_prefix = f"/World/envs/env_{self.env_id}/"
        world_object_paths = []

        # Collect all primitive objects from cuRobo world model
        for primitive_type in self.primitive_types:
            primitive_list = getattr(world_model, primitive_type)
            for primitive in primitive_list:
                if primitive.name and env_prefix in str(primitive.name):
                    world_object_paths.append(str(primitive.name))

        # Match Isaac Lab object names to world paths
        for object_name in rigid_objects.keys():
            # Direct name matching
            for path in world_object_paths:
                if object_name.lower().replace("_", "") in path.lower().replace("_", ""):
                    mappings[object_name] = path
                    self.logger.debug(f"MAPPING: {object_name} -> {path}")
                    break
            else:
                self.logger.debug(f"WARNING: Could not find world path for {object_name}")

        return mappings

    def _update_object_in_world_model(
        self, world_model, object_name: str, object_path: str, pose_list: list[float]
    ) -> bool:
        """Update a single object's pose in cuRobo's collision world model.

        Searches through all primitive types in the world model to find the specified object
        and updates its pose. Uses flexible matching to handle variations in path naming
        between Isaac Lab and cuRobo representations.

        Args:
            world_model: cuRobo's collision world model
            object_name: Short object name from Isaac Lab (e.g., "cube_1")
            object_path: Full USD path for the object in cuRobo world
            pose_list: New pose as [x, y, z, w, x, y, z] list in cuRobo format

        Returns:
            True if object was found and successfully updated, False otherwise
        """
        # Handle case where world_model might be a list
        if isinstance(world_model, list):
            if len(world_model) > self.env_id:
                world_model = world_model[self.env_id]
            else:
                return False

        # Update all primitive types
        for primitive_type in self.primitive_types:
            primitive_list = getattr(world_model, primitive_type)
            for primitive in primitive_list:
                if primitive.name:
                    primitive_name = str(primitive.name)
                    # Use bidirectional matching for robust path matching
                    if object_path == primitive_name or object_path in primitive_name or primitive_name in object_path:
                        primitive.pose = pose_list
                        self.logger.debug(f"Updated {primitive_type} {object_name} pose")
                        return True

        self.logger.debug(f"WARNING: Object {object_name} not found in world model")
        return False

    def _attach_object(self, object_name: str, object_path: str, env_id: int) -> bool:
        """Attach an object to the robot for manipulation planning.

        Establishes an attachment between the specified object and the robot's end-effector
        or configured attachment link. This enables the robot to carry the object during
        motion planning while maintaining proper collision checking. The object's collision
        geometry is disabled in the world model since it's now part of the robot.

        Args:
            object_name: Short name of the object to attach (e.g., "cube_2")
            object_path: Full USD path for the object in cuRobo world model
            env_id: Environment ID for multi-environment support

        Returns:
            True if attachment succeeded, False if attachment failed
        """
        current_joint_state = self._get_current_joint_state_for_curobo()

        self.logger.debug(f"Attaching {object_name} at path {object_path}")

        # Create attachment record (relative pose object-frame to parent link)
        attachment = self.create_attachment(
            object_name,
            self.config.attached_object_link_name,
            current_joint_state,
        )
        self.attached_objects[object_name] = attachment
        success = self.motion_gen.attach_objects_to_robot(
            joint_state=current_joint_state,
            object_names=[object_path],
            link_name=self.config.attached_object_link_name,
            surface_sphere_radius=self.config.surface_sphere_radius,
            sphere_fit_type=SphereFitType.SAMPLE_SURFACE,
            world_objects_pose_offset=None,
        )

        if success:
            self.logger.debug(f"Successfully attached {object_name}")
            self.logger.debug(f"Current attached objects: {list(self.attached_objects.keys())}")

            # Force sphere visualization update
            if self.visualize_spheres:
                self._update_sphere_visualization(force_update=True)

            self.logger.info(f"Sphere count after attach is successful: {self._count_active_spheres()}")

            # Deactivate the original obstacle as it's now carried by the robot
            self.motion_gen.world_coll_checker.enable_obstacle(object_path, enable=False)

            return True
        else:
            self.logger.error(f"cuRobo attach_objects_to_robot failed for {object_name}")
            # Clean up on failure
            if object_name in self.attached_objects:
                del self.attached_objects[object_name]
            return False

    def _detach_objects(self, link_names: set[str] | None = None) -> bool:
        """Detach objects from robot and restore collision checking.

        Removes object attachments from specified links and re-enables collision checking
        for both the objects and the parent links. This is necessary when placing objects
        or changing grasps. All attached objects are detached if no specific links are provided.

        Args:
            link_names: Set of parent link names to detach objects from, detaches all if None

        Returns:
            True if detachment operations completed successfully, False otherwise
        """
        if link_names is None:
            link_names = self.attachment_links

        self.logger.debug(f"Detaching objects from links: {link_names}")
        self.logger.debug(f"Current attached objects: {list(self.attached_objects.keys())}")

        # Get cached object mappings to find the USD path for re-enabling
        object_mappings = self._get_object_mappings()

        detached_info = []
        detached_links = set()
        for object_name, attachment in list(self.attached_objects.items()):
            if attachment.parent not in link_names:
                continue

            # Find object path and re-enable it in the world
            object_path = object_mappings.get(object_name)
            if object_path:
                self.motion_gen.world_coll_checker.enable_obstacle(object_path, enable=True)  # type: ignore
                self.logger.debug(f"Re-enabled obstacle {object_path}")

            # Collect the link that will need re-enabling
            detached_links.add(attachment.parent)

            # Remove from attached objects and log info
            del self.attached_objects[object_name]
            detached_info.append((object_name, attachment.parent))

        if detached_info:
            for obj_name, parent_link in detached_info:
                self.logger.debug(f"Detached {obj_name} from {parent_link}")

        # Re-enable collision checking for the attachment links (following the planning pattern)
        if detached_links:
            self._set_active_links(list(detached_links), active=True)
            self.logger.debug(f"Re-enabled collision for attachment links: {detached_links}")

        # Call cuRobo's detach for each link
        for link_name in link_names:
            self.motion_gen.detach_object_from_robot(link_name=link_name)
            self.logger.debug(f"Called cuRobo detach for link {link_name}")

        return True

    def get_attached_objects(self) -> list[str]:
        """Get list of currently attached object names.

        Returns the short names of all objects currently attached to the robot.
        These names correspond to Isaac Lab scene object names, not full USD paths.

        Returns:
            List of attached object names (e.g., ["cube_1", "cube_2"])"""
        return list(self.attached_objects.keys())

    def has_attached_objects(self) -> bool:
        """Check if any objects are currently attached to the robot.

        Useful for determining gripper state and collision checking configuration
        before planning motions.

        Returns:
            True if one or more objects are attached, False if no attachments exist
        """
        return len(self.attached_objects) != 0

    # =====================================================================================
    # JOINT STATE AND KINEMATICS
    # =====================================================================================

    def _get_current_joint_state_for_curobo(self) -> JointState:
        """
        Construct the current joint state for cuRobo with zero velocity and acceleration.

        This helper reads the robot's joint positions from Isaac Lab for the current environment
        and pairs them with zero velocities and accelerations as required by cuRobo planning.
        All tensors are moved to the cuRobo device and reordered to match the kinematic chain
        used by the cuRobo motion generator.

        Returns:
            JointState on the cuRobo device, ordered according to
            `self.motion_gen.kinematics.joint_names`, with position from the robot
            and zero velocity/acceleration.
        """
        # Fetch joint position (shape: [1, num_joints])
        joint_pos_raw: torch.Tensor = wp.to_torch(self.robot.data.joint_pos)[self.env_id, :].unsqueeze(0)
        joint_vel_raw: torch.Tensor = torch.zeros_like(joint_pos_raw)
        joint_acc_raw: torch.Tensor = torch.zeros_like(joint_pos_raw)

        # Move to cuRobo device
        joint_pos: torch.Tensor = self._to_curobo_device(joint_pos_raw)
        joint_vel: torch.Tensor = self._to_curobo_device(joint_vel_raw)
        joint_acc: torch.Tensor = self._to_curobo_device(joint_acc_raw)

        cu_js: JointState = JointState(
            position=joint_pos,
            velocity=joint_vel,
            acceleration=joint_acc,
            joint_names=self.robot.data.joint_names,
            tensor_args=self.tensor_args,
        )
        return cu_js.get_ordered_joint_state(self.motion_gen.kinematics.joint_names)

    def get_ee_pose(self, joint_state: JointState) -> Pose:
        """Compute end-effector pose from joint configuration.

        Uses cuRobo's forward kinematics to calculate the end-effector pose
        at the specified joint configuration. Handles device conversion to ensure
        compatibility with cuRobo's CUDA-based computations.

        Args:
            joint_state: Robot joint configuration to compute end-effector pose from

        Returns:
            End-effector pose in world coordinates
        """
        # Ensure joint state is on CUDA device for cuRobo
        if isinstance(joint_state.position, torch.Tensor):
            cuda_position = self._to_curobo_device(joint_state.position)
        else:
            cuda_position = self._to_curobo_device(torch.tensor(joint_state.position))

        # Create new joint state with CUDA tensors
        cuda_joint_state = JointState(
            position=cuda_position,
            velocity=(
                self._to_curobo_device(joint_state.velocity.detach().clone())
                if joint_state.velocity is not None
                else torch.zeros_like(cuda_position)
            ),
            acceleration=(
                self._to_curobo_device(joint_state.acceleration.detach().clone())
                if joint_state.acceleration is not None
                else torch.zeros_like(cuda_position)
            ),
            joint_names=joint_state.joint_names,
            tensor_args=self.tensor_args,
        )

        kin_state: Any = self.motion_gen.rollout_fn.compute_kinematics(cuda_joint_state)
        return kin_state.ee_pose

    # =====================================================================================
    # PLANNING CORE METHODS
    # =====================================================================================

    def _make_pose(
        self,
        position: torch.Tensor | np.ndarray | list[float] | None = None,
        quaternion: torch.Tensor | np.ndarray | list[float] | None = None,
        *,
        name: str | None = None,
        normalize_rotation: bool = False,
        quat_is_xyzw: bool = True,
    ) -> Pose:
        """Create a cuRobo Pose with sensible defaults and device/dtype alignment.

        Auto-populates missing fields with identity values and ensures tensors are
        on the cuRobo device with the correct dtype. Handles quaternion format conversion
        from Isaac Lab's (x, y, z, w) to cuRobo's (w, x, y, z) format when needed.

        Args:
            position: Optional position as Tensor/ndarray/list. Defaults to [0, 0, 0].
            quaternion: Optional quaternion as Tensor/ndarray/list. Defaults to identity quaternion.
            name: Optional name of the link that this pose represents.
            normalize_rotation: Whether to normalize the quaternion inside Pose.
            quat_is_xyzw: If True, quaternion is in Isaac Lab format (x, y, z, w) and will be
                converted to cuRobo format. If False, quaternion is already in cuRobo (w, x, y, z) format.

        Returns:
            Pose: A cuRobo Pose on the configured cuRobo device and dtype.
        """
        if position is None:
            position = torch.tensor([0.0, 0.0, 0.0], dtype=self.tensor_args.dtype, device=self.tensor_args.device)
        if quaternion is None:
            quaternion_wxyz = torch.tensor(
                [1.0, 0.0, 0.0, 0.0], dtype=self.tensor_args.dtype, device=self.tensor_args.device
            )
        else:
            if not isinstance(quaternion, torch.Tensor):
                quaternion = torch.tensor(quaternion, dtype=self.tensor_args.dtype, device=self.tensor_args.device)
            else:
                quaternion = self._to_curobo_device(quaternion)

            if quat_is_xyzw:
                quaternion_wxyz = torch.roll(quaternion, shifts=1, dims=-1)
            else:
                quaternion_wxyz = quaternion

        if not isinstance(position, torch.Tensor):
            position = torch.tensor(position, dtype=self.tensor_args.dtype, device=self.tensor_args.device)
        else:
            position = self._to_curobo_device(position)

        return Pose(position=position, quaternion=quaternion_wxyz, name=name, normalize_rotation=normalize_rotation)

    def _set_active_links(self, links: list[str], active: bool) -> None:
        """Configure collision checking for specific robot links.

        Enables or disables collision sphere checking for the specified links.
        This is essential for contact scenarios where certain links (like fingers
        or attachment points) need collision checking disabled to allow contact
        with objects being grasped.

        Args:
            links: List of link names to enable or disable collision checking for
            active: True to enable collision checking, False to disable
        """
        for link in links:
            if active:
                self.motion_gen.kinematics.kinematics_config.enable_link_spheres(link)
            else:
                self.motion_gen.kinematics.kinematics_config.disable_link_spheres(link)

    def plan_motion(
        self,
        target_pose: torch.Tensor,
        step_size: float | None = None,
        enable_retiming: bool | None = None,
    ) -> bool:
        """Plan collision-free motion to target pose.

        Plans a trajectory from the current robot configuration to the specified target pose.
        The method assumes that world updates and locked joint configurations have already
        been handled. Supports optional linear retiming for consistent execution speeds.

        Args:
            target_pose: Target end-effector pose as 4x4 transformation matrix
            step_size: Step size for linear retiming, enables retiming if provided
            enable_retiming: Whether to enable linear retiming, auto-detected from step_size if None

        Returns:
            True if planning succeeded and a valid trajectory was found, False otherwise
        """
        if enable_retiming is None:
            enable_retiming = step_size is not None

        # Ensure target pose is on cuRobo device (CUDA) for device isolation
        target_pose_cuda = self._to_curobo_device(target_pose)

        target_pos: torch.Tensor
        target_rot: torch.Tensor
        target_pos, target_rot = PoseUtils.unmake_pose(target_pose_cuda)
        target_curobo_pose: Pose = self._make_pose(
            position=target_pos,
            quaternion=PoseUtils.quat_from_matrix(target_rot),
        )

        start_state: JointState = self._get_current_joint_state_for_curobo()

        self.logger.debug(f"Retiming enabled: {enable_retiming}, Step size: {step_size}")

        success: bool = self._plan_to_contact(
            start_state=start_state,
            goal_pose=target_curobo_pose,
            retreat_distance=self.config.retreat_distance,
            approach_distance=self.config.approach_distance,
            retime_plan=enable_retiming,
            step_size=step_size,
            contact=False,
        )

        # Visualize plan if enabled
        if success and self.visualize_plan and self._current_plan is not None:
            # Get current spheres for visualization
            self._sync_object_poses_with_isaaclab()
            cu_js = self._get_current_joint_state_for_curobo()
            sphere_list = self.motion_gen.kinematics.get_robot_as_spheres(cu_js.position)[0]

            # Split spheres into robot and attached object spheres
            robot_spheres = []
            attached_spheres = []
            robot_link_count = 0

            # Count robot link spheres
            robot_links = [
                link
                for link in self.robot_cfg["kinematics"]["collision_link_names"]
                if link != self.config.attached_object_link_name
            ]
            for link_name in robot_links:
                link_spheres = self.motion_gen.kinematics.kinematics_config.get_link_spheres(link_name)
                if link_spheres is not None:
                    robot_link_count += int(torch.sum(link_spheres[:, 3] > 0).item())

            # Split spheres
            for i, sphere in enumerate(sphere_list):
                if i < robot_link_count:
                    robot_spheres.append(sphere)
                else:
                    attached_spheres.append(sphere)

            # Compute end-effector positions for visualization
            ee_positions_list = []
            try:
                for i in range(len(self._current_plan.position)):
                    js: JointState = self._current_plan[i]
                    kin = self.motion_gen.compute_kinematics(js)
                    ee_pos = kin.ee_position if hasattr(kin, "ee_position") else kin.ee_pose.position
                    ee_positions_list.append(ee_pos.cpu().numpy().squeeze())

                self.logger.debug(
                    f"Link names from kinematics: {kin.link_names if len(ee_positions_list) > 0 else 'No EE positions'}"
                )

            except Exception as e:
                self.logger.debug(f"Failed to compute EE positions for visualization: {e}")
                ee_positions_list = None

            try:
                world_scene = WorldConfig.get_scene_graph(self.motion_gen.world_coll_checker.world_model)
            except Exception:
                world_scene = None

            # Visualize plan
            self.plan_visualizer.visualize_plan(
                plan=self._current_plan,
                target_pose=target_pose,
                robot_spheres=robot_spheres,
                attached_spheres=attached_spheres,
                ee_positions=np.array(ee_positions_list) if ee_positions_list else None,
                world_scene=world_scene,
            )

            # Animate EE positions over the timeline for playback
            if ee_positions_list:
                self.plan_visualizer.animate_plan(np.array(ee_positions_list))

            # Animate spheres along the path for collision visualization
            self.plan_visualizer.animate_spheres_along_path(
                plan=self._current_plan,
                robot_spheres_at_start=robot_spheres,
                attached_spheres_at_start=attached_spheres,
                timeline="sphere_animation",
                interpolation_steps=15,  # More steps for smoother animation
            )

        return success

    def _plan_to_contact_pose(
        self,
        start_state: JointState,
        goal_pose: Pose,
        contact: bool = True,
    ) -> bool:
        """Plan motion with configurable collision checking for contact scenarios.

        Plans a trajectory while optionally disabling collision checking for hand links and
        attached objects. This is crucial for grasping and placing operations where contact
        is expected and collision checking would prevent successful planning.

        Args:
            start_state: Starting joint configuration for planning
            goal_pose: Target pose to reach in cuRobo coordinate frame
            contact: True to disable hand/attached object collisions for contact planning
            retime_plan: Whether to apply linear retiming to the resulting trajectory
            step_size: Step size for retiming if retime_plan is True

        Returns:
            True if planning succeeded, False if no valid trajectory found
        """
        # Use configured hand link names instead of hardcoded ones
        disable_link_names: list[str] = self.config.hand_link_names.copy()
        link_spheres: dict[str, torch.Tensor] = {}

        # Count spheres before planning
        sphere_counts_before = self._count_active_spheres()
        self.logger.debug(
            f"Planning phase contact={contact}: Spheres before - Total: {sphere_counts_before['total']}, Robot:"
            f" {sphere_counts_before['robot_links']}, Attached: {sphere_counts_before['attached_objects']}"
        )

        if contact:
            # Store current spheres for the attached link so we can restore later
            attached_links: list[str] = list(self.attachment_links)
            for attached_link in attached_links:
                link_spheres[attached_link] = self.motion_gen.kinematics.kinematics_config.get_link_spheres(
                    attached_link
                ).clone()

            self.logger.debug(f"Attached link: {attached_links}")
            # Disable all specified links for contact planning
            self.logger.debug(f"Disable link names: {disable_link_names}")
            self._set_active_links(disable_link_names + attached_links, active=False)
        else:
            self.logger.debug(f"Disable link names: {disable_link_names}")

        # Count spheres after link disabling
        sphere_counts_after_disable = self._count_active_spheres()
        self.logger.debug(
            f"Planning phase contact={contact}: Spheres after disable - Total:"
            f" {sphere_counts_after_disable['total']}, Robot: {sphere_counts_after_disable['robot_links']},"
            f" Attached: {sphere_counts_after_disable['attached_objects']}"
        )

        planning_success = False
        try:
            result: Any = self.motion_gen.plan_single(start_state, goal_pose, self.plan_config)

            if result.success.item():
                if result.optimized_plan is not None and len(result.optimized_plan.position) != 0:
                    self._current_plan = result.optimized_plan
                    self.logger.debug(f"Using optimized plan with {len(self._current_plan.position)} waypoints")
                else:
                    self._current_plan = result.get_interpolated_plan()
                    self.logger.debug(f"Using interpolated plan with {len(self._current_plan.position)} waypoints")

                self._current_plan = self.motion_gen.get_full_js(self._current_plan)
                common_js_names: list[str] = [
                    x for x in self.robot.data.joint_names if x in self._current_plan.joint_names
                ]
                self._current_plan = self._current_plan.get_ordered_joint_state(common_js_names)
                self._plan_index = 0

                planning_success = True
                self.logger.debug(f"Contact planning succeeded with {len(self._current_plan.position)} waypoints")
            else:
                self.logger.debug(f"Contact planning failed: {result.status}")

        except Exception as e:
            self.logger.debug(f"Error during planning: {e}")

        # Always restore sphere state after planning, regardless of success
        if contact:
            self._set_active_links(disable_link_names, active=True)
            for attached_link, spheres in link_spheres.items():
                self.motion_gen.kinematics.kinematics_config.update_link_spheres(attached_link, spheres)
        return planning_success

    def _plan_to_contact(
        self,
        start_state: JointState,
        goal_pose: Pose,
        retreat_distance: float,
        approach_distance: float,
        contact: bool = False,
        retime_plan: bool = False,
        step_size: float | None = None,
    ) -> bool:
        """Execute multi-phase contact planning with approach and retreat phases.

        Implements a planning strategy for manipulation tasks that require approach and contact handling.
        Plans multiple trajectory segments with different collision checking configurations.

        Args:
            start_state: Starting joint state for planning
            goal_pose: Target pose to reach
            retreat_distance: Distance to retreat before transition to contact
            approach_distance: Distance to approach before final pose
            contact: Whether to enable contact planning mode
            retime_plan: Whether to retime the resulting plan
            step_size: Step size for retiming (only used if retime_plan is True)

        Returns:
            True if all planning phases succeeded, False if any phase failed
        """
        self.logger.debug(f"Multi-phase planning: retreat={retreat_distance}, approach={approach_distance}")

        target_poses: list[Pose] = []
        contacts: list[bool] = []

        if retreat_distance is not None and retreat_distance > 0:
            ee_pose: Pose = self.get_ee_pose(start_state)
            retreat_pose: Pose = ee_pose.multiply(
                self._make_pose(
                    position=[0.0, 0.0, -retreat_distance],
                )
            )
            target_poses.append(retreat_pose)
            contacts.append(True)
        contacts.append(contact)
        if approach_distance is not None and approach_distance > 0:
            approach_pose: Pose = goal_pose.multiply(
                self._make_pose(
                    position=[0.0, 0.0, -approach_distance],
                )
            )
            target_poses.append(approach_pose)
            contacts.append(True)

        target_poses.append(goal_pose)

        current_state: JointState = start_state
        full_plan: JointState | None = None

        for i, (target_pose, contact_flag) in enumerate(zip(target_poses, contacts)):
            self.logger.debug(
                f"Planning phase {i + 1} of {len(target_poses)}: contact={contact_flag} (collision"
                f" {'disabled' if contact_flag else 'enabled'})"
            )

            success: bool = self._plan_to_contact_pose(
                start_state=current_state,
                goal_pose=target_pose,
                contact=contact_flag,
            )

            if not success:
                self.logger.debug(f"Phase {i + 1} planning failed")
                return False

            if full_plan is None:
                full_plan = self._current_plan
            else:
                full_plan = full_plan.stack(self._current_plan)

            last_waypoint: torch.Tensor = self._current_plan.position[-1]
            current_state = JointState(
                position=last_waypoint.unsqueeze(0),
                velocity=torch.zeros_like(last_waypoint.unsqueeze(0)),
                acceleration=torch.zeros_like(last_waypoint.unsqueeze(0)),
                joint_names=self._current_plan.joint_names,
            )
            current_state = current_state.get_ordered_joint_state(self.motion_gen.kinematics.joint_names)

        self._current_plan = full_plan
        self._plan_index = 0

        if retime_plan and step_size is not None:
            original_length: int = len(self._current_plan.position)
            self._current_plan = self._linearly_retime_plan(step_size=step_size, plan=self._current_plan)
            self.logger.debug(
                f"Retimed complete plan from {original_length} to {len(self._current_plan.position)} waypoints"
            )

        self.logger.debug(f"Multi-phase planning succeeded with {len(self._current_plan.position)} total waypoints")

        return True

    def _linearly_retime_plan(
        self,
        step_size: float = 0.01,
        plan: JointState | None = None,
    ) -> JointState | None:
        """Apply linear retiming to trajectory for consistent execution speed.

        Resamples the trajectory with uniform spacing between waypoints to ensure
        consistent motion speed during execution.

        Args:
            step_size: Desired spacing between waypoints in joint space
            plan: Trajectory to retime, uses current plan if None

        Returns:
            Retimed trajectory with uniform waypoint spacing, or None if plan is invalid
        """
        if plan is None:
            plan = self._current_plan

        if plan is None or len(plan.position) == 0:
            return plan

        path = plan.position

        if len(path) <= 1:
            return plan

        deltas = path[1:] - path[:-1]
        distances = torch.linalg.norm(deltas, dim=-1)

        waypoints = [path[0]]
        for distance, waypoint in zip(distances, path[1:]):
            if distance > 1e-6:
                waypoints.append(waypoint)

        if len(waypoints) <= 1:
            return plan

        waypoints = torch.stack(waypoints)

        if len(waypoints) > 1:
            deltas = waypoints[1:] - waypoints[:-1]
            distances = torch.linalg.norm(deltas, dim=-1)
            cum_distances = torch.cat([torch.zeros(1, device=distances.device), torch.cumsum(distances, dim=0)])

        if len(waypoints) < 2 or cum_distances[-1] < 1e-6:
            return plan

        total_distance = cum_distances[-1]
        num_steps = int(torch.ceil(total_distance / step_size).item()) + 1

        # Create linearly spaced distances
        sampled_distances = torch.linspace(cum_distances[0], cum_distances[-1], num_steps, device=cum_distances.device)

        # Linear interpolation
        indices = torch.searchsorted(cum_distances, sampled_distances)
        indices = torch.clamp(indices, 1, len(cum_distances) - 1)

        # Get interpolation weights
        weights = (sampled_distances - cum_distances[indices - 1]) / (
            cum_distances[indices] - cum_distances[indices - 1]
        )
        weights = weights.unsqueeze(-1)

        # Interpolate waypoints
        sampled_waypoints = (1 - weights) * waypoints[indices - 1] + weights * waypoints[indices]

        self.logger.debug(
            f"Retiming: {len(path)} to {len(sampled_waypoints)} waypoints, "
            f"Distance: {total_distance:.3f}, Step size: {step_size}"
        )

        retimed_plan = JointState(
            position=sampled_waypoints,
            velocity=torch.zeros(
                (len(sampled_waypoints), plan.velocity.shape[-1]),
                device=plan.velocity.device,
                dtype=plan.velocity.dtype,
            ),
            acceleration=torch.zeros(
                (len(sampled_waypoints), plan.acceleration.shape[-1]),
                device=plan.acceleration.device,
                dtype=plan.acceleration.dtype,
            ),
            joint_names=plan.joint_names,
        )

        return retimed_plan

    def has_next_waypoint(self) -> bool:
        """Check if more waypoints remain in the current trajectory.

        Returns:
            True if there are unprocessed waypoints, False if trajectory is complete or empty
        """
        return self._current_plan is not None and self._plan_index < len(self._current_plan.position)

    def get_next_waypoint_ee_pose(self) -> Pose:
        """Get end-effector pose for the next waypoint in the trajectory.

        Advances the trajectory execution index and computes the end-effector pose
        for the next waypoint using forward kinematics.

        Returns:
            End-effector pose for the next waypoint in world coordinates

        Raises:
            IndexError: If no more waypoints remain in the trajectory
        """
        if not self.has_next_waypoint():
            raise IndexError("No more waypoints in the plan.")
        next_joint_state: JointState = self._current_plan[self._plan_index]
        self._plan_index += 1
        eef_state: CudaRobotModelState = self.motion_gen.compute_kinematics(next_joint_state)
        return eef_state.ee_pose

    def reset_plan(self) -> None:
        """Reset trajectory execution state.

        Clears the current trajectory and resets the execution index to zero.
        This prepares the planner for a new planning operation.
        """
        self._plan_index = 0
        self._current_plan = None
        if self.visualize_plan and hasattr(self, "plan_visualizer"):
            self.plan_visualizer.clear_visualization()
            self.plan_visualizer.mark_idle()

    def get_planned_poses(self) -> list[torch.Tensor]:
        """Extract all end-effector poses from current trajectory.

        Computes end-effector poses for all waypoints in the current trajectory without
        affecting the execution state. Optionally repeats the final pose multiple times
        if configured for stable goal reaching.

        Returns:
            List of end-effector poses as 4x4 transformation matrices, with optional repetition
        """
        if self._current_plan is None:
            return []

        # Save current execution state
        original_plan_index = self._plan_index

        # Iterate through the plan to get all poses
        planned_poses: list[torch.Tensor] = []
        self._plan_index = 0
        while self.has_next_waypoint():
            # Directly use the joint state from the plan to compute pose
            # without advancing the main plan index in get_next_waypoint_ee_pose
            next_joint_state: JointState = self._current_plan[self._plan_index]
            self._plan_index += 1  # Manually advance index for this loop
            eef_state: Any = self.motion_gen.compute_kinematics(next_joint_state)
            planned_pose: Pose | None = eef_state.ee_pose

            if planned_pose is not None:
                # Convert pose to environment device for compatibility
                position = (
                    self._to_env_device(planned_pose.position)
                    if isinstance(planned_pose.position, torch.Tensor)
                    else planned_pose.position
                )
                rotation = (
                    self._to_env_device(planned_pose.get_rotation())
                    if isinstance(planned_pose.get_rotation(), torch.Tensor)
                    else planned_pose.get_rotation()
                )
                planned_poses.append(PoseUtils.make_pose(position, rotation)[0])

        # Restore the original execution state
        self._plan_index = original_plan_index

        if self.n_repeat is not None and self.n_repeat > 0 and len(planned_poses) > 0:
            self.logger.info(f"Repeating final pose {self.n_repeat} times")
            final_pose: torch.Tensor = planned_poses[-1]
            planned_poses.extend([final_pose] * self.n_repeat)

        return planned_poses

    # =====================================================================================
    # VISUALIZATION METHODS
    # =====================================================================================

    def _update_visualization_at_joint_positions(self, joint_positions: torch.Tensor) -> None:
        """Update sphere visualization for the robot at specific joint positions.

        Args:
            joint_positions: Joint configuration to visualize collision spheres at
        """
        if not self.visualize_spheres:
            return

        self.frame_counter += 1
        if self.frame_counter % self.sphere_update_freq != 0:
            return

        original_joints: torch.Tensor = wp.to_torch(self.robot.data.joint_pos)[self.env_id].clone()

        try:
            # Ensure joint positions are on environment device for robot commands
            env_joint_positions = (
                self._to_env_device(joint_positions) if joint_positions.device != self.env.device else joint_positions
            )
            self.robot.set_joint_position_target(env_joint_positions.view(1, -1), env_ids=[self.env_id])
            self._update_sphere_visualization(force_update=False)
        finally:
            self.robot.set_joint_position_target(original_joints.unsqueeze(0), env_ids=[self.env_id])

    def _update_sphere_visualization(self, force_update: bool = True) -> None:
        """Update visual representation of robot collision spheres in USD stage.

        Creates or updates sphere primitives in the USD stage to show the robot's
        collision model. Different colors are used for robot links (green) and
        attached objects (orange) to help distinguish collision boundaries.

        Args:
            force_update: True to recreate all spheres, False to update existing positions only
        """
        # Get current sphere data
        cu_js = self._get_current_joint_state_for_curobo()
        sphere_position = self._to_curobo_device(
            cu_js.position if isinstance(cu_js.position, torch.Tensor) else torch.tensor(cu_js.position)
        )
        sphere_list = self.motion_gen.kinematics.get_robot_as_spheres(sphere_position)[0]
        robot_link_count = self._get_robot_link_sphere_count()

        # Remove existing spheres if force update or first time
        if (self.spheres is None or force_update) and self.spheres is not None:
            self._remove_existing_spheres()

        # Initialize sphere list if needed
        if self.spheres is None or force_update:
            self.spheres = []

        # Create or update all spheres
        for sphere_idx, sphere in enumerate(sphere_list):
            if not self._is_valid_sphere(sphere):
                continue

            sphere_config = self._create_sphere_config(sphere_idx, sphere, robot_link_count)
            prim_path = f"/curobo/robot_sphere_{sphere_idx}"

            # Remove old sphere if updating
            if not (self.spheres is None or force_update):
                if sphere_idx < len(self.spheres) and self.usd_helper.stage.GetPrimAtPath(prim_path).IsValid():
                    self.usd_helper.stage.RemovePrim(prim_path)

            # Spawn sphere
            spawn_mesh_sphere(prim_path=prim_path, translation=sphere_config["position"], cfg=sphere_config["cfg"])

            # Store reference if creating new
            if self.spheres is None or force_update or sphere_idx >= len(self.spheres):
                self.spheres.append((prim_path, float(sphere.radius)))

    def _get_robot_link_sphere_count(self) -> int:
        """Calculate total number of collision spheres for robot links excluding attached objects.

        Iterates through all robot collision links (excluding the attached object link) and
        counts the active collision spheres for each link. This count is used to determine
        which spheres in the visualization represent robot links vs attached objects.

        Returns:
            Total number of active collision spheres for robot links only
        """
        sphere_config = self.motion_gen.kinematics.kinematics_config
        robot_links = [
            link
            for link in self.robot_cfg["kinematics"]["collision_link_names"]
            if link != self.config.attached_object_link_name
        ]
        return sum(
            int(torch.sum(sphere_config.get_link_spheres(link_name)[:, 3] > 0).item()) for link_name in robot_links
        )

    def _remove_existing_spheres(self) -> None:
        """Remove all existing sphere visualization primitives from the USD stage.

        Iterates through all stored sphere references and removes their corresponding
        USD primitives from the stage. This is used during force updates or when
        recreating the sphere visualization from scratch.
        """
        stage = self.usd_helper.stage
        for prim_path, _ in self.spheres:
            if stage.GetPrimAtPath(prim_path).IsValid():
                stage.RemovePrim(prim_path)

    def _is_valid_sphere(self, sphere) -> bool:
        """Validate sphere data for visualization rendering.

        Checks if a sphere has valid position coordinates (no NaN values) and a positive
        radius. Invalid spheres are skipped during visualization to prevent rendering errors.

        Args:
            sphere: Sphere object containing position and radius data

        Returns:
            True if sphere has valid position and positive radius, False otherwise
        """
        pos_tensor = torch.tensor(sphere.position, dtype=torch.float32)
        return not torch.isnan(pos_tensor).any() and sphere.radius > 0

    def _create_sphere_config(self, sphere_idx: int, sphere, robot_link_count: int) -> dict:
        """Create sphere configuration with position and visual properties for USD rendering.

        Determines sphere type (robot link vs attached object), calculates world position,
        and creates the appropriate visual configuration including colors and materials.
        Robot link spheres are green with lower opacity, while attached object spheres
        are orange with higher opacity for better distinction.

        Args:
            sphere_idx: Index of the sphere in the sphere list
            sphere: Sphere object containing position and radius data
            robot_link_count: Total number of robot link spheres (for type determination)

        Returns:
            Dictionary containing 'position' (world coordinates) and 'cfg' (MeshSphereCfg)
        """

        is_attached = sphere_idx >= robot_link_count
        color = (1.0, 0.5, 0.0) if is_attached else (0.0, 1.0, 0.0)
        opacity = 0.9 if is_attached else 0.5

        # Calculate position in world frame (do not use env_origin)
        root_translation = wp.to_torch(self.robot.data.root_pos_w)[self.env_id, :3].detach().cpu().numpy()
        position = sphere.position.cpu().numpy() if hasattr(sphere.position, "cpu") else sphere.position
        if not is_attached:
            position = position + root_translation

        return {
            "position": position,
            "cfg": MeshSphereCfg(
                radius=float(sphere.radius),
                visual_material=PreviewSurfaceCfg(diffuse_color=color, opacity=opacity, emissive_color=color),
            ),
        }

    def _is_sphere_attached_object(self, sphere_index: int, sphere_config: Any) -> bool:
        """Check if a sphere belongs to attached_object link.

        Args:
            sphere_index: Index of the sphere to check
            sphere_config: Sphere configuration object

        Returns:
            True if sphere belongs to an attached object, False if it's a robot link sphere
        """
        # Get total number of robot link spheres (excluding attached_object)
        robot_links = [
            link
            for link in self.robot_cfg["kinematics"]["collision_link_names"]
            if link != self.config.attached_object_link_name
        ]

        total_robot_spheres = 0
        for link_name in robot_links:
            try:
                link_spheres = sphere_config.get_link_spheres(link_name)
                active_spheres = torch.sum(link_spheres[:, 3] > 0).item()
                total_robot_spheres += int(active_spheres)
            except Exception:
                continue

        # If sphere_index >= total_robot_spheres, it's an attached object sphere
        is_attached = sphere_index >= total_robot_spheres

        if sphere_index < 5:  # Debug first few spheres
            self.logger.debug(
                f"SPHERE {sphere_index}: total_robot_spheres={total_robot_spheres}, is_attached={is_attached}"
            )

        return is_attached

    # =====================================================================================
    # HIGH-LEVEL PLANNING INTERFACE
    # =====================================================================================

    def update_world_and_plan_motion(
        self,
        target_pose: torch.Tensor,
        expected_attached_object: str | None = None,
        env_id: int = 0,
        step_size: float | None = None,
        enable_retiming: bool | None = None,
    ) -> bool:
        """Complete planning pipeline with world updates and object attachment handling.

        Provides a high-level interface that handles the complete planning workflow:
        world synchronization, object attachment/detachment, gripper configuration,
        and motion planning.

        Args:
            target_pose: Target end-effector pose as 4x4 transformation matrix
            expected_attached_object: Name of object that should be attached, None for no attachment
            env_id: Environment ID for multi-environment setups
            step_size: Step size for linear retiming if retiming is enabled
            enable_retiming: Whether to enable linear retiming of trajectory

        Returns:
            True if complete planning pipeline succeeded, False if any step failed
        """
        # Always reset the plan before starting a new one to ensure a clean state
        self.reset_plan()

        self.logger.debug("=== MOTION PLANNING DEBUG ===")
        self.logger.debug(f"Expected attached object: {expected_attached_object}")

        self.update_world()
        gripper_closed = expected_attached_object is not None
        self._set_gripper_state(gripper_closed)
        current_attached = self.get_attached_objects()
        gripper_pos = wp.to_torch(self.robot.data.joint_pos)[env_id, -2:]

        self.logger.debug(f"Current attached objects: {current_attached}")

        # Attach object if expected but not currently attached
        if expected_attached_object and expected_attached_object not in current_attached:
            self.logger.debug(f"Need to attach {expected_attached_object}")

            object_mappings = self._get_object_mappings()

            self.logger.debug(f"Object mappings found: {list(object_mappings.keys())}")

            if expected_attached_object in object_mappings:
                expected_path = object_mappings[expected_attached_object]

                self.logger.debug(f"Object path: {expected_path}")

                # Debug object poses
                rigid_objects = self.env.scene.rigid_objects
                if expected_attached_object in rigid_objects:
                    obj = rigid_objects[expected_attached_object]
                    origin = self.env.scene.env_origins[env_id]
                    obj_pos = wp.to_torch(obj.data.root_pos_w)[env_id] - origin
                    self.logger.debug(f"Isaac Lab object position: {obj_pos}")

                    # Debug end-effector position
                    ee_frame_cfg = SceneEntityCfg("ee_frame")
                    ee_frame = self.env.scene[ee_frame_cfg.name]
                    ee_pos = wp.to_torch(ee_frame.data.target_pos_w)[env_id, 0, :] - origin
                    self.logger.debug(f"End-effector position: {ee_pos}")

                    # Debug distance
                    distance = torch.linalg.vector_norm(obj_pos - ee_pos).item()
                    self.logger.debug(f"Distance EE to object: {distance:.4f}")

                    # Debug gripper state
                    gripper_open_val = self.config.grasp_gripper_open_val
                    self.logger.debug(f"Gripper positions: {gripper_pos}")
                    self.logger.debug(f"Gripper open val: {gripper_open_val}")

                is_grasped = self._check_object_grasped(gripper_pos, expected_attached_object)

                self.logger.debug(f"Is grasped check result: {is_grasped}")

                if is_grasped:
                    self._attach_object(expected_attached_object, expected_path, env_id)
                    self.logger.debug(f"Attached {expected_attached_object}")
                else:
                    self.logger.debug(
                        "Object not detected as grasped - attachment skipped"
                    )  # This will cause collision with ghost object!
            else:
                self.logger.debug(f"Object {expected_attached_object} not found in world mappings")

        # Detach objects if no object should be attached (i.e., placing/releasing)
        if expected_attached_object is None and current_attached:
            self.logger.debug("Detaching all objects as no object expected to be attached")
            self._detach_objects()

        self.logger.debug(f"Planning motion with attached objects: {self.get_attached_objects()}")

        plan_success = self.plan_motion(target_pose, step_size, enable_retiming)

        self.logger.debug(f"Planning result: {plan_success}")
        self.logger.debug("=== END POST-GRASP DEBUG ===")

        self._detach_objects()

        return plan_success

    # =====================================================================================
    # UTILITY METHODS
    # =====================================================================================

    def _check_object_grasped(self, gripper_pos: torch.Tensor, object_name: str) -> bool:
        """Check if a specific object is currently grasped by the robot.

        Uses gripper position to determine if an object is grasped.

        Args:
            gripper_pos: Gripper position tensor
            object_name: Name of object to check (e.g., "cube_1")

        Returns:
            True if object is detected as grasped
        """
        gripper_open_val = self.config.grasp_gripper_open_val
        object_grasped = gripper_pos[0].item() < gripper_open_val

        self.logger.info(
            f"Object {object_name} is grasped: {object_grasped}"
            if object_grasped
            else f"Object {object_name} is not grasped"
        )

        return object_grasped

    def _set_gripper_state(self, has_attached_objects: bool) -> None:
        """Configure gripper joint positions based on object attachment status.

        Sets the gripper to closed position when objects are attached and open position
        when no objects are attached. This ensures proper collision checking and planning
        with the correct gripper configuration.

        Args:
            has_attached_objects: True if robot currently has attached objects requiring closed gripper
        """
        if has_attached_objects:
            # Closed gripper for grasping
            locked_joints = self.config.gripper_closed_positions
        else:
            # Open gripper for manipulation
            locked_joints = self.config.gripper_open_positions

        self.motion_gen.update_locked_joints(locked_joints, self.robot_cfg)

    def _count_active_spheres(self) -> dict[str, int]:
        """Count active collision spheres by category for debugging.

        Analyzes the current collision sphere configuration to provide detailed
        statistics about robot links vs attached object spheres. This is helpful
        for debugging collision checking issues and attachment problems.

        Returns:
            Dictionary containing sphere counts by category (total, robot_links, attached_objects)
        """
        cu_js = self._get_current_joint_state_for_curobo()

        # Ensure position tensor is on CUDA for cuRobo
        if isinstance(cu_js.position, torch.Tensor):
            sphere_position = self._to_curobo_device(cu_js.position)
        else:
            # Convert list to tensor and move to CUDA
            sphere_position = self._to_curobo_device(torch.tensor(cu_js.position))

        sphere_list = self.motion_gen.kinematics.get_robot_as_spheres(sphere_position)[0]

        # Get sphere configuration
        sphere_config = self.motion_gen.kinematics.kinematics_config

        # Count robot link spheres (excluding attached_object)
        robot_links = [
            link
            for link in self.robot_cfg["kinematics"]["collision_link_names"]
            if link != self.config.attached_object_link_name
        ]
        robot_sphere_count = 0
        for link_name in robot_links:
            if hasattr(sphere_config, "get_link_spheres"):
                link_spheres = sphere_config.get_link_spheres(link_name)
                if link_spheres is not None:
                    active_spheres = torch.sum(link_spheres[:, 3] > 0).item()
                    robot_sphere_count += int(active_spheres)

        # Count attached object spheres by checking actual sphere list
        attached_sphere_count = 0

        # Handle sphere_list as either a list or single Sphere object
        total_spheres = len(list(sphere_list))

        # Any spheres beyond robot_sphere_count are attached object spheres
        attached_sphere_count = max(0, total_spheres - robot_sphere_count)

        self.logger.debug(
            f"SPHERE COUNT: Total={total_spheres}, Robot={robot_sphere_count},Attached={attached_sphere_count}"
        )

        return {
            "total": total_spheres,
            "robot_links": robot_sphere_count,
            "attached_objects": attached_sphere_count,
        }
