# Copyright (c) 2024-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
import yaml

from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import WorldConfig
from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml

from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, retrieve_file_path
from isaaclab.utils.configclass import configclass


@configclass
class CuroboPlannerCfg:
    """Configuration for CuRobo motion planner.

    This dataclass provides a flexible configuration system for the CuRobo motion planner.
    The base configuration is robot-agnostic, with factory methods providing pre-configured
    settings for specific robots and tasks.

    Example Usage:
        >>> # Use a pre-configured robot
        >>> config = CuroboPlannerCfg.franka_config()
        >>>
        >>> # Or create from task name
        >>> config = CuroboPlannerCfg.from_task_name("Isaac-Stack-Cube-Franka-v0")
        >>>
        >>> # Initialize planner with config
        >>> planner = CuroboPlanner(env, robot, config)

    To add support for a new robot, see the factory methods section below for detailed instructions.
    """

    # Robot configuration
    robot_config_file: str | None = None
    """cuRobo robot configuration file (path defined by curobo api)."""

    robot_name: str = ""
    """Robot name for visualization and identification."""

    ee_link_name: str | None = None
    """End-effector link name (auto-detected from robot config if None)."""

    # Gripper configuration
    gripper_joint_names: list[str] = []
    """Names of gripper joints."""

    gripper_open_positions: dict[str, float] = {}
    """Open gripper positions for cuRobo to update spheres"""

    gripper_closed_positions: dict[str, float] = {}
    """Closed gripper positions for cuRobo to update spheres"""

    # Hand link configuration (for contact planning)
    hand_link_names: list[str] = []
    """Names of hand/finger links to disable during contact planning."""

    # Attachment configuration
    attached_object_link_name: str = "attached_object"
    """Name of the link used for attaching objects."""

    # World configuration
    world_config_file: str = "collision_table.yml"
    """CuRobo world configuration file (without path)."""

    # Static objects to not update in the world model
    static_objects: list[str] = []
    """Names of static objects to not update in the world model."""

    # Optional prim path configuration
    robot_prim_path: str | None = None
    """Absolute USD prim path to the robot root for world extraction; None derives it from environment root."""

    world_ignore_substrings: list[str] | None = None
    """List of substring patterns to ignore when extracting world obstacles (e.g., default ground plane, debug prims)."""

    # Motion planning parameters
    collision_checker_type: CollisionCheckerType = CollisionCheckerType.MESH
    """Type of collision checker to use."""

    num_trajopt_seeds: int = 12
    """Number of seeds for trajectory optimization."""

    num_graph_seeds: int = 12
    """Number of seeds for graph search."""

    interpolation_dt: float = 0.05
    """Time step for interpolating waypoints."""

    collision_cache_size: dict[str, int] = {"obb": 150, "mesh": 150}
    """Cache sizes for different collision types."""

    trajopt_tsteps: int = 32
    """Number of trajectory optimization time steps."""

    collision_activation_distance: float = 0.0
    """Distance at which collision constraints are activated."""

    approach_distance: float = 0.05
    """Distance to approach at the end of the plan."""

    retreat_distance: float = 0.05
    """Distance to retreat at the start of the plan."""

    grasp_gripper_open_val: float = 0.04
    """Gripper joint value when considered open for grasp detection."""

    # Planning configuration
    enable_graph: bool = True
    """Whether to enable graph-based planning."""

    enable_graph_attempt: int = 5
    """Number of graph planning attempts."""

    max_planning_attempts: int = 15
    """Maximum number of planning attempts."""

    enable_finetune_trajopt: bool = True
    """Whether to enable trajectory optimization fine-tuning."""

    time_dilation_factor: float = 1.0
    """Time dilation factor for planning."""

    surface_sphere_radius: float = 0.005
    """Radius of surface spheres for collision checking."""

    # Debug and visualization
    n_repeat: int | None = None
    """Number of times to repeat final waypoint for stabilization. If None, no repetition."""

    motion_step_size: float | None = None
    """Step size (in radians) for retiming motion plans. If None, no retiming."""

    visualize_spheres: bool = False
    """Visualize robot collision spheres. Note: only works for env 0."""

    visualize_plan: bool = False
    """Visualize motion plan in Rerun. Note: only works for env 0."""

    debug_planner: bool = False
    """Enable detailed motion planning debug information."""

    sphere_update_freq: int = 5
    """Frequency to update sphere visualization, specified in number of frames."""

    motion_noise_scale: float = 0.0
    """Scale of Gaussian noise to add to the planned waypoints. Defaults to 0.0 (no noise)."""

    # Collision sphere configuration
    collision_spheres_file: str | None = None
    """Collision spheres configuration file (auto-detected if None)."""

    extra_collision_spheres: dict[str, int] = {"attached_object": 100}
    """Extra collision spheres for attached objects."""

    position_threshold: float = 0.005
    """Position threshold for motion planning."""

    rotation_threshold: float = 0.05
    """Rotation threshold for motion planning."""

    cuda_device: int | None = 0
    """Preferred CUDA device index; None uses torch.cuda.current_device() (respects CUDA_VISIBLE_DEVICES)."""

    def get_world_config(self) -> WorldConfig:
        """Load and prepare the world configuration.

        This method can be overridden in subclasses or customized per task
        to provide different world configuration setups.

        Returns:
            WorldConfig: The configured world for collision checking
        """
        # Default implementation: just load the world config file
        world_cfg = WorldConfig.from_dict(load_yaml(join_path(get_world_configs_path(), self.world_config_file)))
        return world_cfg

    def _get_world_config_with_table_adjustment(self) -> WorldConfig:
        """Load world config with standard table adjustments.

        This is a helper method that implements the common pattern of adjusting
        table height and combining mesh/cuboid worlds. Used by specific task configs.

        Returns:
            WorldConfig: World configuration with adjusted table
        """
        # Load the base world config
        world_cfg_table = WorldConfig.from_dict(load_yaml(join_path(get_world_configs_path(), self.world_config_file)))

        # Adjust table height if cuboid exists and has a pose
        if world_cfg_table.cuboid and len(world_cfg_table.cuboid) > 0 and world_cfg_table.cuboid[0].pose:
            world_cfg_table.cuboid[0].pose[2] -= 0.02

        # Get mesh world for additional collision objects
        world_cfg_mesh = WorldConfig.from_dict(
            load_yaml(join_path(get_world_configs_path(), self.world_config_file))
        ).get_mesh_world()

        # Adjust mesh configuration if it exists
        if world_cfg_mesh.mesh and len(world_cfg_mesh.mesh) > 0:
            mesh_obj = world_cfg_mesh.mesh[0]
            if mesh_obj.name:
                mesh_obj.name += "_mesh"
            if mesh_obj.pose:
                mesh_obj.pose[2] = -10.5  # Move mesh below scene

        # Combine cuboid and mesh worlds
        world_cfg = WorldConfig(cuboid=world_cfg_table.cuboid, mesh=world_cfg_mesh.mesh)
        return world_cfg

    @classmethod
    def _create_temp_robot_yaml(cls, base_yaml: str, urdf_path: str) -> str:
        """Create a temporary robot configuration YAML with custom URDF path.

        Args:
            base_yaml: Base robot configuration file name
            urdf_path: Absolute path to the URDF file

        Returns:
            Path to the temporary YAML file

        Raises:
            FileNotFoundError: If the URDF file doesn't exist
        """
        # Validate URDF path
        if not os.path.isabs(urdf_path) or not os.path.isfile(urdf_path):
            raise FileNotFoundError(f"URDF must be a local file: {urdf_path}")

        # Load base configuration
        robot_cfg_path = get_robot_configs_path()
        base_path = join_path(robot_cfg_path, base_yaml)
        data = load_yaml(base_path)
        print(f"urdf_path: {urdf_path}")
        # Update URDF path
        data["robot_cfg"]["kinematics"]["urdf_path"] = urdf_path

        # Write to temporary file
        tmp_dir = tempfile.mkdtemp(prefix="curobo_robot_cfg_")
        out_path = os.path.join(tmp_dir, base_yaml)
        with open(out_path, "w") as f:
            yaml.safe_dump(data, f, sort_keys=False)

        return out_path

    # =====================================================================================
    # FACTORY METHODS FOR ROBOT CONFIGURATIONS
    # =====================================================================================
    """
    Creating Custom Robot Configurations
    =====================================

    To create a configuration for your own robot, follow these steps:

    1. Create a Factory Method
    ---------------------------
    Define a classmethod that returns a configured instance:

    .. code-block:: python

        @classmethod
        def my_robot_config(cls) -> "CuroboPlannerCfg":
            # Option 1: Download from Nucleus (like Franka example)
            urdf_path = f"{ISAACLAB_NUCLEUS_DIR}/path/to/my_robot.urdf"
            local_urdf = retrieve_file_path(urdf_path, force_download=True)

            # Option 2: Use local file directly
            # local_urdf = "/absolute/path/to/my_robot.urdf"

            # Create temporary YAML with custom URDF path
            robot_cfg_file = cls._create_temp_robot_yaml("my_robot.yml", local_urdf)

            return cls(
                # Required: Specify robot configuration file
                robot_config_file=robot_cfg_file,  # Use the generated YAML with custom URDF
                robot_name="my_robot",

                # Gripper configuration (if robot has grippers)
                gripper_joint_names=["gripper_left", "gripper_right"],
                gripper_open_positions={"gripper_left": 0.05, "gripper_right": 0.05},
                gripper_closed_positions={"gripper_left": 0.01, "gripper_right": 0.01},

                # Hand/finger links to disable during contact planning
                hand_link_names=["finger_link_1", "finger_link_2", "palm_link"],

                # Optional: Absolute USD prim path to the robot root for world extraction; None derives it from environment root.
                robot_prim_path=None,

                # Optional: List of substring patterns to ignore when extracting world obstacles (e.g., default ground plane, debug prims).
                # None derives it from the environment root and adds some default patterns. This is useful for environments with a lot of prims.
                world_ignore_substrings=None,

                # Optional: Custom collision spheres configuration
                collision_spheres_file="spheres/my_robot_spheres.yml",  # Path relative to curobo (can override with custom spheres file)

                # Grasp detection threshold
                grasp_gripper_open_val=0.05,

                # Motion planning parameters (tune for your robot)
                approach_distance=0.05,  # Distance to approach before grasping
                retreat_distance=0.05,   # Distance to retreat after grasping
                time_dilation_factor=0.5,  # Speed factor (0.5 = half speed)

                # Visualization options
                visualize_spheres=False,
                visualize_plan=False,
                debug_planner=False,
            )

    2. Task-Specific Configurations
    --------------------------------
    For task-specific variants, create methods that modify the base config:

    .. code-block:: python

        @classmethod
        def my_robot_pick_place_config(cls) -> "CuroboPlannerCfg":
            config = cls.my_robot_config()  # Start from base config

            # Override for pick-and-place tasks
            config.approach_distance = 0.08
            config.retreat_distance = 0.10
            config.enable_finetune_trajopt = True
            config.collision_activation_distance = 0.02

            # Custom world configuration if needed
            config.get_world_config = lambda: config._get_world_config_with_table_adjustment()

            return config

    3. Register in from_task_name()
    --------------------------------
    Add your robot detection logic to the from_task_name method:

    .. code-block:: python

        @classmethod
        def from_task_name(cls, task_name: str) -> "CuroboPlannerCfg":
            task_lower = task_name.lower()

            # Add your robot detection
            if "my-robot" in task_lower:
                if "pick-place" in task_lower:
                    return cls.my_robot_pick_place_config()
                else:
                    return cls.my_robot_config()

            # ... existing robot checks ...

    Important Notes
    ---------------
    - The _create_temp_robot_yaml() helper creates a temporary YAML with your custom URDF
    - If using Nucleus assets, retrieve_file_path() downloads them to a local temp directory
    - The base robot YAML (e.g., "my_robot.yml") should exist in cuRobo's robot configs

    Best Practices
    --------------
    1. Start with conservative parameters (slow speed, large distances)
    2. Test with visualization enabled (visualize_plan=True) for debugging
    3. Tune collision_activation_distance based on controller precision to follow collision-free motion
    4. Adjust sphere counts in extra_collision_spheres for attached objects
    5. Use debug_planner=True when developing new configurations
    """

    @classmethod
    def franka_config(cls) -> "CuroboPlannerCfg":
        """Create configuration for Franka Panda robot.

        This method uses a custom URDF from Nucleus for the Franka robot.

        Returns:
            CuroboPlannerCfg: Configuration for Franka robot
        """
        urdf_path = f"{ISAACLAB_NUCLEUS_DIR}/Controllers/SkillGenAssets/FrankaPanda/franka_panda.urdf"
        local_urdf = retrieve_file_path(urdf_path, force_download=True)

        robot_cfg_file = cls._create_temp_robot_yaml("franka.yml", local_urdf)

        return cls(
            robot_config_file=robot_cfg_file,
            robot_name="franka",
            gripper_joint_names=["panda_finger_joint1", "panda_finger_joint2"],
            gripper_open_positions={"panda_finger_joint1": 0.04, "panda_finger_joint2": 0.04},
            gripper_closed_positions={"panda_finger_joint1": 0.023, "panda_finger_joint2": 0.023},
            hand_link_names=["panda_leftfinger", "panda_rightfinger", "panda_hand"],
            collision_spheres_file="spheres/franka_mesh.yml",
            grasp_gripper_open_val=0.04,
            approach_distance=0.0,
            retreat_distance=0.0,
            max_planning_attempts=1,
            time_dilation_factor=0.6,
            enable_finetune_trajopt=True,
            n_repeat=None,
            motion_step_size=None,
            visualize_spheres=False,
            visualize_plan=False,
            debug_planner=False,
            sphere_update_freq=5,
            motion_noise_scale=0.02,
            # World extraction tuning for Franka envs
            world_ignore_substrings=["/World/defaultGroundPlane", "/curobo"],
        )

    @classmethod
    def franka_stack_cube_bin_config(cls) -> "CuroboPlannerCfg":
        """Create configuration for Franka stacking cube in a bin."""
        config = cls.franka_config()
        config.static_objects = ["bin", "table"]
        config.gripper_closed_positions = {"panda_finger_joint1": 0.024, "panda_finger_joint2": 0.024}
        config.approach_distance = 0.05
        config.retreat_distance = 0.07
        config.surface_sphere_radius = 0.01
        config.debug_planner = False
        config.collision_activation_distance = 0.02
        config.visualize_plan = False
        config.enable_finetune_trajopt = True
        config.motion_noise_scale = 0.02
        config.get_world_config = lambda: config._get_world_config_with_table_adjustment()
        return config

    @classmethod
    def franka_stack_cube_config(cls) -> "CuroboPlannerCfg":
        """Create configuration for Franka stacking a normal cube."""
        config = cls.franka_config()
        config.static_objects = ["table"]
        config.visualize_plan = False
        config.debug_planner = False
        config.motion_noise_scale = 0.02
        config.collision_activation_distance = 0.01
        config.approach_distance = 0.05
        config.retreat_distance = 0.05
        config.surface_sphere_radius = 0.01
        config.get_world_config = lambda: config._get_world_config_with_table_adjustment()
        return config

    @classmethod
    def from_task_name(cls, task_name: str) -> "CuroboPlannerCfg":
        """Create configuration from task name.

        Args:
            task_name: Task name (e.g., "Isaac-Stack-Cube-Bin-Franka-v0")

        Returns:
            CuroboPlannerCfg: Configuration for the specified task
        """
        task_lower = task_name.lower()

        if "stack-cube-bin" in task_lower:
            return cls.franka_stack_cube_bin_config()
        elif "stack-cube" in task_lower:
            return cls.franka_stack_cube_config()
        else:
            # Default to Franka configuration
            print(f"Warning: Unknown robot in task '{task_name}', using Franka configuration")
            return cls.franka_config()
