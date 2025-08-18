# Copyright (c) 2024-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import WorldConfig
from curobo.util_file import get_world_configs_path, join_path, load_yaml


@dataclass
class CuroboPlannerConfig:
    """Configuration for CuRobo motion planner."""

    # Robot configuration
    robot_config_file: str = "franka.yml"
    """cuRobo robot configuration file (path defined by curobo api)."""

    robot_name: str = "franka"
    """Robot name for visualization and identification."""

    ee_link_name: str | None = None
    """End-effector link name (auto-detected from robot config if None)."""

    # Gripper configuration
    gripper_joint_names: list[str] = field(default_factory=lambda: ["panda_finger_joint1", "panda_finger_joint2"])
    """Names of gripper joints."""

    gripper_open_positions: dict[str, float] = field(
        default_factory=lambda: {"panda_finger_joint1": 0.04, "panda_finger_joint2": 0.04}
    )
    """Open gripper positions for cuRobo to update spheres"""

    gripper_closed_positions: dict[str, float] = field(
        default_factory=lambda: {"panda_finger_joint1": 0.008, "panda_finger_joint2": 0.008}
    )
    """Closed gripper positions for cuRobo to update spheres"""

    # Hand link configuration (for contact planning)
    hand_link_names: list[str] = field(default_factory=lambda: ["panda_leftfinger", "panda_rightfinger", "panda_hand"])
    """Names of hand/finger links to disable during contact planning."""

    # Attachment configuration
    attached_object_link_name: str = "attached_object"
    """Name of the link used for attaching objects."""

    # World configuration
    world_config_file: str = "collision_table.yml"
    """CuRobo world configuration file (without path)."""

    # Motion planning parameters
    collision_checker_type: CollisionCheckerType = CollisionCheckerType.MESH
    """Type of collision checker to use."""

    num_trajopt_seeds: int = 12
    """Number of seeds for trajectory optimization."""

    num_graph_seeds: int = 12
    """Number of seeds for graph search."""

    interpolation_dt: float = 0.05
    """Time step for interpolating waypoints."""

    collision_cache_size: dict[str, int] = field(default_factory=lambda: {"obb": 150, "mesh": 150})
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

    extra_collision_spheres: dict[str, int] = field(default_factory=lambda: {"attached_object": 100})
    """Extra collision spheres for attached objects."""

    position_threshold: float = 0.005
    """Position threshold for motion planning."""

    rotation_threshold: float = 0.05
    """Rotation threshold for motion planning."""

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

        # Adjust table height if cuboid exists
        if world_cfg_table.cuboid is not None:
            if len(world_cfg_table.cuboid) > 0:
                world_cfg_table.cuboid[0].pose[2] -= 0.02

        # Get mesh world for additional collision objects
        world_cfg_mesh = WorldConfig.from_dict(
            load_yaml(join_path(get_world_configs_path(), self.world_config_file))
        ).get_mesh_world()

        # Adjust mesh configuration if it exists
        if world_cfg_mesh.mesh is not None:
            if len(world_cfg_mesh.mesh) > 0:
                world_cfg_mesh.mesh[0].name += "_mesh"
                world_cfg_mesh.mesh[0].pose[2] = -10.5  # Move mesh below scene

        # Combine cuboid and mesh worlds
        world_cfg = WorldConfig(cuboid=world_cfg_table.cuboid, mesh=world_cfg_mesh.mesh)
        return world_cfg

    @classmethod
    def franka_config(cls) -> "CuroboPlannerConfig":
        """Create configuration for Franka Panda robot."""
        return cls(
            robot_config_file="franka.yml",
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
            enable_finetune_trajopt=False,
            n_repeat=None,
            motion_step_size=None,
            visualize_spheres=False,
            visualize_plan=False,
            debug_planner=False,
            sphere_update_freq=5,
            motion_noise_scale=0.02,
        )

    @classmethod
    def franka_stack_cube_bin_config(cls) -> "CuroboPlannerConfig":
        """Create configuration for Franka stacking cube in a bin."""
        config = cls.franka_config()
        config.gripper_closed_positions = {"panda_finger_joint1": 0.024, "panda_finger_joint2": 0.024}
        config.grasp_gripper_open_val = 0.04
        config.approach_distance = 0.05
        config.retreat_distance = 0.07
        config.surface_sphere_radius = 0.01
        config.debug_planner = True
        config.collision_activation_distance = 0.02
        config.visualize_plan = True
        config.enable_finetune_trajopt = True
        config.motion_noise_scale = 0.02
        config.get_world_config = lambda: config._get_world_config_with_table_adjustment()
        return config

    @classmethod
    def franka_stack_square_nut_config(cls) -> "CuroboPlannerConfig":
        """Create configuration for Franka stacking a square nut."""
        config = cls.franka_config()
        config.gripper_closed_positions = {"panda_finger_joint1": 0.021, "panda_finger_joint2": 0.021}
        config.grasp_gripper_open_val = 0.04
        config.approach_distance = 0.11
        config.retreat_distance = 0.11
        config.extra_collision_spheres = {"attached_object": 200}
        config.surface_sphere_radius = 0.005
        config.n_repeat = None
        config.motion_step_size = None
        config.visualize_spheres = False
        config.visualize_plan = True
        config.debug_planner = True
        config.motion_noise_scale = 0.0
        config.time_dilation_factor = 0.4
        config.get_world_config = lambda: config._get_world_config_with_table_adjustment()
        return config

    @classmethod
    def franka_stack_cube_config(cls) -> "CuroboPlannerConfig":
        """Create configuration for Franka stacking a normal cube."""
        config = cls.franka_config()
        config.n_repeat = None
        config.motion_step_size = None
        config.visualize_spheres = False
        config.visualize_plan = False
        config.debug_planner = True
        config.motion_noise_scale = 0.0
        config.motion_step_size = None
        config.n_repeat = None
        config.collision_activation_distance = 0.01
        config.approach_distance = 0.05
        config.retreat_distance = 0.05
        config.get_world_config = lambda: config._get_world_config_with_table_adjustment()
        return config

    @classmethod
    def from_task_name(cls, task_name: str) -> "CuroboPlannerConfig":
        """Create configuration from task name.

        Args:
            task_name: Task name (e.g., "Isaac-Stack-Cube-Bin-Franka-v0")

        Returns:
            CuroboPlannerConfig: Configuration for the specified task
        """
        task_lower = task_name.lower()

        if "stack-cube-bin" in task_lower:
            return cls.franka_stack_cube_bin_config()
        elif "stack-square-nut" in task_lower:
            return cls.franka_stack_square_nut_config()
        elif "stack-cube" in task_lower:
            return cls.franka_stack_cube_config()
        else:
            # Default to Franka configuration
            print(f"Warning: Unknown robot in task '{task_name}', using Franka configuration")
            return cls.franka_config()
