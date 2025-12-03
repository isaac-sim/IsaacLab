# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Generate robot configuration templates for new robots.

This tool creates boilerplate configuration files for new robot assets,
following IsaacLab conventions and best practices.

Usage:
    # Generate a basic robot config template
    python scripts/tools/generate_config_template.py \
        --name MyRobot \
        --type manipulator \
        --output my_robot_cfg.py
    
    # Generate with USD path
    python scripts/tools/generate_config_template.py \
        --name Quadruped \
        --type quadruped \
        --usd_path /path/to/robot.usd \
        --output quadruped_cfg.py
"""

import argparse
from pathlib import Path
from datetime import datetime


ROBOT_TYPES = {
    "manipulator": {
        "description": "Robotic arm for manipulation tasks",
        "num_joints": 7,
        "actuator_type": "implicit"
    },
    "quadruped": {
        "description": "Four-legged walking robot",
        "num_joints": 12,
        "actuator_type": "implicit"
    },
    "humanoid": {
        "description": "Humanoid robot",
        "num_joints": 21,
        "actuator_type": "implicit"
    },
    "wheeled": {
        "description": "Wheeled mobile robot",
        "num_joints": 2,
        "actuator_type": "velocity"
    },
    "aerial": {
        "description": "Flying robot (drone/quadcopter)",
        "num_joints": 4,
        "actuator_type": "velocity"
    }
}


def generate_template(robot_name: str, robot_type: str, usd_path: str = None) -> str:
    """Generate a robot configuration template.
    
    Args:
        robot_name: Name of the robot
        robot_type: Type of robot (manipulator, quadruped, etc.)
        usd_path: Optional path to USD file
        
    Returns:
        Generated configuration code as string
    """
    config_name = f"{robot_name.upper()}_CFG"
    type_info = ROBOT_TYPES.get(robot_type, ROBOT_TYPES["manipulator"])
    
    usd_line = f'usd_path="{usd_path}"' if usd_path else 'usd_path="{PLACEHOLDER_USD_PATH}"'
    
    template = f'''# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Configuration for {robot_name} robot.

Type: {robot_type}
Description: {type_info['description']}

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg


{config_name} = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        {usd_line},
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={{
            # TODO: Define initial joint positions
            # "joint_.*": 0.0,
        }},
        joint_vel={{
            ".*": 0.0,
        }},
    ),
    actuators={{
        "body": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness={{".*": 80.0}},
            damping={{".*": 4.0}},
        ),
    }},
)
"""
Configuration for {robot_name} robot - {type_info['description']}.

Estimated joints: {type_info['num_joints']}
Recommended actuator: {type_info['actuator_type']}

NOTE: This is a template. You need to:
1. Replace PLACEHOLDER_USD_PATH with actual USD file path
2. Define proper joint names and initial positions
3. Tune actuator parameters (stiffness, damping)
4. Add sensors if needed
5. Configure collision properties
"""
'''
    return template


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Generate robot configuration template.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Robot name (e.g., 'MyRobot', 'CustomArm')"
    )
    parser.add_argument(
        "--type",
        type=str,
        choices=list(ROBOT_TYPES.keys()),
        required=True,
        help=f"Robot type: {', '.join(ROBOT_TYPES.keys())}"
    )
    parser.add_argument(
        "--usd-path",
        type=str,
        default=None,
        help="Path to USD file (optional)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output Python file path"
    )
    
    args = parser.parse_args()
    
    # Generate template
    print(f"Generating {args.type} robot configuration for '{args.name}'...")
    template_code = generate_template(args.name, args.type, args.usd_path)
    
    # Write to file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(template_code)
    
    print(f"[SUCCESS] Configuration template generated: {output_path}")
    print(f"\nNext steps:")
    print(f"1. Edit {output_path} and replace placeholders")
    print(f"2. Add proper joint names and configurations")
    print(f"3. Tune actuator parameters for your robot")
    print(f"4. Test with: ./isaaclab.sh -p your_script.py")


if __name__ == "__main__":
    main()
