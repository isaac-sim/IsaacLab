# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""
Utility to export robot configuration and properties to JSON format.

This tool extracts detailed information about robot configurations including
joint names, limits, actuators, and other properties for documentation or analysis.

Usage:
    # Export robot info to console
    ./isaaclab.sh -p scripts/tools/export_robot_info.py --config isaaclab_assets.CRAZYFLIE_CFG
    
    # Export to JSON file
    ./isaaclab.sh -p scripts/tools/export_robot_info.py --config isaaclab_assets.ANYMAL_D_CFG --output robot_info.json
    
    # Pretty print with indentation
    ./isaaclab.sh -p scripts/tools/export_robot_info.py --config isaaclab_assets.FRANKA_PANDA_CFG --output franka.json --pretty
"""

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Dict


def load_config(config_path: str) -> Any:
    """Load configuration from module path."""
    try:
        parts = config_path.rsplit('.', 1)
        if len(parts) != 2:
            raise ValueError(f"Config must be 'module.CONFIG_NAME', got: {config_path}")
        
        module_name, config_name = parts
        module = importlib.import_module(module_name)
        return getattr(module, config_name)
    except (ImportError, AttributeError) as e:
        print(f"[ERROR] Error loading config: {e}")
        sys.exit(1)


def extract_actuator_info(actuator_cfg: Any) -> Dict[str, Any]:
    """Extract actuator configuration details."""
    actuator_info = {
        "class": actuator_cfg.__class__.__name__,
    }
    
    # Add common actuator properties if they exist
    if hasattr(actuator_cfg, 'joint_names_expr'):
        actuator_info['joint_names_expr'] = actuator_cfg.joint_names_expr
    if hasattr(actuator_cfg, 'effort_limit'):
        actuator_info['effort_limit'] = actuator_cfg.effort_limit
    if hasattr(actuator_cfg, 'velocity_limit'):
        actuator_info['velocity_limit'] = actuator_cfg.velocity_limit
    if hasattr(actuator_cfg, 'stiffness'):
        actuator_info['stiffness'] = actuator_cfg.stiffness
    if hasattr(actuator_cfg, 'damping'):
        actuator_info['damping'] = actuator_cfg.damping
    if hasattr(actuator_cfg, 'armature'):
        actuator_info['armature'] = actuator_cfg.armature
    
    return actuator_info


def extract_robot_info(robot_cfg: Any) -> Dict[str, Any]:
    """Extract detailed information from robot configuration."""
    info = {
        "config_name": robot_cfg.__class__.__name__,
        "spawn": {
            "class": robot_cfg.spawn.__class__.__name__,
        },
    }
    
    # Extract spawn configuration details
    if hasattr(robot_cfg.spawn, 'usd_path'):
        info['spawn']['usd_path'] = str(robot_cfg.spawn.usd_path)
    if hasattr(robot_cfg.spawn, 'rigid_props'):
        if robot_cfg.spawn.rigid_props is not None:
            info['spawn']['rigid_props'] = {
                k: v for k, v in vars(robot_cfg.spawn.rigid_props).items() 
                if not k.startswith('_')
            }
    if hasattr(robot_cfg.spawn, 'articulation_props'):
        if robot_cfg.spawn.articulation_props is not None:
            info['spawn']['articulation_props'] = {
                k: v for k, v in vars(robot_cfg.spawn.articulation_props).items()
                if not k.startswith('_')
            }
    
    # Extract init state if available
    if hasattr(robot_cfg, 'init_state'):
        init_state_dict = {}
        for key, value in vars(robot_cfg.init_state).items():
            if not key.startswith('_'):
                # Convert numpy arrays or tensors to lists for JSON serialization
                if hasattr(value, 'tolist'):
                    init_state_dict[key] = value.tolist()
                else:
                    init_state_dict[key] = value
        info['init_state'] = init_state_dict
    
    # Extract actuators
    if hasattr(robot_cfg, 'actuators'):
        actuators_info = {}
        for name, actuator_cfg in robot_cfg.actuators.items():
            actuators_info[name] = extract_actuator_info(actuator_cfg)
        info['actuators'] = actuators_info
    
    # Extract soft joint position limits if available
    if hasattr(robot_cfg, 'soft_joint_pos_limit_factor'):
        info['soft_joint_pos_limit_factor'] = robot_cfg.soft_joint_pos_limit_factor
    
    return info


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Export robot configuration to JSON format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Export to console
    python export_robot_info.py --config isaaclab_assets.CRAZYFLIE_CFG
    
    # Export to file
    python export_robot_info.py --config isaaclab_assets.ANYMAL_D_CFG --output robot_info.json
    
    # Pretty print
    python export_robot_info.py --config isaaclab_assets.FRANKA_PANDA_CFG --output franka.json --pretty
"""
    )
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Robot configuration path (e.g., 'isaaclab_assets.FRANKA_PANDA_CFG')"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path (prints to console if not specified)"
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty print JSON with indentation"
    )
    
    args = parser.parse_args()
    
    # Load the robot configuration
    robot_cfg = load_config(args.config)
    
    # Extract information
    robot_info = extract_robot_info(robot_cfg)
    
    # Serialize to JSON
    indent = 2 if args.pretty else None
    json_output = json.dumps(robot_info, indent=indent, default=str)
    
    # Output results
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(json_output)
        print(f"[SUCCESS] Robot information exported to: {output_path}")
    else:
        print(json_output)
    
    if not args.output:
        print(f"\n[SUCCESS] Successfully extracted information for {args.config}")


if __name__ == "__main__":
    main()
