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
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        sys.exit(1)


def extract_config_info(config: Any, config_name: str) -> Dict[str, Any]:
    """Extract robot configuration information into a dictionary.
    
    Args:
        config: Robot configuration object
        config_name: Name of the configuration
        
    Returns:
        Dictionary containing robot information
    """
    info = {
        'config_name': config_name,
        'prim_path': getattr(config, 'prim_path', None),
    }
    
    # Extract init_state information
    if hasattr(config, 'init_state'):
        init_state = config.init_state
        info['init_state'] = {
            'pos': list(init_state.pos) if hasattr(init_state, 'pos') else None,
            'rot': list(init_state.rot) if hasattr(init_state, 'rot') else None,
            'lin_vel': list(init_state.lin_vel) if hasattr(init_state, 'lin_vel') else None,
            'ang_vel': list(init_state.ang_vel) if hasattr(init_state, 'ang_vel') else None,
        }
    
    # Extract spawn configuration
    if hasattr(config, 'spawn'):
        spawn = config.spawn
        spawn_info = {
            'func_name': str(spawn.func) if hasattr(spawn, 'func') else None,
        }
        
        # Try to extract common spawn parameters
        for attr in ['usd_path', 'scale', 'visible', 'semantic_tags']:
            if hasattr(spawn, attr):
                value = getattr(spawn, attr)
                # Convert tuples to lists for JSON serialization
                if isinstance(value, tuple):
                    value = list(value)
                spawn_info[attr] = value
        
        info['spawn'] = spawn_info
    
    # Extract actuator information
    if hasattr(config, 'actuators'):
        actuators = config.actuators
        actuator_info = {}
        
        for key in dir(actuators):
            if not key.startswith('_'):
                actuator = getattr(actuators, key)
                if hasattr(actuator, '__dict__'):
                    actuator_data = {}
                    for attr_name in dir(actuator):
                        if not attr_name.startswith('_'):
                            try:
                                attr_value = getattr(actuator, attr_name)
                                if not callable(attr_value):
                                    # Convert to JSON-serializable format
                                    if isinstance(attr_value, (list, tuple)):
                                        actuator_data[attr_name] = list(attr_value)
                                    elif isinstance(attr_value, (str, int, float, bool, type(None))):
                                        actuator_data[attr_name] = attr_value
                                    else:
                                        actuator_data[attr_name] = str(attr_value)
                            except:
                                pass
                    
                    actuator_info[key] = actuator_data
        
        if actuator_info:
            info['actuators'] = actuator_info
    
    # Extract articulation root properties
    if hasattr(config, 'articulation_props'):
        props = config.articulation_props
        props_info = {}
        for attr in dir(props):
            if not attr.startswith('_') and not callable(getattr(props, attr)):
                try:
                    value = getattr(props, attr)
                    if isinstance(value, (str, int, float, bool, type(None))):
                        props_info[attr] = value
                    elif isinstance(value, (list, tuple)):
                        props_info[attr] = list(value)
                except:
                    pass
        
        if props_info:
            info['articulation_props'] = props_info
    
    return info


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Export robot configuration information to JSON.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Robot configuration to export (e.g., 'isaaclab_assets.CRAZYFLIE_CFG')"
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
        default=False,
        help="Pretty print JSON with indentation"
    )
    
    args = parser.parse_args()
    
    print(f"Loading configuration: {args.config}")
    config = load_config(args.config)
    
    print("Extracting robot information...")
    robot_info = extract_config_info(config, args.config)
    
    # Convert to JSON
    indent = 2 if args.pretty else None
    json_output = json.dumps(robot_info, indent=indent, default=str)
    
    # Output results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write(json_output)
        
        print(f"✓ Robot information exported to: {output_path}")
    else:
        print("\n" + "="*60)
        print("Robot Configuration Information")
        print("="*60)
        print(json_output)
        print("="*60)
    
    print(f"\n✓ Successfully extracted information for {args.config}")


if __name__ == "__main__":
    main()
