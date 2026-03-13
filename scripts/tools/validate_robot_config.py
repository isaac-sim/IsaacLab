# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""
Utility script to validate robot configuration files before running simulations.

This tool checks if robot configuration dictionaries or config objects have all required
fields and valid values, helping catch configuration errors early in development.

Usage:
    # Validate a robot config from Python module
    ./isaaclab.sh -p scripts/tools/validate_robot_config.py --config isaaclab_assets.CRAZYFLIE_CFG
    
    # Check specific fields
    ./isaaclab.sh -p scripts/tools/validate_robot_config.py --config isaaclab_assets.ANYMAL_D_CFG --verbose
"""

import argparse
import importlib
import sys
from typing import Any


def validate_config(config: Any, config_name: str, verbose: bool = False) -> tuple[bool, list[str]]:
    """Validate a robot configuration object.
    
    Args:
        config: The configuration object to validate
        config_name: Name of the configuration for error messages
        verbose: Whether to print detailed information
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    warnings = []
    
    # Check if config has basic required attributes
    required_attrs = ['prim_path']
    recommended_attrs = ['spawn', 'init_state', 'actuators']
    
    print(f"\n{'='*60}")
    print(f"Validating configuration: {config_name}")
    print(f"{'='*60}\n")
    
    # Required attributes check
    for attr in required_attrs:
        if not hasattr(config, attr):
            errors.append(f"Missing required attribute: {attr}")
        elif verbose:
            print(f"[OK] Required attribute '{attr}' present")
    
    # Recommended attributes check
    for attr in recommended_attrs:
        if not hasattr(config, attr):
            warnings.append(f"Missing recommended attribute: {attr}")
        elif verbose:
            print(f"[OK] Recommended attribute '{attr}' present")
    
    # Additional validation checks
    if hasattr(config, 'init_state'):
        init_state = config.init_state
        
        # Check for position
        if not hasattr(init_state, 'pos'):
            warnings.append("init_state missing 'pos' attribute")
        elif verbose:
            print(f"[OK] init_state.pos present: {init_state.pos}")
        
        # Check for rotation
        if not hasattr(init_state, 'rot'):
            warnings.append("init_state missing 'rot' attribute")
        elif verbose:
            print(f"[OK] init_state.rot present: {init_state.rot}")
    
    if hasattr(config, 'spawn'):
        spawn = config.spawn
        if not hasattr(spawn, 'func'):
            errors.append("spawn missing 'func' attribute")
        elif verbose:
            print(f"[OK] spawn.func present: {spawn.func}")
    
    # Print summary
    print(f"\n{'='*60}")
    if len(errors) == 0 and len(warnings) == 0:
        print(f"[SUCCESS] Configuration is valid! No errors or warnings found.")
        print(f"{'='*60}\n")
        return True, []
    elif len(errors) == 0:
        print(f"[WARNING] Found {len(warnings)} warning(s):")
        for warning in warnings:
            print(f"  - {warning}")
    else:
        print(f"[ERROR] Found {len(errors)} error(s):")
        for error in errors:
            print(f"  - {error}")
        print(f"\n[ERROR] Configuration validation FAILED!")
    
    if len(warnings) > 0 and len(errors) == 0:
        print(f"[SUCCESS] Configuration is valid! (with warnings)")
    
    print(f"{'='*60}\n")
    
    return len(errors) == 0, errors


def main():
    """Main function to parse arguments and validate configuration."""
    parser = argparse.ArgumentParser(description="Validate robot configuration files")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Config module path (e.g., 'isaaclab_assets.CRAZYFLIE_CFG')"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Print detailed validation information"
    )
    
    args = parser.parse_args()
    
    # Parse module and config name
    try:
        module_path, config_attr = args.config.rsplit('.', 1)
        module = importlib.import_module(module_path)
        config = getattr(module, config_attr)
    except Exception as e:
        print(f"[ERROR] Error importing module: {e}")
        sys.exit(1)
    
    try:
        is_valid, errors = validate_config(config, args.config, args.verbose)
        sys.exit(0 if is_valid else 1)
    except Exception as e:
        print(f"[ERROR] Error loading config: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
