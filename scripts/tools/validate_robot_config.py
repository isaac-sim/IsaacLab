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
    
    # Check required attributes
    for attr in required_attrs:
        if not hasattr(config, attr):
            errors.append(f"Missing required attribute: '{attr}'")
        elif verbose:
            print(f"✓ Required attribute '{attr}': {getattr(config, attr)}")
    
    # Check recommended attributes
    for attr in recommended_attrs:
        if not hasattr(config, attr):
            warnings.append(f"Missing recommended attribute: '{attr}'")
        elif verbose:
            print(f"✓ Recommended attribute '{attr}': present")
    
    # Validate prim_path format
    if hasattr(config, 'prim_path'):
        prim_path = config.prim_path
        if not isinstance(prim_path, str):
            errors.append(f"prim_path must be string, got {type(prim_path).__name__}")
        elif not prim_path.startswith('/'):
            errors.append(f"prim_path must start with '/', got: {prim_path}")
        elif ' ' in prim_path:
            errors.append(f"prim_path should not contain spaces: {prim_path}")
    
    # Validate init_state if present
    if hasattr(config, 'init_state'):
        init_state = config.init_state
        if hasattr(init_state, 'pos'):
            pos = init_state.pos
            if hasattr(pos, '__len__') and len(pos) != 3:
                errors.append(f"init_state.pos must have 3 elements, got {len(pos)}")
            elif verbose:
                print(f"✓ init_state.pos: {pos}")
        
        if hasattr(init_state, 'rot'):
            rot = init_state.rot
            if hasattr(rot, '__len__') and len(rot) != 4:
                errors.append(f"init_state.rot must have 4 elements (quaternion), got {len(rot)}")
            elif verbose:
                print(f"✓ init_state.rot: {rot}")
    
    # Validate spawn configuration if present
    if hasattr(config, 'spawn'):
        spawn = config.spawn
        if hasattr(spawn, 'func'):
            if verbose:
                print(f"✓ spawn.func: {spawn.func}")
        else:
            warnings.append("spawn config missing 'func' attribute")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Validation Summary for {config_name}")
    print(f"{'='*60}")
    
    if not errors and not warnings:
        print("\n✅ Configuration is valid! No errors or warnings found.")
        return True, []
    
    if warnings:
        print(f"\n⚠️  Found {len(warnings)} warning(s):")
        for i, warning in enumerate(warnings, 1):
            print(f"  {i}. {warning}")
    
    if errors:
        print(f"\n❌ Found {len(errors)} error(s):")
        for i, error in enumerate(errors, 1):
            print(f"  {i}. {error}")
        print("\n❌ Configuration validation FAILED!")
        return False, errors
    
    print("\n✅ Configuration is valid! (with warnings)")
    return True, warnings


def load_config_from_module(module_path: str):
    """Load a configuration object from a Python module path.
    
    Args:
        module_path: String in format 'module.submodule.CONFIG_NAME'
        
    Returns:
        The configuration object
    """
    try:
        parts = module_path.rsplit('.', 1)
        if len(parts) != 2:
            raise ValueError(f"Config path must be in format 'module.CONFIG_NAME', got: {module_path}")
        
        module_name, config_name = parts
        module = importlib.import_module(module_name)
        
        if not hasattr(module, config_name):
            raise AttributeError(f"Module '{module_name}' has no attribute '{config_name}'")
        
        return getattr(module, config_name)
    except ImportError as e:
        print(f"❌ Error importing module: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        sys.exit(1)


def main():
    """Main function to run configuration validation."""
    parser = argparse.ArgumentParser(
        description="Validate robot configuration files for IsaacLab.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Configuration to validate (e.g., 'isaaclab_assets.CRAZYFLIE_CFG')"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print detailed validation information"
    )
    
    args = parser.parse_args()
    
    # Load the configuration
    print(f"Loading configuration: {args.config}")
    config = load_config_from_module(args.config)
    
    # Validate the configuration
    is_valid, messages = validate_config(config, args.config, args.verbose)
    
    # Exit with appropriate code
    sys.exit(0 if is_valid else 1)


if __name__ == "__main__":
    main()
