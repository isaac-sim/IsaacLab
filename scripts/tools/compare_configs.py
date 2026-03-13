# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Utility to compare two robot configurations and highlight differences.

This tool loads two robot configurations and performs a detailed comparison,
showing differences in spawn settings, actuators, initial states, and other properties.

Usage:
    # Compare two robot configs
    python scripts/tools/compare_configs.py \
        --config1 isaaclab_assets.ANYMAL_D_CFG \
        --config2 isaaclab_assets.ANYMAL_C_CFG
    
    # Export comparison to JSON
    python scripts/tools/compare_configs.py \
        --config1 isaaclab_assets.FRANKA_PANDA_CFG \
        --config2 isaaclab_assets.UR10_CFG \
        --output comparison.json
"""

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


def load_config(config_path: str) -> Any:
    """Load a robot configuration from module path.
    
    Args:
        config_path: Module path like 'isaaclab_assets.ROBOT_CFG'
        
    Returns:
        Loaded configuration object
    """
    try:
        module_path, config_name = config_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        return getattr(module, config_name)
    except (ImportError, AttributeError, ValueError) as e:
        print(f"[ERROR] Failed to load config '{config_path}': {e}")
        sys.exit(1)


def get_config_attributes(config: Any) -> Dict[str, Any]:
    """Extract all attributes from a configuration object.
    
    Args:
        config: Configuration object to extract from
        
    Returns:
        Dictionary of attribute names and values
    """
    attrs = {}
    for attr_name in dir(config):
        if not attr_name.startswith('_'):
            try:
                attr_value = getattr(config, attr_name)
                if not callable(attr_value):
                    # Convert to string for comparison
                    if hasattr(attr_value, '__dict__'):
                        attrs[attr_name] = str(type(attr_value).__name__)
                    else:
                        attrs[attr_name] = str(attr_value)
            except Exception:
                pass
    return attrs


def compare_attributes(attrs1: Dict[str, Any], attrs2: Dict[str, Any]) -> Tuple[List, List, List]:
    """Compare two attribute dictionaries.
    
    Args:
        attrs1: First configuration attributes
        attrs2: Second configuration attributes
        
    Returns:
        Tuple of (common_attrs, only_in_1, only_in_2)
    """
    keys1 = set(attrs1.keys())
    keys2 = set(attrs2.keys())
    
    common = sorted(keys1 & keys2)
    only_1 = sorted(keys1 - keys2)
    only_2 = sorted(keys2 - keys1)
    
    return common, only_1, only_2


def print_comparison(config1_name: str, config2_name: str, attrs1: Dict, attrs2: Dict):
    """Print a formatted comparison of two configurations.
    
    Args:
        config1_name: Name of first config
        config2_name: Name of second config
        attrs1: Attributes from first config
        attrs2: Attributes from second config
    """
    print("\n" + "="*80)
    print(f"Configuration Comparison")
    print("="*80)
    print(f"Config 1: {config1_name}")
    print(f"Config 2: {config2_name}")
    print("="*80 + "\n")
    
    common, only_1, only_2 = compare_attributes(attrs1, attrs2)
    
    # Print common attributes with differences
    print("[COMMON ATTRIBUTES]")
    print("-"*80)
    differences = 0
    identical = 0
    
    for attr in common:
        val1 = attrs1[attr]
        val2 = attrs2[attr]
        
        if val1 != val2:
            differences += 1
            print(f"\n{attr}:")
            print(f"  Config 1: {val1}")
            print(f"  Config 2: {val2}")
        else:
            identical += 1
    
    print(f"\nSummary: {identical} identical, {differences} different")
    
    # Print attributes only in config 1
    if only_1:
        print("\n[ONLY IN CONFIG 1]")
        print("-"*80)
        for attr in only_1:
            print(f"  {attr}: {attrs1[attr]}")
    
    # Print attributes only in config 2
    if only_2:
        print("\n[ONLY IN CONFIG 2]")
        print("-"*80)
        for attr in only_2:
            print(f"  {attr}: {attrs2[attr]}")
    
    print("\n" + "="*80)


def export_comparison(config1_name: str, config2_name: str, 
                     attrs1: Dict, attrs2: Dict, output_path: Path):
    """Export comparison results to JSON file.
    
    Args:
        config1_name: Name of first config
        config2_name: Name of second config
        attrs1: Attributes from first config
        attrs2: Attributes from second config
        output_path: Path to output JSON file
    """
    common, only_1, only_2 = compare_attributes(attrs1, attrs2)
    
    comparison_data = {
        "config1": config1_name,
        "config2": config2_name,
        "common_attributes": {},
        "only_in_config1": {},
        "only_in_config2": {}
    }
    
    # Add common attributes with values from both
    for attr in common:
        comparison_data["common_attributes"][attr] = {
            "config1_value": attrs1[attr],
            "config2_value": attrs2[attr],
            "identical": attrs1[attr] == attrs2[attr]
        }
    
    # Add unique attributes
    for attr in only_1:
        comparison_data["only_in_config1"][attr] = attrs1[attr]
    
    for attr in only_2:
        comparison_data["only_in_config2"][attr] = attrs2[attr]
    
    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(comparison_data, f, indent=2)
    
    print(f"[SUCCESS] Comparison exported to: {output_path}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Compare two robot configurations.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--config1",
        type=str,
        required=True,
        help="First robot configuration (e.g., 'isaaclab_assets.ANYMAL_D_CFG')"
    )
    parser.add_argument(
        "--config2",
        type=str,
        required=True,
        help="Second robot configuration (e.g., 'isaaclab_assets.ANYMAL_C_CFG')"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for comparison results (optional)"
    )
    
    args = parser.parse_args()
    
    # Load configurations
    print(f"Loading {args.config1}...")
    config1 = load_config(args.config1)
    
    print(f"Loading {args.config2}...")
    config2 = load_config(args.config2)
    
    # Extract attributes
    attrs1 = get_config_attributes(config1)
    attrs2 = get_config_attributes(config2)
    
    # Print comparison
    print_comparison(args.config1, args.config2, attrs1, attrs2)
    
    # Export if requested
    if args.output:
        export_comparison(args.config1, args.config2, attrs1, attrs2, Path(args.output))


if __name__ == "__main__":
    main()
