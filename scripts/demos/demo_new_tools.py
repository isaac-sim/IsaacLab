# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Demo script showcasing the new utility tools for IsaacLab.

This script demonstrates how to use the validation, export, and benchmarking tools
to analyze robot configurations and measure performance.

Usage:
    ./isaaclab.sh -p scripts/demos/demo_new_tools.py
"""

import subprocess
import sys
from pathlib import Path


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")


def run_command(cmd: list, description: str):
    """Run a command and display results."""
    print(f"\n🛠️  {description}")
    print(f"Command: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        print(result.stdout)
        if result.stderr:
            print("Stderr:", result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error running command: {e}")
        return False


def main():
    """
    Main demo function showcasing new tools.
    """
    print("\n" + "#"*70)
    print("#" + " "*68 + "#")
    print("#" + "  IsaacLab New Tools Demo".center(68) + "#")
    print("#" + "  Validation | Export | Benchmarking".center(68) + "#")
    print("#" + " "*68 + "#")
    print("#"*70)
    
    # Get the IsaacLab root directory
    isaac_lab_root = Path(__file__).resolve().parents[2]
    
    # Example robot configurations to test
    test_configs = [
        "isaaclab_assets.CRAZYFLIE_CFG",
        "isaaclab_assets.ANYMAL_D_CFG",
    ]
    
    # ========================================================================
    # PART 1: Configuration Validation
    # ========================================================================
    print_section("🔍 PART 1: Robot Configuration Validation")
    
    print("📝 Testing configuration validation tool...")
    print("This tool checks if robot configs have all required fields.\n")
    
    for config in test_configs:
        print(f"\n--- Validating: {config} ---")
        cmd = [
            sys.executable,
            str(isaac_lab_root / "scripts" / "tools" / "validate_robot_config.py"),
            "--config", config
        ]
        run_command(cmd, f"Validating {config}")
    
    # ========================================================================
    # PART 2: Configuration Export
    # ========================================================================
    print_section("📤 PART 2: Robot Configuration Export")
    
    print("📝 Exporting robot configuration to JSON...")
    print("This tool extracts detailed robot information for documentation.\n")
    
    config = test_configs[0]  # Use first config for demo
    output_file = isaac_lab_root / "robot_config_demo.json"
    
    cmd = [
        sys.executable,
        str(isaac_lab_root / "scripts" / "tools" / "export_robot_info.py"),
        "--config", config,
        "--output", str(output_file),
        "--pretty"
    ]
    
    if run_command(cmd, f"Exporting {config} to JSON"):
        if output_file.exists():
            print(f"\n✅ JSON file created: {output_file}")
            print(f"   File size: {output_file.stat().st_size} bytes")
            
            # Show a snippet of the JSON
            with open(output_file, 'r') as f:
                lines = f.readlines()[:10]
                print("\n   First 10 lines:")
                for line in lines:
                    print(f"   {line.rstrip()}")
    
    # ========================================================================
    # PART 3: Tool Summary
    # ========================================================================
    print_section("📊 PART 3: Tools Summary")
    
    print("✨ Available Tools in scripts/tools/:\n")
    
    tools_info = [
        {
            "name": "validate_robot_config.py",
            "purpose": "Validate robot configurations",
            "simulation": "No",
            "output": "Console/Exit code"
        },
        {
            "name": "export_robot_info.py",
            "purpose": "Export config to JSON",
            "simulation": "No",
            "output": "JSON file"
        },
        {
            "name": "benchmark_performance.py",
            "purpose": "Measure simulation performance",
            "simulation": "Yes",
            "output": "Console/CSV"
        },
    ]
    
    # Print table
    print(f"{'Tool':<30} {'Purpose':<35} {'Sim?':<6} {'Output'}")
    print("-" * 90)
    for tool in tools_info:
        print(
            f"{tool['name']:<30} "
            f"{tool['purpose']:<35} "
            f"{tool['simulation']:<6} "
            f"{tool['output']}"
        )
    
    # ========================================================================
    # Usage Examples
    # ========================================================================
    print_section("🚀 Quick Usage Examples")
    
    print("📝 Validate a configuration:")
    print("   ./isaaclab.sh -p scripts/tools/validate_robot_config.py \\")
    print("       --config your_robot.YOUR_CFG --verbose\n")
    
    print("📤 Export configuration to JSON:")
    print("   ./isaaclab.sh -p scripts/tools/export_robot_info.py \\")
    print("       --config your_robot.YOUR_CFG --output robot.json --pretty\n")
    
    print("📊 Benchmark performance:")
    print("   ./isaaclab.sh -p scripts/tools/benchmark_performance.py \\")
    print("       --robot your_robot.YOUR_CFG --counts 1,10,50,100 \\")
    print("       --output benchmark.csv\n")
    
    # ========================================================================
    # Completion
    # ========================================================================
    print("\n" + "="*70)
    print("✅ Demo completed successfully!")
    print("="*70)
    
    print("\n📚 For more information, see: scripts/tools/README.md")
    print("👍 These tools help you validate configs, analyze robots, and optimize performance!\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
