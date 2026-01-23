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
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def run_command(cmd: list, description: str):
    """Run a command and display results."""
    print(f"\n[TOOL]  {description}")
    print(f"Command: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        print(result.stdout)
        if result.stderr:
            print(f"[WARNING] {result.stderr}")
        return result.returncode == 0
    except Exception as e:
        print(f"[ERROR] Failed to run command: {e}")
        return False


def main():
    """Main demonstration function."""

    print_section("[SPARKLE] IsaacLab Utility Tools Demo [SPARKLE]")

    print("This demo showcases three new utility tools:")
    print("  1. validate_robot_config.py - [SEARCH] Validates robot configurations")
    print("  2. export_robot_info.py - [EXPORT] Exports robot information to JSON")
    print("  3. benchmark_performance.py - [STATS] Benchmarks robot loading performance\n")

    # Determine script directory
    script_dir = Path(__file__).resolve().parent.parent / "tools"

    # ========================================================================
    # Tool Overview Table
    # ========================================================================
    print_section("[FILE] Tool Overview")

    tools_info = [
        {
            "name": "validate_robot_config.py",
            "purpose": "Validates robot configuration",
            "simulation": "No",
            "output": "Console output",
        },
        {
            "name": "export_robot_info.py",
            "purpose": "Exports robot to JSON",
            "simulation": "No",
            "output": "JSON file",
        },
        {
            "name": "benchmark_performance.py",
            "purpose": "Benchmarks loading performance",
            "simulation": "Yes",
            "output": "CSV file",
        },
    ]

    # Print table header
    print(f"{'Tool Name':<30} {'Purpose':<35} {'Sim?':<6} {'Output'}")
    print("-" * 85)

    # Print table rows
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
    print_section("[LAUNCH] Quick Usage Examples")

    print("[NOTE] Validate a configuration:")
    print("   ./isaaclab.sh -p scripts/tools/validate_robot_config.py \\")
    print("       --config your_robot.YOUR_CFG --verbose\n")

    print("[EXPORT] Export configuration to JSON:")
    print("   ./isaaclab.sh -p scripts/tools/export_robot_info.py \\")
    print("       --config your_robot.YOUR_CFG --output robot.json --pretty\n")

    print("[STATS] Benchmark performance:")
    print("   ./isaaclab.sh -p scripts/tools/benchmark_performance.py \\")
    print("       --robot your_robot.YOUR_CFG --counts 1,10,50,100 \\")
    print("       --output benchmark.csv\n")

    # ========================================================================
    # Completion
    # ========================================================================
    print("\n" + "=" * 70)
    print("[SUCCESS] Demo completed successfully!")
    print("=" * 70)

    print("\n[DOCS] For more information, see: scripts/tools/README.md")
    print("[THUMBS_UP] These tools help you validate configs, analyze robots, and optimize performance!\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[WARNING]  Demo interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
        sys.exit(1)
