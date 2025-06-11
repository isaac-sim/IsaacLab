# Copyright (c) 2024-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to construct prompts to control the Cosmos model's generation.

Required arguments:
    --templates_path         Path to the file containing templates for the prompts.

Optional arguments:
    --num_prompts            Number of prompts to generate (default: 1).
    --output_path            Path to the output file to write generated prompts (default: prompts.txt).
"""

import argparse
import json
import random


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate prompts for controlling Cosmos model's generation.")
    parser.add_argument(
        "--templates_path", type=str, required=True, help="Path to the JSON file containing prompt templates"
    )
    parser.add_argument("--num_prompts", type=int, default=1, help="Number of prompts to generate (default: 1)")
    parser.add_argument(
        "--output_path", type=str, default="prompts.txt", help="Path to the output file to write generated prompts"
    )
    args = parser.parse_args()

    return args


def generate_prompt(templates_path: str):
    """Generate a random prompt for controlling the Cosmos model's visual augmentation.

    The prompt describes the scene and desired visual variations, which the model
    uses to guide the augmentation process while preserving the core robotic actions.

    Args:
        templates_path (str): Path to the JSON file containing prompt templates.

    Returns:
        str: Generated prompt string that specifies visual aspects to modify in the video.
    """
    try:
        with open(templates_path) as f:
            templates = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Prompt templates file not found: {templates_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON in prompt templates file: {templates_path}")

    prompt_parts = []

    for section_name, section_options in templates.items():
        if not isinstance(section_options, list):
            continue
        if len(section_options) == 0:
            continue
        selected_option = random.choice(section_options)
        prompt_parts.append(selected_option)

    return " ".join(prompt_parts)


def main():
    # Parse command line arguments
    args = parse_args()

    prompts = [generate_prompt(args.templates_path) for _ in range(args.num_prompts)]

    try:
        with open(args.output_path, "w") as f:
            for prompt in prompts:
                f.write(prompt + "\n")
    except Exception as e:
        print(f"Failed to write to {args.output_path}: {e}")


if __name__ == "__main__":
    main()
