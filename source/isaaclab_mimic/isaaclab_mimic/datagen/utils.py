# Copyright (c) 2024-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from IPython.display import display
from ipywidgets import widgets

from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import EventTermCfg
from isaaclab.utils.datasets import HDF5DatasetFileHandler


def get_nested_value(d: dict[str, Any], keys: list[str]) -> Any:
    """Retrieve a nested value from dictionary d using list of keys."""
    for k in keys:
        d = d[k]
    return d


def update_nested_value(d: dict[str, Any], keys: list[str], value: Any) -> None:
    """Update a nested value in dictionary d using list of keys."""
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def reset_env(env: ManagerBasedEnv, steps: int = 1) -> None:
    """Reset environment and step simulation to stabilize state."""
    # Get sim and scene from unwrapped environment
    sim = env.unwrapped.sim
    scene = env.unwrapped.scene

    # Reset environment
    env.reset()

    # Step simulation multiple times to stabilize
    for _ in range(steps):
        # Write data to sim
        scene.write_data_to_sim()
        # Perform step
        sim.step()
        # Update buffers
        scene.update(dt=env.physics_dt)


def get_parameter_input(
    param_name: str,
    current_val: float | tuple[float, float] | list[float],
    allowed_range: tuple[float, float, float | None],
    update_fn: Callable[[float | tuple[float, float]], None],
    env: ManagerBasedEnv | None = None,
    event_term_name: str | None = None,
) -> widgets.FloatSlider | widgets.FloatRangeSlider:
    """Get parameter input using ipywidgets with immediate value updates."""

    if isinstance(current_val, (tuple, list)):
        step_size = allowed_range[2] if len(allowed_range) > 2 else 0.01
        full_param_name = f"{event_term_name}.{param_name}" if event_term_name else param_name

        # Create container with label and range slider
        container = widgets.HBox([
            widgets.Label(full_param_name, layout=widgets.Layout(width="auto")),
            widgets.FloatRangeSlider(
                value=[current_val[0], current_val[1]],
                min=allowed_range[0],
                max=allowed_range[1],
                step=step_size,
                layout=widgets.Layout(width="300px"),
                readout=True,
                readout_format=".3f",
            ),
        ])

        def on_value_change(change):
            new_tuple = (change["new"][0], change["new"][1])
            update_fn(new_tuple)
            if env is not None:
                reset_env(env, steps=50)

        container.children[1].observe(on_value_change, names="value")

        # Create help text showing the allowed range
        help_text = widgets.HTML(value=f'<p style="color:gray">Allowed range: {allowed_range[:2]}</p>')

        display(container)
        display(help_text)

        return container.children[1]
    else:
        step_size = allowed_range[2] if len(allowed_range) > 2 else 0.01
        full_param_name = f"{event_term_name}.{param_name}" if event_term_name else param_name

        # Create container with label and slider
        container = widgets.HBox([
            widgets.Label(full_param_name, layout=widgets.Layout(width="auto")),
            widgets.FloatSlider(
                value=current_val,
                min=allowed_range[0],
                max=allowed_range[1],
                step=step_size,
                layout=widgets.Layout(width="300px"),
                readout=True,
                readout_format=".3f",
            ),
        ])

        def on_value_change(change):
            update_fn(change["new"])
            if env is not None:
                reset_env(env, steps=50)

        container.children[1].observe(on_value_change, names="value")

        # Create help text showing the allowed range
        help_text = widgets.HTML(value=f'<p style="color:gray">Allowed range: {allowed_range[:2]}</p>')

        display(container)
        display(help_text)

        return container.children[1]


def interactive_update_randomizable_params(
    event_term: EventTermCfg,
    event_term_name: str,
    param_config: dict[str, dict | tuple[float, float, float | None]],
    param_path: str = "",
    env: ManagerBasedEnv | None = None,
) -> list[tuple[list[str], widgets.FloatSlider | widgets.FloatRangeSlider]]:
    """Interactive parameter updates using ipywidgets."""
    inputs = []

    for key, allowed_range in param_config.items():
        current_path = f"{param_path}.{key}" if param_path else key
        keys = current_path.split(".")

        if isinstance(allowed_range, dict):
            interactive_update_randomizable_params(event_term, event_term_name, allowed_range, current_path, env)
        else:
            try:
                current_val = get_nested_value(event_term.params, keys)

                def make_update_fn(k, full_path):
                    def update_fn(new_val):
                        update_nested_value(event_term.params, k, new_val)
                        print(f"Updated '{full_path}' to {new_val}.")

                    return update_fn

                input_widget = get_parameter_input(
                    current_path,
                    current_val,
                    allowed_range,
                    make_update_fn(keys, current_path),
                    env=env,
                    event_term_name=event_term_name,
                )
                inputs.append((keys, input_widget))
            except KeyError:
                print(f"Key '{current_path}' not found in event_term.params; skipping.")
                continue

    return inputs


def setup_output_paths(output_file_path: str) -> tuple[str, str]:
    """Set up output directory and get file name for dataset generation.

    Args:
        output_file_path: Full path to the desired output file

    Returns:
        tuple containing:
            - output_dir: Path to the output directory
            - output_file_name: Name of the output file without extension
    """
    output_dir = os.path.dirname(output_file_path)
    output_file_name = os.path.splitext(os.path.basename(output_file_path))[0]

    # create directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    return output_dir, output_file_name


def get_env_name_from_dataset(input_file_path: str) -> str:
    """Get environment name from an input dataset file.

    Args:
        input_file_path: Path to the input dataset file

    Returns:
        env_name: Name of the environment from the dataset

    Raises:
        FileNotFoundError: If the input file does not exist
    """
    if not os.path.exists(input_file_path):
        raise FileNotFoundError(f"The dataset file {input_file_path} does not exist.")

    dataset_file_handler = HDF5DatasetFileHandler()
    dataset_file_handler.open(input_file_path)
    env_name = dataset_file_handler.get_env_name()
    if env_name is None:
        raise ValueError("Environment name not found in dataset")

    return env_name
