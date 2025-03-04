# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import base64
import os
import re
import toml

import ipywidgets as widgets
from IPython.display import display


def create_variable_dropdowns(preset_path):
    display(widgets.HTML("<h3>Prompt Generator</h3>"))

    class PromptManager:
        """Manages prompt state and provides access to current prompt value."""

        def __init__(self, preset, widget_dict, prompt_display):
            self.preset = preset
            self.widget_dict = widget_dict
            self.prompt_display = prompt_display
            self._current_prompt = ""

        def update_prompt(self, *args):
            """Updates the prompt based on current dropdown values."""
            current_values = {k: v.value for k, v in self.widget_dict.items()}
            self._current_prompt = self.preset["prompt"]["description"].format(**current_values)
            self.prompt_display.value = self._current_prompt

        @property
        def prompt(self):
            """Returns the current prompt value."""
            return self._current_prompt.replace("\n", " ")

    with open(preset_path) as f:
        preset = toml.load(f)

    widget_dict = {}
    prompt_display = widgets.HTML()

    # Create prompt manager instance
    prompt_manager = PromptManager(preset, widget_dict, prompt_display)

    for var, values in preset["variables"].items():
        widget = widgets.Dropdown(
            options=values,
            value=values[0],
            description=f"{var.replace('_', ' ').title()}:",
            style={"description_width": "initial"},
            layout={"width": "200", "margin": "0 20px 0 0"},
        )
        # Add observer to each dropdown
        widget.observe(prompt_manager.update_prompt, names="value")

        widget_dict[var] = widget

    display(widgets.HBox(list(widget_dict.values())))

    # Initial prompt update
    prompt_manager.update_prompt()
    display(widgets.HTML("<h4>Prompt</h4>"))
    display(prompt_display)

    return prompt_manager


def create_cosmos_params():
    seed = widgets.IntText(
        value=42,
        description="Seed:",
        disabled=False,
        style={"description_width": "initial"},  # This allows description to use natural width
        layout=widgets.Layout(width="150px", margin="0 20px 0 0"),
    )

    control_weight = widgets.FloatSlider(
        value=0.2,
        min=0.1,
        max=0.9,
        step=0.1,
        description="Control Weight:",
        style={"description_width": "initial"},  # This allows description to use natural width
        disabled=False,
        layout=widgets.Layout(width="250px", margin="0 20px 0 0"),
    )

    sigma_max = widgets.IntSlider(
        value=35,
        min=35,
        max=80,
        step=1,
        description="Sigma Max:",
        style={"description_width": "initial"},  # This allows description to use natural width
        disabled=False,
        layout=widgets.Layout(width="250px", margin="0 20px 0 0"),
    )

    use_canny_edge = widgets.Checkbox(
        value=True,
        description="Use Canny Edges",
        style={"description_width": "initial"},  # This allows description to use natural width
        disabled=False,
        layout=widgets.Layout(width="150px", margin="0 20px 0 0"),
    )

    blur_strength = widgets.Dropdown(
        options=["Very Low", "Low", "Medium", "High", "Very High"],
        value="Very Low",
        description="Blur Strength:",
        disabled=False,
        style={"description_width": "initial"},  # This allows description to use natural width
        layout=widgets.Layout(width="200px"),
    )
    display(widgets.HTML("<h3>Cosmos Parameters</h3>"))
    display(
        widgets.HBox(
            [seed, control_weight, sigma_max, use_canny_edge, blur_strength],
        )
    )

    return {
        "seed": seed,
        "control_weight": control_weight,
        "sigma_max": sigma_max,
        "use_canny_edge": use_canny_edge,
        "blur_strength": blur_strength,
    }


def create_download_link(file_path, link_text="Download File"):
    """Creates an HTML link to download a local file.

    Args:
        file_path (str): Path to the file to be downloaded
        link_text (str, optional): Text to display for the link. Defaults to "Download File"

    Returns:
        IPython.display.HTML: HTML display object with download link
    """
    with open(file_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    filename = file_path.split("/")[-1]

    button_style = """
        background-color: #4CAF50;
        border: none;
        color: white;
        padding: 15px 32px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 4px;
    """

    return widgets.HTML(
        f'<a download="{filename}" '
        f'href="data:application/octet-stream;base64,{b64}" '
        f'style="{button_style}">{link_text}</a>'
    )


def create_camera_input(root_dir):
    """Create a dropdown for video source selection."""
    files = os.listdir(root_dir)
    valid_files = {f.split("_normals_")[0] for f in files if "_normals_" in f}

    available_cameras = list(valid_files)

    camera_widget = widgets.Dropdown(
        options=available_cameras,
        value=available_cameras[0],
        description="Source Camera:",
        style={"description_width": "initial"},
        layout={"width": "auto"},
    )

    display(widgets.HTML("<h3>1. Select Camera</h3>"))
    display(camera_widget)
    return camera_widget


def create_start_frame_input(root_dir):
    pattern = r".*?_(\d+)\.png$"
    files = (f for f in os.listdir(root_dir) if f.endswith(".png"))
    # Extract numbers and sort them
    max_frame = max([int(re.match(pattern, filename).group(1)) for filename in files if re.match(pattern, filename)])

    start_frame_widget = widgets.IntSlider(
        value=1,
        min=1,
        max=max_frame,
        step=1,
        description="Start Frame:",
        style={"description_width": "initial"},
        layout={"width": "500px"},
    )
    display(widgets.HTML("<h3>1. Select Start Frame</h3>"))
    display(start_frame_widget)
    return start_frame_widget


def create_task_input():
    """Create a dropdown for task selection."""
    available_tasks = [
        "Isaac-Stack-Cube-Franka-IK-Rel-Mimic-v0",
    ]

    task_widget = widgets.Dropdown(
        options=available_tasks,
        value=available_tasks[0],
        description="Task:",
        style={"description_width": "initial"},
        layout={"width": "auto"},
    )

    display(widgets.HTML("<h3>1. Select Task</h3>"))
    display(task_widget)
    return task_widget


def create_num_envs_input():
    """Create a slider for number of environments."""
    num_envs_widget = widgets.IntSlider(
        value=9,
        min=1,
        max=100,
        step=1,
        description="Number of Environments:",
        style={"description_width": "initial"},
        layout={"width": "500px"},
    )

    display(widgets.HTML("<h3>2. Set Number of Environments</h3>"))
    display(num_envs_widget)
    return num_envs_widget


def create_num_trials_input():
    """Create a number input for trials."""
    num_trials_widget = widgets.BoundedIntText(
        value=1,
        min=1,
        max=100,
        description="Number of Trials:",
        style={"description_width": "initial"},
        layout={"width": "300px"},
    )

    display(widgets.HTML("<h3>3. Set Number of Trials</h3>"))
    display(widgets.HTML("<p>How many demonstrations to generate</p>"))
    display(num_trials_widget)
    return num_trials_widget
