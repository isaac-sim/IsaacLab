# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import ipywidgets as widgets
from IPython.display import display


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
        value=10,
        min=1,
        max=1000000,
        description="Number of Trials:",
        style={"description_width": "initial"},
        layout={"width": "300px"},
    )

    display(widgets.HTML("<h3>3. Set Number of Trials</h3>"))
    display(widgets.HTML("<p>How many demonstrations to generate</p>"))
    display(num_trials_widget)
    return num_trials_widget
