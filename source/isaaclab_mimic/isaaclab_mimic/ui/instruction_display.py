# Copyright (c) 2024-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Module for handling instruction displays in Isaac Lab environments."""

from typing import Any

from pxr import Gf

from isaaclab.envs.mimic_env_cfg import MimicEnvCfg


class InstructionDisplay:
    """Handles instruction display for different teleop devices."""

    def __init__(self, teleop_device):
        self.teleop_device = teleop_device.lower()

        if "handtracking" in self.teleop_device.lower():
            from isaaclab.ui.xr_widgets import show_instruction

            self._display_subtask = lambda text: show_instruction(
                text, "/_xr/stage/xrCamera", Gf.Vec3f(1.25, 0.3, -2), target_prim_path="/subtask_instruction"
            )
            self._display_demo = lambda text: show_instruction(
                text, "/_xr/stage/xrCamera", Gf.Vec3f(-1.25, 0.3, -2), target_prim_path="/demo_complete"
            )
        else:
            self.subtask_label = None
            self.demo_label = None
            self._display_subtask = lambda text: setattr(self.subtask_label, "text", text)
            self._display_demo = lambda text: setattr(self.demo_label, "text", text)

    def set_labels(self, subtask_label, demo_label):
        """Set the instruction labels for non-handtracking displays."""
        self.subtask_label = subtask_label
        self.demo_label = demo_label

    def show_subtask(self, text):
        """Display subtask instruction."""
        self._display_subtask(text)

    def show_demo(self, text):
        """Display demo completion message."""
        self._display_demo(text)


def show_subtask_instructions(
    instruction_display: InstructionDisplay, prev_subtasks: dict, obv: dict, env_cfg: Any
) -> None:
    """
    Detect changes in subtasks and display the changes.

    Args:
        instruction_display: Display handler for showing instructions
        prev_subtasks: Previous subtask terms
        obv: Current observation with subtask terms
        env_cfg: Environment configuration containing subtask descriptions
    """
    if not isinstance(env_cfg, MimicEnvCfg):
        return
    subtasks = obv[0].get("subtask_terms")
    if subtasks is None:
        return

    # Currently only supports one end effector
    eef_name = list(env_cfg.subtask_configs.keys())[0]
    subtask_configs = env_cfg.subtask_configs[eef_name]

    all_false = True
    for subtask_config in subtask_configs:
        term_signal = subtask_config.subtask_term_signal
        if term_signal is None:
            continue

        current_state = subtasks[term_signal].item()
        prev_state = prev_subtasks.get(term_signal, False)

        if current_state:
            all_false = False

        # Show message when state changes from False to True
        if current_state and not prev_state:
            instruction_display.show_subtask(f"Next objective: {subtask_config.next_subtask_description}")

        # Update the previous state
        prev_subtasks[term_signal] = current_state

    # If all tasks are false, show the first task's description
    if all_false and subtask_configs:
        first_task = subtask_configs[0]
        instruction_display.show_subtask(f"Current objective: {first_task.description}")
