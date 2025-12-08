# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any

from isaaclab.ui.xr_widgets import DataCollector, TriggerType, VisualizationManager
from isaaclab.ui.xr_widgets.instruction_widget import hide_instruction


class TeleopVisualizationManager(VisualizationManager):
    """Specialized visualization manager for teleoperation scenarios.
    For sample and debug use.

    Provides teleoperation-specific visualization features including:
    - IK error handling and display
    """

    def __init__(self, data_collector: DataCollector):
        """Initialize the teleop visualization manager and register callbacks.

        Args:
            data_collector: DataCollector instance to read data for visualization use.
        """
        super().__init__(data_collector)

        # Handle error event
        self._error_text_color = 0xFF0000FF
        self.ik_error_widget_id = "/ik_solver_failed"

        self.register_callback(TriggerType.TRIGGER_ON_EVENT, {"event_name": "ik_error"}, self._handle_ik_error)

    def _handle_ik_error(self, mgr: VisualizationManager, data_collector: DataCollector, params: Any = None) -> None:
        """Handle IK error events by displaying an error message widget.

        Args:
            data_collector: DataCollector instance (unused in this handler)
        """
        # Todo: move display_widget to instruction_widget.py
        if not hasattr(mgr, "_ik_error_widget_timer"):
            self.display_widget(
                "IK Error Detected",
                mgr.ik_error_widget_id,
                VisualizationManager.message_widget_preset()
                | {"text_color": self._error_text_color, "display_duration": None},
            )
            mgr._ik_error_widget_timer = mgr.register_callback(
                TriggerType.TRIGGER_ON_PERIOD, {"period": 3.0, "initial_countdown": 3.0}, self._hide_ik_error_widget
            )
            if mgr._ik_error_widget_timer is None:
                mgr.cancel_rule(TriggerType.TRIGGER_ON_PERIOD, mgr._ik_error_widget_timer)
                mgr.cancel_rule(TriggerType.TRIGGER_ON_EVENT, "ik_solver_failed")
                raise RuntimeWarning("Failed to register IK error widget timer")
        else:
            mgr._ik_error_widget_timer.countdown = 3.0

    def _hide_ik_error_widget(self, mgr: VisualizationManager, data_collector: DataCollector) -> None:
        """Hide the IK error widget.

        Args:
            data_collector: DataCollector instance (unused in this handler)
        """

        hide_instruction(mgr.ik_error_widget_id)
        mgr.cancel_rule(TriggerType.TRIGGER_ON_PERIOD, mgr._ik_error_widget_timer)
        delattr(mgr, "_ik_error_widget_timer")
