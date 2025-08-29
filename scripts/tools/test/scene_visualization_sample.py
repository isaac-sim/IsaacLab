from isaaclab.ui.xr_widgets import XRVisualization, TriggerType, VisualizationManager, DataCollector,update_instruction
from pxr import Gf

def _sample_handle_ik_error(self, mgr: VisualizationManager, data_collector: DataCollector) -> None:
    """Handle IK error events by displaying an error message widget.
    
    Args:
        data_collector: DataCollector instance (unused in this handler)
    """

    self.display_widget("IK Error Detected", "/ik_error", VisualizationManager.message_widget_preset() | {"text_color": self._error_text_color})

def _sample_update_error_text_color(self, mgr: VisualizationManager, data_collector: DataCollector) -> None:
    self._error_text_color = self._error_text_color + 0x100
    if self._error_text_color >= 0xFFFFFFFF:
        self._error_text_color = 0xFF0000FF

def _sample_update_left_panel(self, mgr, data_collector) -> None:
    """Update the left panel with current data and update counter.
    
    Args:
        data_collector: DataCollector instance containing current data
    """
    left_panel_id = getattr(self, '_left_panel_id', None)
    if left_panel_id is not None:
        content = f"{mgr._left_panel_updated_times}\n{data_collector.make_panel_content()}"
        update_instruction(left_panel_id, content)
        mgr._left_panel_updated_times += 1

def _sample_update_right_panel(self, mgr, data_collector) -> None:
    """Update the right panel with current data and update counter.
    
    Args:
        data_collector: DataCollector instance containing current data
    """
    right_panel_id = getattr(self, '_right_panel_id', None)
    if right_panel_id is not None:
        content = f"{mgr._right_panel_updated_times}\n{data_collector.make_panel_content()}"
        update_instruction(right_panel_id, content)
        mgr._right_panel_updated_times += 1

def apply_sample_visualization():
    # Error Message
    XRVisualization.register_callback(TriggerType.TRIGGER_ON_EVENT, {"event_name": "ik_error"}, _sample_handle_ik_error)

    # Display a panel on the left to display DataCollector data
    # Refresh periodically
    # Todo: use a better way to add '/' to pathname
    XRVisualization.set_attrs({
        "left_panel_id": "/left_panel",
        "left_panel_translation": Gf.Vec3f(-2, 2.6, 2),
        "left_panel_updated_times": 0,
        "right_panel_updated_times": 0,
    })
    XRVisualization.register_callback(TriggerType.TRIGGER_ON_PERIOD, {"period": 1.0}, _sample_update_left_panel)

    # Display a panel on the right to display DataCollector data
    # Refresh when data changes
    XRVisualization.set_attrs({
        "right_panel_id": "/right_panel",
        "right_panel_translation": Gf.Vec3f(1.5, 2, 2),
    })
    XRVisualization.register_callback(TriggerType.TRIGGER_ON_EVENT, {"event_name": "default_event_has_change"}, _sample_update_right_panel)

    # Change error text color every second
    XRVisualization.set_attrs({
        "error_text_color": 0xFF0000FF,
    })
    XRVisualization.register_callback(TriggerType.TRIGGER_ON_UPDATE, {}, _sample_update_error_text_color)

