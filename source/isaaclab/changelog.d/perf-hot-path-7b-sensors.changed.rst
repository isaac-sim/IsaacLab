* Changed sensors with ``debug_vis=True`` to refresh their buffers lazily from the debug
  visualization callback instead of on every :meth:`~isaaclab.sensors.SensorBase.update`.
  Custom sensors must refresh outdated buffers (for example through ``data``) in their
  ``_debug_vis_callback`` before reading internal data.
* Skipped ray-caster drift sampling on reset when ``drift_range`` and ``ray_cast_drift_range``
  are zero, and uploaded the ray-cast drift ranges to the device only when they change.
* Changed :meth:`~isaaclab.markers.VisualizationMarkers.visualize` to return without validating
  or processing inputs when no visualizer backend is active, such as in headless runs.
