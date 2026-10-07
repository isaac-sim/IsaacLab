* **Breaking:** Switched Kit, Newton, Rerun, and Viser streaming views to scene-owned camera sensors.
  Moved generated-camera settings to scene ``CameraCfg`` declarations; visualizers selected them with
  ``streaming_sensor_prim_path``. Removed late camera creation, follow-pose discovery, and camera teardown
  from visualizers. Closing one viewer no longer affected a camera used by another viewer.
* Added the shared ``cameras`` list for perspective and existing scene-camera sources.
* Bound scene camera references before backend initialization and consumed the supplied camera
  objects without cloning-plan dependencies. Custom visualizers must forward ``cameras`` and
  ``stage`` to ``super().initialize()``.
* **Breaking:** Removed ``newton_adapter.resolve_visible_env_indices`` and ``apply_viewer_visible_worlds``.
  Backends consumed the base visualizer's final selection directly; custom visualizers should pass
  ``get_visualized_env_ids()`` to their viewer after calling ``super().initialize()``.
