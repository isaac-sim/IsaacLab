* **Breaking:** Switched Kit, Newton, Rerun, and Viser streaming views to scene-owned camera sensors.
  Moved generated-camera settings to scene ``CameraCfg`` declarations; visualizers selected them with
  ``streaming_sensor_prim_path``. Removed late camera creation, follow-pose discovery, and camera teardown
  from visualizers. Closing one viewer no longer affected a camera used by another viewer.
* Added the shared ``cameras`` list for perspective and existing scene-camera sources.
