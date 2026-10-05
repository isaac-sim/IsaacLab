* **Breaking:** Switched Kit, Newton, Rerun, and Viser streaming views to scene-owned camera sensors.
  Moved generated-camera settings to scene ``CameraCfg`` declarations; visualizers selected them with
  ``streaming_sensor_prim_path``. Removed late camera creation, follow-pose discovery, and camera teardown
  from visualizers. Closing one viewer no longer affected a camera used by another viewer.
* Used scene-authored lights in Newton RTX by default, including HDR textures, intensity, initial
  transforms, and clone placements. An explicit solid background preserved scene illumination.
  Retained the interactive camera controls and picking; no additional lighting configuration was required.
* Added the shared ``cameras`` list for perspective and scene-camera sources. Scene-camera selection replaced the perspective viewport instead of drawing a floating panel
  over it. Navigation applied the same camera-local motion to every copy of the selected sensor;
  unselected sensors were neither read nor moved by the viewer.
