Changed
^^^^^^^

* **Breaking:** Switched Kit, Newton, Rerun, and Viser streaming views to scene-owned camera sensors.
  Moved generated-camera settings to scene ``CameraCfg`` declarations; visualizers selected them with
  ``streaming_sensor_prim_path``. Removed late camera creation, follow-pose discovery, and camera teardown
  from visualizers. Closing one viewer no longer affected a camera used by another viewer.
* Used scene-authored lights in Newton RTX by default, including HDR textures, intensity, initial
  transforms, and clone placements. An explicit solid background preserved scene illumination.
  Retained the interactive camera controls and picking; no additional lighting configuration was required.

Fixed
^^^^^

* Fixed Newton RTX frame capture with OVRTX 0.5's RenderVar path keys.
