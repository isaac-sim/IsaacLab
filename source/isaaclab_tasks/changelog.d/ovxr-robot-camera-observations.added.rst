* Added fixed-base G1 wrist cameras, and hid the robot's head mesh from the OVXR render.
* Added an ``rgb`` preset to ``IsaacContrib-PickPlace-FixedBaseUpperBodyIK-G1-Abs`` that adds the
  wrist cameras, their XR camera panels (Kit ``--xr`` uploads capped at 10 Hz) and the head and wrist
  RGB images to its policy observations (``presets=rgb``). It re-renders three times after each reset
  (``num_rerenders_on_reset = 3``), so the images are not stale after a reset. The default renders
  only the head camera and keeps the policy observations state-only.
