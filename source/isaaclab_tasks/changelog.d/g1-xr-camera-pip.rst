Changed
^^^^^^^

* Enabled the calibrated ``robot_pov_cam`` as a head-locked XR picture-in-picture
  panel for the G1 locomanipulation and fixed-base upper-body IK tasks. Fixed-base
  G1 gained a camera sensor for PiP without changing its policy observations.
  XR runs with enabled PiP now require ``--num_envs 1``. To retain multi-environment
  XR operation, set ``env.isaac_teleop.xr_camera_feeds=[]`` to disable PiP without
  removing the camera sensor or recorded observations. Isolated PiP requires a
  Kit runtime with the XR scene-partition propagation and mesh-bounds fixes;
  a supported released-runtime minimum has not yet been established.
