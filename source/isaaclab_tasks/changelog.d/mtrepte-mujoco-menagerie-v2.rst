Fixed
^^^^^

* Fixed the Unitree Go2, Unitree H1, Unitree G1, and ANYmal-C velocity-tracking task configurations
  to reference the body and joint names used by their MuJoCo Menagerie USD assets (for example,
  ``.*_calf`` instead of ``.*_foot`` for Go2, ``.*SHANK`` instead of ``.*FOOT`` for ANYmal-C, and
  ``pelvis`` instead of ``torso_link`` as the G1 root body), and selected the Newton-specific H1
  configuration (:data:`~isaaclab_assets.robots.unitree.H1_NEWTON_MINIMAL_CFG`) when training with
  the Newton backend so joint position targets are converted to PD torques correctly.
* Fixed the direct-workflow Allegro Hand reorientation task to use the Menagerie joint and
  fingertip body names (``ffj0``, ``ff_tip``, etc.) instead of the stale legacy Allegro asset
  names, which no longer matched the robot's Menagerie-sourced USD asset.
* Fixed the Spot flat velocity-tracking task configuration to reference ``.*_lleg`` instead of
  ``.*_foot`` for contact, air-time, slip, and gait reward terms, matching the MuJoCo Menagerie
  Spot asset's body names (it has no separate foot bodies, the same as ANYmal). The
  ``foot_clearance`` reward's ``target_height`` was tuned against a foot-tip body and may need
  re-tuning once the asset gets dedicated foot bodies.
