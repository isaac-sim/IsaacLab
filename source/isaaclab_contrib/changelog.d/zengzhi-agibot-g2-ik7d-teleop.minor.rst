Added
^^^^^

* Added :class:`~isaaclab_contrib.mdp.Ik7dAction` and :class:`~isaaclab_contrib.controllers.Ik7dController`,
  wrapping AgiBot's ``ik_7d`` 7-DoF redundant-arm IK solver for Isaac Lab teleoperation environments.
* Added ``Isaac-Teleop-G2-Ik7d-v0``, a VR teleoperation environment for the AgiBot G2 (``G2_t2_crs`` variant)
  driven through ``ik_7d``. The environment attaches via the ``--external_callback`` hook
  (``isaaclab_contrib.tasks.agibot_g2.register``) and requires out-of-band assets (robot USD/URDF
  and the ``ik_7d`` wheel) under ``AGIBOT_G2_USD_DIR`` / ``AGIBOT_G2_URDF_DIR``.
* Added :class:`~isaaclab_contrib.tasks.agibot_g2.SwivelRetargeter`, an IsaacTeleop retargeter
  that integrates thumbstick input into elbow-swivel commands for 7-DoF arms.
