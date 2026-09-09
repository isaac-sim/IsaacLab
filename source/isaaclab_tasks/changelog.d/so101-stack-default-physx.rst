Fixed
^^^^^

* Defaulted the SO-101 cube-stacking tasks to the PhysX backend, so the gripper no longer
  penetrates the cubes and grasps hold. ``IsaacContrib-Stack-Cube-SO101-v0``,
  ``IsaacContrib-Stack-Cube-SO101-IK-Abs-v0``, and
  ``IsaacContrib-Stack-Cube-SO101-Joint-Teleop-v0`` previously resolved to Newton MJWarp, whose
  gripper contact response is still being tuned for this robot. Pass ``physics=newton_mjwarp``
  to select the previous backend.
