Added
^^^^^

* Added the ``contrib/franka_pick_berries`` task: continuous-grasp Franka teleoperation with deformable
  Gaussian-splat berries, registered as ``IsaacContrib-Pick-Berry-Franka-IK-Rel-Newton``.

Changed
^^^^^^^

* Replaced the berry task's EBC studio lighting with broad, neutral indoor lights to soften robot shadows
  and better integrate the robot and support table with the scanned room.
* Restored the robot's authored surface finishes and the support table's normal, roughness, and metalness
  texture maps in the standalone berry viewer.
