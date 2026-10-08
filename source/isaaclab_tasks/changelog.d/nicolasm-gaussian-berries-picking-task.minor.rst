Added
^^^^^

* Added the ``contrib/franka_pick_berries`` task: continuous-grasp Franka teleoperation with deformable
  Gaussian-splat berries, registered as ``IsaacContrib-Pick-Berry-Franka-IK-Rel-Newton``.
* Added a ``--bowl porcelain`` option to the berry teleoperation script that renders the receiving bowl as
  glazed white porcelain.

Changed
^^^^^^^

* Replaced the berry task's EBC studio lighting with broad, neutral indoor lights to soften robot shadows
  and better integrate the robot and support table with the scanned room.
* Restored the robot's authored surface finishes and the support table's normal, roughness, and metalness
  texture maps in the standalone berry viewer.
* Rendered the berry task's punnet as glossy clear plastic that casts no shadow.
* Lowered the berry task's receiving bowl from 4.5 cm to 3 cm.
