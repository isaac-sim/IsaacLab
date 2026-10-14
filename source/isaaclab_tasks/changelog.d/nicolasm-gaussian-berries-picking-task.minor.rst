Added
^^^^^

* Added the ``contrib/franka_pick_berries`` task: continuous-grasp Franka teleoperation with deformable
  Gaussian-splat berries, registered as ``IsaacContrib-Pick-Berry-Franka-IK-Rel-Newton``.
* Added a ``--bowl_material porcelain`` option to the berry demo script that renders the receiving bowl as
  glazed white porcelain.
* Added a ``vbd`` tissue solver to the berry task: Newton's VBD on a tetrahedral mesh of each berry, with the
  Gaussians skinned to the mesh.
* Added a ``--demo_length {medium,short}`` option to the berry demo script: scripted demos of about 30 s and 15 s
  that handle only the first two berries.

Changed
^^^^^^^

* Replaced the berry task's EBC studio lighting with broad, neutral indoor lights to soften robot shadows
  and better integrate the robot and support table with the scanned room.
* Restored the robot's authored surface finishes and the support table's normal, roughness, and metalness
  texture maps in the standalone berry viewer.
* Replaced the berry task's procedural punnet with a smaller, clear PET punnet asset, whose collision walls follow
  its tapered cavity.
* Lowered the berry task's receiving bowl from 4.5 cm to 3 cm.
* Set accepted berries down just above the receiving bowl's floor instead of dropping them, side by side in
  the scripted demo.
* Replaced the berry task's fixed layout, side by side and upright, with scattered and turned berries whose grasps
  keep the fingers clear of the punnet's wall; the scripted demo uses it, the same for every tissue solver.
* Replaced the berry task's room scan with a newer Gaussian capture of the same room, aligned like the old one and
  packaged as a single USDZ.
* Moved the berry task's raspberry, punnet and room scan assets to Isaac Lab's Nucleus asset root; the
  ``ISAACLAB_BERRY_ASSET_ROOT`` override now mirrors its layout.

Fixed
^^^^^

* Fixed the berry task's scripted demo squashing its first berry into a translucent sheet with a frayed rim: the crush
  now stops with the fingers 2 mm apart instead of 1 mm.
