Added
^^^^^

* Added the joint-drive schema-fragment API: the
  :class:`~isaaclab.sim.schemas.JointDriveFragment` marker and
  :class:`~isaaclab.sim.schemas.UsdPhysicsDriveCfg` (writing the typed multi-instance
  ``UsdPhysics.DriveAPI`` attributes). The drive fragment overrides its ``func`` with
  :func:`~isaaclab.sim.schemas.apply_drive`, which selects the angular/linear instance, performs
  the radian-to-degree conversion for angular drives, and skips tendon child prims.
* Added :func:`~isaaclab.sim.schemas.apply_joint_drive_properties` (applies a list of joint-drive
  fragments to all joint prims under a path; ``UsdPhysics.DriveAPI`` is presence-gated and applied
  only when a :class:`~isaaclab.sim.schemas.UsdPhysicsDriveCfg` fragment is present).

Changed
^^^^^^^

* Changed the spawner ``joint_drive_props`` slot
  (:attr:`~isaaclab.sim.spawners.FileCfg.joint_drive_props`) to also accept a list of
  :class:`~isaaclab.sim.schemas.JointDriveFragment` fragments. Legacy single cfgs continue to work
  through a transition bridge at the from-files spawn site. Added the spawner-level
  ``ensure_drives_exist`` flag to reproduce the legacy minimal-stiffness behaviour for the fragment
  path.
