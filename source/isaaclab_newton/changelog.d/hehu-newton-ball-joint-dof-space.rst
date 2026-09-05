Fixed
^^^^^

* Fixed :attr:`~isaaclab_newton.assets.articulation.articulation_data.ArticulationData.joint_pos`
  exposing Newton's ``joint_q`` -- joint *coordinate* space -- rather than DOF positions. A ball
  joint occupies 4 quaternion components against 3 DOFs, so on an articulation containing one the
  array was wider than ``num_joints`` while ``joint_names``, ``joint_vel``, ``default_joint_pos``
  and the joint gains stayed in DOF space, and every consumer indexing them with the same joint ids
  read a different joint past the first ball joint. Added
  :class:`~isaaclab_newton.assets.articulation.joint_coordinates.JointCoordinateMap`, which converts
  between the two spaces on read and on write. Articulations whose joints all have one coordinate
  per DOF keep the existing zero-copy view and are unaffected.
