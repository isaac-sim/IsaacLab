* Fixed the Lee multirotor controllers rotating each body inertia twice when aggregating the robot
  inertia. :attr:`~isaaclab.assets.ArticulationData.body_inertia` is already expressed in the link
  frame, so bodies whose principal axes are rotated relative to their link frame produced a wrong
  robot inertia on the PhysX and OVPhysX backends.
* **Breaking:** Removed the ``body_com_quat_b`` argument from
  ``isaaclab_contrib.utils.math.aggregate_inertia_about_robot_com``, which now rotates the
  link-frame inertias by the link orientation only. Drop the argument from existing calls.
