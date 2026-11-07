* Fixed the Lee multirotor controllers rotating each body inertia twice when aggregating the robot
  inertia. :attr:`~isaaclab.assets.ArticulationData.body_inertia` is already expressed in the link
  frame, so bodies whose principal axes are rotated relative to their link frame produced a wrong
  robot inertia on the PhysX and OVPhysX backends.
