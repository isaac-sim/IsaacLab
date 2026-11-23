* Added the ``retirement_complete`` property of :class:`~isaaclab_teleop.OvLibrariesXrSession` for
  host-thread environment teardown. The session also retires on the simulation's
  :attr:`~isaaclab.physics.PhysicsEvent.STOP` event before cameras, renderers and the OVStage close,
  and keeps its renderer and OVStage alive when retirement cannot be proved.
