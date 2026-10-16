* Deprecated the ``"controller"`` component of :func:`~isaaclab.actuators.newton.read_group_parameter` and
  :func:`~isaaclab.actuators.newton.write_group_parameter` in favor of ``"drive"``, following the Newton 1.6
  drive API. The old name still resolves to the drive and emits a ``DeprecationWarning``; it is scheduled for
  removal in 3.2.
