Changed
^^^^^^^

* Changed the pinned ``ovphysx`` wheel to ``0.5.10`` and the pinned ``ovstage``
  wheel to ``0.1.1.355824``.

Fixed
^^^^^

* Fixed :class:`~isaaclab_ovphysx.physics.OvPhysxManager` attaching its OVStage
  at an unsealed write ordinal. ``ovstage.population.open_usd_from_string()``
  only completes population; it never commits the ordinal it wrote to. Under
  ``ovphysx`` 0.5.10 attaching at an unsealed ordinal fails the parse and yields
  an empty scene, so every articulation, rigid body, and sensor binding resolved
  to zero prims. The manager now calls ``advance_write_floor().wait()`` to seal
  the ordinal before ``attach_ovstage()``.
