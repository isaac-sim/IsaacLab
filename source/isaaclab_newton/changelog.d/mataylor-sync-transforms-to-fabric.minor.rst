Changed
^^^^^^^

* Renamed :meth:`~isaaclab_newton.physics.NewtonManager.sync_transforms_to_usd` to
  :meth:`~isaaclab_newton.physics.NewtonManager.sync_transforms_to_fabric`. The method writes
  ``omni:fabric:worldMatrix`` through Fabric and never authors a USD attribute, so the old name
  named the wrong destination and obscured that the poses reach the RTX renderer but not a stage
  export or save.

Deprecated
^^^^^^^^^^

* Deprecated :meth:`~isaaclab_newton.physics.NewtonManager.sync_transforms_to_usd`. It now logs a
  warning and forwards to :meth:`~isaaclab_newton.physics.NewtonManager.sync_transforms_to_fabric`,
  and will be removed in a future release. Replace calls to ``sync_transforms_to_usd`` with
  ``sync_transforms_to_fabric``; behaviour is unchanged.
