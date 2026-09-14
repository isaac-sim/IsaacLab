Changed
^^^^^^^

* Renamed :meth:`~isaaclab_newton.physics.NewtonManager.sync_transforms_to_usd` to
  :meth:`~isaaclab_newton.physics.NewtonManager.sync_transforms_to_fabric`. The method writes
  ``omni:fabric:worldMatrix`` through Fabric and never authors a USD attribute, so the old name
  described the wrong destination. Replace calls to ``sync_transforms_to_usd`` with
  ``sync_transforms_to_fabric``; the old name still works but now logs a deprecation warning and
  will be removed in a future release.
