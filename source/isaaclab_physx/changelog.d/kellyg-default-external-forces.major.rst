Changed
^^^^^^^

* **Breaking:** Changed :attr:`~isaaclab_physx.physics.PhysxCfg.enable_external_forces_every_iteration`
  to default to ``True``. Remove explicit ``True`` overrides; explicit ``False``
  overrides emit a ``DeprecationWarning`` because the PhysX flag will be removed
  in a future release.
