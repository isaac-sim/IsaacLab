Changed
^^^^^^^

* **Breaking:** Changed :attr:`~isaaclab_ov.physics.OvPhysxCfg.enable_external_forces_every_iteration`
  to default to ``True``, matching the PhysX backend's TGS force integration setting.
  This changed the default velocity updates and contact-force response of OvPhysX simulations.
  Set ``enable_external_forces_every_iteration=False`` explicitly in ``OvPhysxCfg`` to retain
  the previous behavior.
