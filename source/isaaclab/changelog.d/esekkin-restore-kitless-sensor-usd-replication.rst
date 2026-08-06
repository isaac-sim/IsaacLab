Fixed
^^^^^

* Fixed kitless sensor cloning by restoring the default
  :attr:`~isaaclab.sensors.SensorBaseCfg.cloning_contexts` request for
  :class:`~isaaclab.cloner.UsdReplicateContext`. Without it a spawned sensor was authored only
  under ``env_0``, so backends that resolve their sensor views from USD reported a
  per-environment prim count mismatch when Kit was absent.
