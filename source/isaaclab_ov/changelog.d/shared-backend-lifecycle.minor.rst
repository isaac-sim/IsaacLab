Added
^^^^^

* Added ``OvPhysxBackendCfg`` for cfg-only native runtime acquisition through the simulation registry.
  ``OvPhysxCfg`` continued to configure the physics manager and its scene policy.

Changed
^^^^^^^

* Moved OVPhysX runtime and attached OVStage ownership into the simulation backend registry, preserving
  the existing reset, native teardown order, and public ``OvPhysxManager.get_physx_instance()`` API.
