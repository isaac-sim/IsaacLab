Added
^^^^^

* Added ``OvPhysxBackendCfg`` and ``OVRTXBackendCfg`` for cfg-only native resource acquisition through the registry.
  ``OvPhysxCfg`` continued to configure the physics manager and its scene policy.

Changed
^^^^^^^

* Moved OVPhysX runtime and attached OVStage ownership into the simulation backend registry, preserving
  the existing reset, native teardown order, and public ``OvPhysxManager.get_physx_instance()`` API.
* Moved OVRTX engine and detached-stage ownership into the same registry. Camera bindings stayed on the renderer;
  native engine destruction preceded detached-stage release, including cleanup before scene attachment.
