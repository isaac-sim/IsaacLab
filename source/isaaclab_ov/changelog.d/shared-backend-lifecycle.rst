Changed
^^^^^^^

* Moved OVPhysX runtime and attached OVStage ownership into the simulation backend registry, preserving
  the existing reset, native teardown order, and public ``OvPhysxManager.get_physx_instance()`` API.
