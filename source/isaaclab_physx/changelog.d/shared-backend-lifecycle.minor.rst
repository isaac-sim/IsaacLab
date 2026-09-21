Added
^^^^^

* Added ``PhysxBackendCfg`` to share a native simulation view by its stage identifier through
  ``SimulationContext.get_or_create_backend(cfg)``.
* Exposed the registry-owned resource as ``PhysxManager.backend`` and ``PhysxSceneDataBackend.backend``.

Changed
^^^^^^^

* Shared one registry-owned native simulation view between physics and scene data, removing an
  unused duplicate Warp view. Existing physics-view access remained unchanged.
