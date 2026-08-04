Changed
^^^^^^^

* Changed Newton Kit viewport transform sync to call
  ``IFabricHierarchy.update_world_xforms_gpu_with_options`` with
  ``FabricHierarchyGpuUpdateOptions.RIGID_BODY | FORCE_UPDATE`` instead of the
  private ctypes ``omni::cubric::IAdapter`` shim. Older Kit builds without the
  new API continue to fall back to ``IFabricHierarchy.update_world_xforms``.

Removed
^^^^^^^

* Removed :mod:`isaaclab_newton.physics._cubric` ctypes bindings for
  ``omni::cubric::IAdapter``. Use
  ``IFabricHierarchy.update_world_xforms_gpu_with_options`` instead.
