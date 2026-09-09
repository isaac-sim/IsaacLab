Fixed
^^^^^

* Fixed ``SceneDataProvider`` (``get_transforms``, ``init_output``, ``create_mapping``,
  ``create_geometry_mapping``, ``get_points``) allocating its internal Warp arrays on Warp's
  process-global default device instead of the sim backend's own device. This caused a CUDA
  illegal memory access (``Warp CUDA error 700``) when running with ``--device cpu`` on a machine
  that also has a CUDA device present, since the conversion kernel launched on ``cuda:0`` against
  CPU-resident sim data. All internal allocations now derive their device from the sim backend.
* Fixed ``PhysicsManager.initialize`` never synchronizing Warp's process-global default device for
  ``--device cpu`` runs (only the CUDA branch called ``wp.set_device``). Any Warp array or kernel
  launch anywhere in the codebase that omits an explicit ``device=`` now correctly defaults to the
  configured sim device instead of silently defaulting to ``cuda:0`` whenever a CUDA device is
  present in the process.
