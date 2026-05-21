Fixed
^^^^^

* Pinned ``torch.cuda.set_device(PhysicsManager._device)`` before PhysX GPU warmup
  and again before the ``PHYSICS_READY`` dispatch inside
  :meth:`~isaaclab_physx.physics.PhysxManager.reset` and
  :meth:`~isaaclab_physx.physics.PhysxManager._warmup_and_create_views`. PhysX
  (``force_load_physics_from_usd`` / ``start_simulation`` /
  ``create_simulation_view``) could otherwise bring up its CUDA context on a
  device that differs from ``PhysicsManager._device``, so the first
  ``torch.tensor(..., device=self.device)`` allocation in an asset/sensor
  ``_initialize_impl`` callback (e.g.
  :class:`~isaaclab_physx.assets.articulation.ArticulationData` initializing
  ``GRAVITY_VEC_W``) would hit a CudaContextManager that PhysX had failed to
  rebuild and surface as ``CUDA error: an illegal memory access was encountered``
  alongside ``Failed to create Cuda Context Manager`` from
  ``omni.physx.foundation.plugin``. Observed on aarch64 with
  ``isaacsim==6.0.0rc48`` + ``torch==2.10.0+cu130`` while launching
  ``Isaac-Navigation-3DObstacles-ARL-Robot-1-v0``.
