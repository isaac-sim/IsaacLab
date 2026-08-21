Fixed
^^^^^

* Fixed a GPU crash on the first environment step when the PhysX backend is combined with a non-Kit
  visualizer, such as ``--viz newton_rtx physics=isaacsim_physx``. Such runs aborted with ``CUDA
  error: unspecified launch failure`` after ``omni.physx.fabric`` reported "mismatched prototypes on
  point instancer". Creating the Kit ``UsdGeom.PointInstancer`` marker backend was gated on
  :attr:`~isaaclab.sim.SimulationContext.is_rendering`, which is ``True`` whenever any visualizer is
  configured, so the instancer was created even though the Kit render pipeline never runs and never
  populates it in Fabric. The Kit marker backend is now created only when the Kit render pipeline is
  actually active. Runs using a Kit visualizer are unaffected.
