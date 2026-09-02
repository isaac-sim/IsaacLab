Removed
^^^^^^^

* **Breaking:** Removed the ``ovphysx`` physics, deformable, scene, event, and curriculum presets
  from ``Isaac-Lift-Soft-Franka``, ``Isaac-Lift-Cloth-Franka``, and their camera variants because
  OVPhysX currently produces incorrect deformable behavior for these tasks. Existing commands using
  ``physics=ovphysx`` or ``presets=ovphysx`` must use ``physics=isaacsim_physx``, or drop the
  override to use the default ``newton_mjwarp_vbd_proxy``. As a result, ``physics=physx`` now always
  resolves to Isaac Sim PhysX for these tasks instead of selecting OVPhysX when Isaac Sim is absent.
