Changed
^^^^^^^

* Moved the Newton deformable asset, data buffers, and kernels into
  :mod:`isaaclab_newton.assets`, removing the optional dependency on ``isaaclab_contrib``.
  Imported declared deformable prototypes once during cloning and selected their native particle
  ranges during asset initialization, removing the geometry registry and per-world construction hook.
  The shared :class:`isaaclab.assets.DeformableObjectCfg` and backend-independent asset API remained unchanged.

* **Breaking:** Made ``NewtonManager.instantiate_builder_from_stage()`` consume the active clone plan
  instead of discovering environment roots. Construct an ``InteractiveScene``, explicitly replicate a
  ``ClonePlan``, or supply a native builder with ``NewtonManager.set_builder()``.

Fixed
^^^^^

* Applied clone-plan row selection to Newton deformables and imported shared deformables once.
  Preserved rotated particle positions, velocities, and tetrahedral rest frames during builder composition.

* Moved MPM particle-range and visual-geometry binding into clone/import, removing asset-side registration.

* Moved Fabric body-prim preparation out of Newton physics startup and into the shared Fabric rendering resource.

* Scoped deformable kinematic defaults to each asset's selected particles instead of copying the entire model.
  Preserved imported cloth rest angles during asset initialization.
