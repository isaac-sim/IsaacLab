Changed
^^^^^^^

* Moved the Newton deformable asset, data buffers, and kernels into
  :mod:`isaaclab_newton.assets`, removing the optional dependency on ``isaaclab_contrib``.
  Imported declared deformable prototypes once during cloning and selected their native particle
  ranges during asset initialization, removing the geometry registry and per-world construction hook.
  The shared :class:`isaaclab.assets.DeformableObjectCfg` and backend-independent asset API remained unchanged.

Fixed
^^^^^

* Applied clone-plan row selection to Newton deformables and imported shared deformables once.
  Preserved rotated particle positions, velocities, and tetrahedral rest frames during builder composition.
