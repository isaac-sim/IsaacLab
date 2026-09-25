Changed
^^^^^^^

* Moved the Newton deformable asset, data buffers, and kernels into
  :mod:`isaaclab_newton.assets`, removing the optional dependency on ``isaaclab_contrib``.
  The shared :class:`isaaclab.assets.DeformableObjectCfg` and backend-independent asset API remained unchanged.
