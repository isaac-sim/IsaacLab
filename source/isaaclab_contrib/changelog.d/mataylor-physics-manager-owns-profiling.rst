Changed
^^^^^^^

* Renamed :meth:`NewtonCoupledMJWarpVBDManager.step` to ``NewtonCoupledMJWarpVBDManager._step`` to
  match the ``PhysicsManager`` backend contract, where the public ``step`` classmethod now owns
  profiling and each backend implements ``_step``. See
  ``source/isaaclab/changelog.d/physics-nvtx-profile-scope.rst``.
