Added
^^^^^

* Added isolated multi-world and bounded sparse-grid configuration to
  :class:`~isaaclab_newton.physics.MPMSolverCfg`.
* Added standard visual-material binding for MPM particle spawners.
* Added opt-in cell-centered particle placement to
  :class:`~isaaclab_newton.sim.spawners.mpm.MPMGridCfg` while preserving its
  existing boundary-placement default.
* Added graph-captured bounded-sparse MPM snowball-smash and teapot-fill demos,
  including a rigid-MPM proxy-coupling example.
* Added an implicit MPM authoring and tuning guide.
* Enabled CUDA graph capture for capacity-bounded rebuildable sparse MPM,
  including nested coupled-solver entries.
* Added a safe copy helper for Newton clone-source builders used by offline IK
  and collision screening.

Changed
^^^^^^^

* Renamed ``scripts/demos/mpm/particle_pour.py`` to
  ``scripts/demos/mpm/teapot_fill.py``. Invoke the teapot-fill path for the
  maintained container-filling example. Use the canonical ``--max_steps``,
  ``--voxel_size``, and ``--container_usd`` options.

Deprecated
^^^^^^^^^^

* Deprecated the ``"instantaneous"`` and ``"finite_difference"`` values of
  :attr:`~isaaclab_newton.physics.MPMSolverCfg.collider_velocity_mode` in favor
  of ``"forward"`` and ``"backward"``, respectively.

Fixed
^^^^^

* Kept kitless MPM particle visuals on their fallback display color when
  Kit-only render materials are unavailable.
* Fixed stale solver-owned history during task-driven resets on both active
  state buffers through Newton's shared local/global reset-mask contract.
* Prevented the first deferred CUDA graph capture from advancing physics twice.
* Preserved eager fallback for dense and unbounded sparse MPM grids when CUDA
  graphs are enabled.
* Surfaced asynchronous sparse-grid rebuild failures after CUDA graph replay.
* Deferred automatic coupled MPM history resets until tasks finish rewriting
  state, while allowing isolated-world tasks to reset selected worlds exactly.
