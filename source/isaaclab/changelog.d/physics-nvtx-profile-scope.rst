Added
^^^^^

* Added :func:`~isaaclab.benchmark.stepping.profile_physics_steps` to time the selected physics backend
  during runtime benchmarks with ``ISAACLAB_PHYSICS_PROFILE=1``. The benchmark wrapped
  ``step`` once after warmup for the rest of the process. Each complete step, including
  inherited calls, emitted one synchronized timing under
  :data:`~isaaclab.benchmark.stepping.PHYSICS_PROFILE_SCOPE`. Normal simulation runs incurred no profiling overhead.

Changed
^^^^^^^

* **Breaking:** Moved render profiling into the runtime benchmark through
  :func:`~isaaclab.benchmark.stepping.profile_renderers`. To collect render timings with
  ``ISAACLAB_RENDER_PROFILE=1``, use the runtime benchmark; normal simulation runs no longer
  allocate render timers. Scene updates and output readback remained outside the timed scope.

Deprecated
^^^^^^^^^^

* Deprecated ``isaaclab.renderers.render_context.RENDER_PROFILE_SCOPE``; use
  :data:`~isaaclab.benchmark.stepping.RENDER_PROFILE_SCOPE` instead.
