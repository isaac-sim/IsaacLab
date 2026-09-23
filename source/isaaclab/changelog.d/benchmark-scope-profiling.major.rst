Added
^^^^^

* Added :func:`~isaaclab.benchmark.stepping.profile_physics_steps` and
  :func:`~isaaclab.benchmark.stepping.profile_renderers` context managers for synchronized
  runtime benchmark timings. Wrappers recorded complete calls after warmup and restored
  the original methods when measurement ended, including on failure.

Changed
^^^^^^^

* Restricted scope capture to tasks with a non-``None`` ``benchmark_mode``. To collect timings,
  enable ``ISAACLAB_PHYSICS_PROFILE=1`` or ``ISAACLAB_RENDER_PROFILE=1`` for such a task.
  Other tasks continued to produce standard runtime reports without scope profiling.

* Included scalar physics and render profiling summaries in ``BenchmarkResult.bundle.extra``
  without changing schema version 1.4. Schema and OmniPerf output included each scope's mean,
  standard deviation, maximum time per call [ms], and call count. Consumers should read
  ``physics_mean_ms``, ``physics_std_ms``, ``physics_max_ms``, ``physics_calls``, and the
  corresponding ``render_*`` keys for these summaries. Raw ordered samples remained in
  ``<output_path>/profile_timings.json`` as ``timings_ms`` pairs for local analysis instead
  of parsing printed timer lines; use ``--output_path`` to select the output directory.

* **Breaking:** Moved render profiling into the runtime benchmark through
  :func:`~isaaclab.benchmark.stepping.profile_renderers`. To collect render timings with
  ``ISAACLAB_RENDER_PROFILE=1``, use the runtime benchmark; normal simulation runs no longer
  allocate render timers. Scene updates and output readback remained outside the timed scope.

Deprecated
^^^^^^^^^^

* Deprecated ``isaaclab.renderers.render_context.RENDER_PROFILE_SCOPE``; use
  :data:`~isaaclab.benchmark.stepping.RENDER_PROFILE_SCOPE` instead.
