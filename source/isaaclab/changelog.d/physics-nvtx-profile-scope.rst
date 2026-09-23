Added
^^^^^

* Added :func:`~isaaclab.benchmark.stepping.profile_physics_steps` to time the selected physics backend
  during runtime benchmarks with ``ISAACLAB_PHYSICS_PROFILE=1``. The benchmark wrapped
  ``step`` once after warmup for the rest of the process. Each complete step, including
  inherited calls, emitted one synchronized timing under
  :data:`~isaaclab.benchmark.stepping.PHYSICS_PROFILE_SCOPE`. Normal simulation runs incurred no profiling overhead.

Changed
^^^^^^^

* Disabled render and physics scope capture when the task configuration's ``benchmark_mode``
  was absent or ``None``, even with profiling flags enabled. Standard runtime reports remained
  available. To collect these scopes, use a task with a non-``None`` ``benchmark_mode`` and
  enable the corresponding profiling flags.

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
