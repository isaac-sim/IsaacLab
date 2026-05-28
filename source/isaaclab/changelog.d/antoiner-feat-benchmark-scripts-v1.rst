Added
^^^^^

* Added an opt-in ``--schema_v1_output <path>`` flag to ``benchmark_startup.py``,
  ``benchmark_rsl_rl.py``, and ``benchmark_skrl.py``. When set, each script
  emits a self-contained ``training.json`` / ``startup.json`` JSON file
  conforming to :mod:`isaaclab.benchmark.schema` (v1.0) — run identity,
  software versions, host hardware, aggregated runtime + resource metrics,
  and EMA-smoothed reward / episode-length curves. The legacy per-backend
  output format remains the default when the flag is omitted.
* Added ``benchmark_skrl.py``: the SKRL-framework counterpart to
  ``benchmark_rsl_rl.py``. Emits an identical v1.0 ``TrainingBundle`` with
  ``framework: "skrl"``.
* Added :doc:`/source/features/benchmarking` documenting the three scripts
  and the v1.0 bundle schema.

Changed
^^^^^^^

* Extended :func:`scripts.benchmarks.utils.parse_cprofile_stats` to return a
  4-tuple ``(function_label, tottime_ms, cumtime_ms, ncalls)`` instead of a
  3-tuple, exposing the primitive call count from ``pstats`` for downstream
  consumers. Existing tuple-unpacking call sites updated.
* Reworked ``scripts/benchmarks/startup_whitelist.yaml`` to track the
  IsaacLab v3 configclass / cloner / scene-init call paths and added an
  explicit ``task_config`` phase entry.
