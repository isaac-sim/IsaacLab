Changed
^^^^^^^

* **Breaking:** Removed the legacy benchmark entry-point scripts now superseded by the unified
  ``runtime.py``, ``startup.py``, and ``training.py`` scripts: ``benchmark_non_rl.py``,
  ``benchmark_startup.py``, ``benchmark_rsl_rl.py``, and ``benchmark_rlgames.py``. The
  ``run_non_rl_benchmarks.sh``, ``run_physx_benchmarks.sh``, and ``run_training_benchmarks.sh``
  runner shells and the obsolete ``scripts/benchmarks/utils.py`` helper module were removed as
  well. Use ``runtime.py``, ``startup.py``, and ``training.py --rl_library <lib>`` instead;
  run the PhysX micro-benchmarks under ``source/isaaclab_physx/benchmark/`` directly. See the
  "Benchmark Scripts" section of the Isaac Lab 3.0 migration guide for the full command mapping.
