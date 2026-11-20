* Added the ``Isaac-RizonSharpa-Coffee-Kitchen-Camera-Benchmark`` rendering benchmark: the Rizon-Sharpa coffee
  demo's kitchen scene replaying the demo's recorded arm motion in a loop from a different time offset per
  environment, with one camera per environment, selectable through ``<data type><resolution>`` presets. Its
  ``benchmark_mode`` writes the demo's arm poses after physics (``render``, the default) or replays the demo's
  actuator targets (``physics_render``). ``scripts/benchmarks/record_coffee_arm_motion.py`` records the motion.
* Added :mod:`isaaclab_tasks.benchmark.benchmark_mode`, which holds the benchmark modes and direct-pose checks
  shared by the rendering benchmarks.
