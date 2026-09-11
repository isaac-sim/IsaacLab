Changed
^^^^^^^

* Moved benchmark-only tasks into a new ``isaaclab_tasks.benchmark`` package so they are no longer mixed in with
  the trainable ``core`` and ``contrib`` task families. The ``IsaacContrib-Reorient-Cube-Shadow-Camera-Benchmark-Direct``
  task id is unchanged, but its configuration moved from
  ``isaaclab_tasks.contrib.reorient.config.shadow_hand.shadow_hand_camera_benchmark_env_cfg`` to
  ``isaaclab_tasks.benchmark.shadow_hand_camera.shadow_hand_camera_benchmark_env_cfg``. Code that imports the
  configuration class directly must update the import path; code that resolves the task through ``gym.make`` or
  ``load_cfg_from_registry`` needs no change.
