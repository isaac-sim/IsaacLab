Changed
^^^^^^^

* The run summary of the reinforcement learning workflows is printed after the launch and reports the resolved
  backends, visualizers, and device, so ``--viz none`` shows no visualizer and distributed runs show each
  rank's device. Automatic backend selectors are no longer shown next to the backend they resolved to.
* **Breaking:** Call :func:`~isaaclab_rl.entrypoints.common.apply_video_recording` inside
  :func:`~isaaclab.app.launch_simulation`: without declared recorders it records from the first capture-capable
  visualizer the launch resolved into ``env_cfg.sim.visualizer_cfgs`` and raises when there is none. Only
  :func:`~isaaclab_rl.entrypoints.common.pre_launch_video_config`, called before the launch, adds a headless Kit
  visualizer for ``--video``. The zero and random agents now configure recording after the launch.

Removed
^^^^^^^

* **Breaking:** Removed the ``apply_device`` argument of :func:`~isaaclab_rl.entrypoints.common.apply_env_overrides`,
  which no longer writes ``--device``; :func:`~isaaclab.app.launch_simulation` writes it to
  ``env_cfg.sim.device``. Call :func:`~isaaclab_rl.entrypoints.common.show_run_summary` inside
  :func:`~isaaclab.app.launch_simulation`.
* **Breaking:** Removed :func:`~isaaclab_rl.entrypoints.common.validate_distributed_device`: with ``--distributed``,
  :func:`~isaaclab.app.launch_simulation` always assigns each rank a ``cuda:N`` device, so the check could not fail.
