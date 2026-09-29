Changed
^^^^^^^

* The run summary of the reinforcement learning workflows is printed after the launch and reports the resolved
  backends, visualizers, and device, so ``--viz none`` shows no visualizer and distributed runs show each
  rank's device. Automatic backend selectors are no longer shown next to the backend they resolved to.

Removed
^^^^^^^

* **Breaking:** Removed the ``apply_device`` argument of :func:`~isaaclab_rl.entrypoints.common.apply_env_overrides`,
  which no longer writes ``--device``; :func:`~isaaclab.app.launch_simulation` writes it to
  ``env_cfg.sim.device``. Call :func:`~isaaclab_rl.entrypoints.common.show_run_summary` inside
  :func:`~isaaclab.app.launch_simulation`.
