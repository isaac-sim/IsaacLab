* **Breaking:** Removed the ``apply_device`` argument of :func:`~isaaclab_rl.entrypoints.common.apply_env_overrides`,
  which no longer writes ``--device``; :func:`~isaaclab.app.launch_simulation` writes it to
  ``env_cfg.sim.device``. Call :func:`~isaaclab_rl.entrypoints.common.show_run_summary` inside
  :func:`~isaaclab.app.launch_simulation`.
* **Breaking:** Removed :func:`~isaaclab_rl.entrypoints.common.validate_distributed_device`: with ``--distributed``,
  :func:`~isaaclab.app.launch_simulation` always assigns each rank a ``cuda:N`` device, so the check could not fail.
