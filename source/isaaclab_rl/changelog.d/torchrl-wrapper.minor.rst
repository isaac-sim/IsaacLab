Added
^^^^^

* Added :class:`~isaaclab_rl.torchrl.IsaacLabTorchRLWrapper` to wrap Isaac Lab environments for the
  `TorchRL <https://github.com/pytorch/rl>`_ library. The wrapper implements :class:`~torchrl.envs.EnvBase`
  directly, preserving per-group observation structure (e.g. ``"policy"``/``"critic"``) as a
  :class:`~torchrl.data.Composite` spec and exposing ``done``/``terminated``/``truncated`` as separate spec
  keys. Reset requests that TorchRL issues after a step with done environments are served from the current
  observations, since Isaac Lab already reset those environments inside ``step()``, and the terminal
  observation (``extras["final_obs"]``, when ``cfg.compute_final_obs`` is enabled) is reported for done
  transitions so that time-limit bootstrapping is correct.
* Added :func:`~isaaclab_rl.torchrl.train_ppo` with :class:`~isaaclab_rl.torchrl.TorchRlPpoCfg`, a PPO example
  built from TorchRL's collector, GAE, and clipped PPO loss, and the ``torchrl`` backend of the unified ``train``
  entrypoint (``--rl_library torchrl``).
