Added
^^^^^

* Added ``use_mixed_precision`` to :class:`~isaaclab_rl.rsl_rl.RslRlPpoAlgorithmCfg` and
  ``torch_compile_mode`` to :class:`~isaaclab_rl.rsl_rl.RslRlOnPolicyRunnerCfg` to configure RSL-RL's
  bfloat16 policy update and ``torch.compile`` of the actor and critic. Both options remained disabled by default.
  Set ``torch_compile_mode="default"`` (CLI: ``agent.torch_compile_mode=default``) to enable compilation.
  Compilation added startup overhead and could change floating-point results and sampled training trajectories.

Changed
^^^^^^^

* Enabled ``torch.backends.cudnn.benchmark`` in the RSL-RL training entrypoint, matching the existing
  non-deterministic environment seeding behavior.
