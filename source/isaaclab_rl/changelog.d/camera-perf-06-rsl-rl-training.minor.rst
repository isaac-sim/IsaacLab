Added
^^^^^

* Added ``use_mixed_precision`` to :class:`~isaaclab_rl.rsl_rl.RslRlPpoAlgorithmCfg` and
  ``torch_compile_mode`` to :class:`~isaaclab_rl.rsl_rl.RslRlOnPolicyRunnerCfg` to configure RSL-RL's
  bfloat16 policy update and ``torch.compile`` of the actor and critic. Mixed precision remained disabled by default.

Changed
^^^^^^^

* Enabled actor and critic compilation with ``torch_compile_mode="default"`` for RSL-RL on-policy runners.
  Set ``torch_compile_mode=None`` (CLI: ``agent.torch_compile_mode=null``) to retain eager execution.
  Compilation added startup overhead and could change floating-point results and sampled training trajectories.
* Enabled ``torch.backends.cudnn.benchmark`` in the RSL-RL training entrypoint, matching the existing
  non-deterministic environment seeding behavior.
