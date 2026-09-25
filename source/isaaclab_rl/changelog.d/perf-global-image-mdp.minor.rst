Added
^^^^^

* Added ``use_mixed_precision`` to :class:`~isaaclab_rl.rsl_rl.RslRlPpoAlgorithmCfg` and
  ``torch_compile_mode`` to :class:`~isaaclab_rl.rsl_rl.RslRlOnPolicyRunnerCfg` to enable RSL-RL's
  bfloat16 policy update and ``torch.compile`` of the actor and critic.

Changed
^^^^^^^

* Changed RSL-RL training to enable ``torch.backends.cudnn.benchmark``. Observation and minibatch
  shapes are fixed during training, so autotuned convolution algorithms speed up CNN policies.
