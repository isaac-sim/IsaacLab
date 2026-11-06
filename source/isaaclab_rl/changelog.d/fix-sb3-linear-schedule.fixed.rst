* Fixed :func:`~isaaclab_rl.sb3.process_sb3_cfg` giving every ``lin_<value>`` schedule the value of the last one
  parsed. With ``learning_rate: lin_3e-4`` and ``clip_range: lin_0.2``, the learning rate started at 0.2.
