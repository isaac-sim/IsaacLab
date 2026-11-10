* Added video recording to the Warp environments. :class:`~isaaclab_experimental.envs.ManagerBasedEnvWarp`,
  :class:`~isaaclab_experimental.envs.ManagerBasedRLEnvWarp` and :class:`~isaaclab_experimental.envs.DirectRLEnvWarp`
  now create the recorders in ``env_cfg.video_recorders``, advance them once per step and flush them on close.
  Previously the setting was ignored, with a warning only on the manager-based environments.
