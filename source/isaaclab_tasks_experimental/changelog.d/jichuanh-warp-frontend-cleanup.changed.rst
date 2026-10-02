* Changed the Cartpole, locomotion and reorient Warp direct environments to record their ``_apply_action``,
  ``_get_dones``, ``_get_rewards``, ``_get_observations`` and ``_reset_idx`` methods as CUDA graphs with
  :func:`~isaaclab_experimental.utils.captured`, since :class:`~isaaclab_experimental.envs.DirectRLEnvWarp` no
  longer records them.
