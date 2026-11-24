* Added published SAC checkpoints for RSL-RL tasks that register ``rsl_rl_sac_cfg_entry_point``. They are named with
  a ``_sac`` variant suffix, such as ``Isaac-Ant_newtonmjwarp_none_rsl_rl_sac.pt``, so they sit next to the PPO
  checkpoint of the same task. ``scripts/tools/train_and_publish_checkpoints.py --core`` trains them on Newton, with
  the env count a task declares through ``pretrained_checkpoint_num_envs``, and publishes them. Playback, export and
  benchmark playback with a SAC agent fetch them, and :func:`~isaaclab_rl.rsl_rl.rsl_rl_checkpoint_variant` maps an
  RSL-RL agent config to its checkpoint variant.
