Changed
^^^^^^^

* Updated all play entrypoints (``play_rsl_rl``, ``play_sb3``, ``play_skrl``,
  ``play_rl_games``) to use :func:`~isaaclab_rl.entrypoints.common.create_isaaclab_env`
  instead of bare ``gym.make``, restoring warp frontend support and MARL-to-single-agent
  conversion at play time (parity with the train entrypoints).

* Replaced the ``gym.wrappers.RecordVideo`` wrapper approach with
  :func:`~isaaclab_rl.entrypoints.common.apply_video_recording`, which injects a
  :class:`~isaaclab.envs.utils.video_recorder_cfg.VideoRecorderCfg` into the env config
  before creation so recording is driven inside ``env.step()`` rather than via a wrapper.
