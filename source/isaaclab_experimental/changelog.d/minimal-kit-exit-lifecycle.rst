Changed
^^^^^^^

* Changed ``seed()`` of the Warp environments to no longer seed Replicator; the Replicator event terms seed it
  with ``env.cfg.seed``.
* Changed ``render()`` with ``render_mode="rgb_array"`` in the Warp environments to warn and return ``None``,
  matching the core environments. Use ``VideoRecorderCfg`` on ``env_cfg.video_recorders`` to capture frames.
