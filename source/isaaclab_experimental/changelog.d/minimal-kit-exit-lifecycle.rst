Changed
^^^^^^^

* Changed the Warp environments to seed Replicator through :func:`~isaaclab.utils.seed.register_seed_hook`.
* Changed ``render()`` with ``render_mode="rgb_array"`` in the Warp environments to warn and return ``None``,
  matching the core environments. Use ``VideoRecorderCfg`` on ``env_cfg.video_recorders`` to capture frames.
