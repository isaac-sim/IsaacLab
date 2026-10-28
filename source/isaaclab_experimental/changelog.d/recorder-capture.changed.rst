* **Breaking:** Removed the implicit observation computation before Warp RL post-step recorder callbacks.
  Capture runs before automatic reset. Custom terms needing observations must request them explicitly
  from the observation manager; ``env.obs_buf`` is no longer refreshed for recorder dispatch.
