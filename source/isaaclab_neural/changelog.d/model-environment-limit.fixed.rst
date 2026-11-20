* Checked ``WorldModelSpec.max_envs`` during world-model manager binding, rejecting scenes that exceed a model's
  declared environment limit before allocating scene buffers. ``None`` keeps the environment count unrestricted.
