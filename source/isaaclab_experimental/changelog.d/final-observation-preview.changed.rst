* Warp RL environments now warn at creation when ``compute_final_obs`` is enabled and continue without terminal
  captures. Consumers can retain their existing fallback or reject missing terminal observations explicitly.
