* Added native SuperMimic training and playback through the unified reinforcement-learning entrypoints,
  with MLP and optional one-frame NeRD dynamics, learner-owned replay, resumable checkpoints and W&B uploads.
* Added joint motion and velocity training with a shared decoder and world model, separate critics,
  isolated encoder gradients, resumable optimizer state and held-command velocity playback.
