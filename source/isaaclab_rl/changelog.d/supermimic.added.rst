* Added native SuperMimic training and playback through the unified reinforcement-learning entrypoints,
  with MLP and optional one-frame NeRD dynamics, learner-owned replay, resumable checkpoints and W&B uploads.

* Reused PhysicsNeMo Robotics online normalization for both model architectures, with persistent
  device statistics, compatible checkpoint buffers and differentiable imagined policy rollouts.
