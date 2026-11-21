* Added a standalone model-based backend with differentiable world-model rollouts, actor-critic learning, and distributed training through the unified CLI, with state representations and replay storage grouped with their consuming modules.
* Grouped physical collection, replay, context normalization, and imagined trajectories in the rollout module.
* Bound differentiable FK and analytic IK to the loaded USD articulation, removing the separate MJCF input.
* Added model-based pretrained checkpoint playback and single-file reference downloads using the shared asset utilities.
