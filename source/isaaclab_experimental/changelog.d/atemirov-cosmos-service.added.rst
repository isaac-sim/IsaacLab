* Added batched Cosmos camera views: each session requests its camera count and generates the environments
  as one transformer batch, each with its own prompt, seed, history, and episode resets (compiled runtime).
  New sessions can change the count without restarting the server, subject to GPU memory and transport limits.
* Added per-episode prompts: ``CosmosModelCfg.prompt`` also accepts a list, cycled per episode with one view and
  assigned per view with several; the service rebuilds the text conditioning at each episode reset while the
  model stays loaded.
* Added :func:`~isaaclab_experimental.cosmos.episode_frame_budget`, which counts a camera's captures per episode
  as the environment and sensor schedule them and returns the Cosmos frame budget that holds them.
