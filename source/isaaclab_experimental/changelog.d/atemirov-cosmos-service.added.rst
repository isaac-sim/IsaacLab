* Added batched Cosmos camera views: each session requests its camera count and generates the environments
  as one transformer batch, each with its own prompt, seed, history, and episode resets (compiled runtime).
  New sessions can change the count without restarting the server, subject to GPU memory and transport limits.
* Added per-episode prompts: ``CosmosModelCfg.prompt`` also accepts a list, cycled per episode with one view and
  assigned per view with several; the service rebuilds the text conditioning at each episode reset while the
  model stays loaded.
* Added an optional ``isaaclab-cosmos-server --max-episode-frames`` cap, reported by ``status``. The default
  is uncapped (``0``); each session requests its own finite episode budget.
* Added the ``blur`` Cosmos control: cameras send their RGB and the service blurs it with the Cosmos Framework's
  own filter, the Sim-Transfer recipe's medium preset. :func:`~isaaclab_experimental.cosmos.blur_processor`
  builds the camera chain.
