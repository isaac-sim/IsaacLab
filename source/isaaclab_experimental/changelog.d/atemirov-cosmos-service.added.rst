* Added batched Cosmos camera views: ``isaaclab-cosmos-server --max-views N`` lets one session generate the
  cameras of up to N environments as one transformer batch, each with its own prompt, seed, history, and
  episode resets (compiled runtime).
* Added per-episode prompts: ``CosmosModelCfg.prompt`` also accepts a list, cycled per episode with one view and
  assigned per view with several; the service rebuilds the text conditioning at each episode reset while the
  model stays loaded.
* Added ``isaaclab-cosmos-server --max-episode-frames`` (default 201, the model's trained horizon; ``0`` for no
  cap), reported by ``status``. The server warns when the cap allows episodes longer than the trained horizon.
* Added the ``blur`` Cosmos control: cameras send their RGB and the service blurs it with the Cosmos Framework's
  own filter, the Sim-Transfer recipe's medium preset. :func:`~isaaclab_experimental.cosmos.blur_processor`
  builds the camera chain.
