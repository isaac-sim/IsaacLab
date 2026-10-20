* Added :attr:`~isaaclab_experimental.cosmos.CosmosModelCfg.cameras` so several cameras share one Cosmos session:
  each capture generates all their environments as one batch, and each camera publishes its own views at its own
  output size. :func:`~isaaclab_experimental.cosmos.apply_cosmos` takes a list of camera names and renders them at one
  canvas; :func:`~isaaclab_experimental.cosmos.cosmos_camera` takes the canvas to use.
