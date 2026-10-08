* Added batched Cosmos camera views: ``isaaclab-cosmos-server --max-views N`` lets one session generate the
  cameras of up to N environments as one transformer batch, each with its own prompt, seed, history, and
  episode resets (compiled runtime). ``CosmosModelCfg.prompt`` lists assign one prompt per environment.
  :func:`~isaaclab_experimental.cosmos.service_capabilities` reports the running service's capabilities.
