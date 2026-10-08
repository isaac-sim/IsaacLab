* Added per-episode prompts: ``CosmosModelCfg.prompt`` also accepts a list, cycled per episode; the service
  rebuilds the text conditioning at each episode reset while the model stays loaded.
