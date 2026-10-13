* Added splitting a Cosmos camera's environments across several services, for example one per GPU:
  :attr:`~isaaclab_experimental.cosmos.CosmosModelCfg.endpoint` takes a list, and the environments are split into
  contiguous shares of near-equal size, one per service. Each step sends every service its share before waiting
  for any reply, so the services generate at the same time. A failure on any service closes every session.
