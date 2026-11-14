* Fixed environments of heterogeneous clone plans being listed out of index order on Isaac Sim PhysX. Sources are
  authored into the first environment of their world prototype, so the environment roots were created out of order,
  and stage queries and PhysX views followed that order. Asset, sensor and view rows did not match the environment
  index of the clone plan, for example the environment origins and per-world variant data, and resets wrote to
  other environments. :class:`~isaaclab.cloner.UsdReplicateContext` now orders the environment roots by index.
