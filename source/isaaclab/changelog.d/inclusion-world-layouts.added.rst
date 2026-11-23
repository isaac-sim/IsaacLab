* Added an overridable :meth:`~isaaclab.cloner.InclusionSet.worlds` definition that lets task-local
  inclusion rules select compatible spawner configurations using their existing semantic tags.
  Scene planning resolved these selections into prototype IDs before replication, without enumerating
  excluded Cartesian combinations or adding a separate variant-label system.
