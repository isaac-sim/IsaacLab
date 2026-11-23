* Added explicit :attr:`~isaaclab.cloner.InclusionSet.layouts` and an overridable
  :meth:`~isaaclab.cloner.InclusionSet.worlds` definition for legal world pairings. Scene planning
  resolves these definitions before replication, without enumerating excluded Cartesian combinations.
