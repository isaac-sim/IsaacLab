Added
^^^^^

* Added :attr:`~isaaclab.cloner.ClonePlan.env_template`, the destination template for one
  environment. Every row's destination is that template followed by the asset's path below the
  environment, so it names the part a clone varies while the remainder is shared. It was
  previously a constructor argument that the plan discarded, leaving a consumer holding a row
  unable to recover it -- a destination carries no mark of where the environment ends. Backend
  replication contexts receive it alongside ``global_paths``.
