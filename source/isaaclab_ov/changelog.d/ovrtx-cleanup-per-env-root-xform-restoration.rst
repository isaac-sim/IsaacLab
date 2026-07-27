Changed
^^^^^^^

* Changed the OVRTX renderer to author per-environment root transforms after cloning from
  :meth:`~isaaclab.cloner.ClonePlan.env_root_transforms`, instead of snapshotting and restoring them
  around the clone. This removes the pre-clone transform capture on both the OVRTX and ovstage
  paths.
