Added
^^^^^

* Added :meth:`~isaaclab.cloner.ClonePlan.env_root_transforms` to build per-environment root poses
  from the plan's env positions, as homogeneous ``[num_clones, 4, 4]`` transforms. Rotations are
  assumed to be identity, since the plan carries env-root translations only. Plans without positions
  yield identity transforms.
