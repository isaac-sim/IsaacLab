Changed
^^^^^^^

* Changed :class:`~isaaclab_contrib.deformable.DeformableObject` to follow the backend's
  default physics context (``isaaclab_newton.cloner.PHYSICS_CONTEXT``) directed by the asset
  cfg: USD clones now accompany Newton replication only under Kit, instead of unconditionally,
  matching the other Newton assets.
