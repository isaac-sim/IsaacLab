Fixed
^^^^^

* Fixed :func:`~isaaclab.cloner.replicate` and :class:`~isaaclab.cloner.ReplicateSession`
  to honor :attr:`~isaaclab.scene.InteractiveSceneCfg.replicate_physics`. When set to
  ``False``, physics-engine backends are now skipped so only USD geometry is replicated
  and the physics engine parses each environment individually. This removes spurious
  ``Replication of this type is not supported`` errors for deformable objects under the
  PhysX backend.
