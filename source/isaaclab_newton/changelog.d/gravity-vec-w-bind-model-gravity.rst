Fixed
^^^^^

* Fixed :attr:`~isaaclab_newton.assets.ArticulationData.GRAVITY_VEC_W` (and the
  matching attribute on :class:`~isaaclab_newton.assets.RigidObjectData` and
  :class:`~isaaclab_newton.assets.RigidObjectCollectionData`) to bind directly
  to Newton's per-world ``model.gravity`` array. Previously the constructor
  snapshotted env 0's gravity as a unit vector and broadcast it to every
  environment, so per-env gravity randomization (e.g.
  :class:`~isaaclab.envs.mdp.randomize_physics_scene_gravity`) was invisible to
  every consumer of ``GRAVITY_VEC_W`` and to the lazily-recomputed
  :attr:`~isaaclab_newton.assets.ArticulationData.projected_gravity_b`.
  ``GRAVITY_VEC_W`` now carries the actual world-frame gravity vector
  (m/s\\ :sup:`2`); ``projected_gravity_b`` continues to expose a unit vector
  in the body frame via a new dedicated kernel that normalizes internally.
