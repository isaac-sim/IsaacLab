Fixed
^^^^^

* Fixed :func:`~isaaclab.envs.mdp.body_projected_gravity_b` to normalize
  ``GRAVITY_VEC_W`` before projecting it into the body frame, so the observation
  stays a unit direction when the Newton backend exposes gravity as a raw
  m/s\ :sup:`2` vector. This is a no-op for the PhysX/OvPhysX backends, whose
  ``GRAVITY_VEC_W`` is already unit length.
