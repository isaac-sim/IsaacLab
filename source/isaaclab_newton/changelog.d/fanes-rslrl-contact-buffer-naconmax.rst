Fixed
^^^^^

* Fixed :class:`~isaaclab_newton.physics.NewtonManager` sizing its contact buffer from the
  collision pipeline alone when ``use_mujoco_contacts=False``, which raised
  ``MuJoCo naconmax (25600) exceeds contacts.rigid_contact_max (3840)`` at reset whenever the
  MuJoCo Warp solver's ``nconmax`` demanded more contacts than the pipeline estimate. The
  buffer now grows to the solver's maximum contact count, matching the
  ``use_mujoco_contacts=True`` path.
