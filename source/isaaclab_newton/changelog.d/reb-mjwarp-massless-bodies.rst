Fixed
^^^^^

* Fixed MuJoCo model compilation failing with "mass and inertia of moving bodies must be
  larger than mjMINVAL" when a USD asset authors massless dynamic frame bodies (e.g.
  end-effector frames). :class:`~isaaclab_newton.physics.NewtonMJWarpManager` now raises
  such bodies to a minimal mass and inertia before model finalization, mirroring MuJoCo's
  ``boundmass``/``boundinertia`` compiler options.
