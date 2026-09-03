Fixed
^^^^^

* Fixed ``--deterministic`` not making training runs reproducible. The flag configured PyTorch and
  the Isaac RTX renderer but never reached the physics solver, so runs on Newton backends stayed
  free-running and their reward curves diverged.

Changed
^^^^^^^

* Changed ``--deterministic`` to set :attr:`~isaaclab.physics.PhysicsCfg.deterministic` on the
  resolved physics config. The entrypoint no longer selects backend-specific determinism settings or
  validates solvers; each physics manager translates the request and rejects what it cannot support.
  Deterministic physics costs runtime and memory; drop the flag to opt out.
