Added
^^^^^

* Added ``motion_vectors`` to the rendering correctness test matrix for cartpole, shadow hand, and
  dexsuite kuka allegro lift environments.

Fixed
^^^^^

* Fixed flaky ``motion_vectors`` golden-image comparisons on PhysX backends (``physx`` and
  ``ovphysx``) by enabling enhanced determinism and per-iteration external forces on the PhysX
  solver, which otherwise produces run-to-run noisy velocities that this AOV encodes directly.
