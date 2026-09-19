Added
^^^^^

* Added opt-in Newton model-free solvers for differential IK, joint impedance, and operational
  space controllers through ``cfg.use_newton=True``, independently of the simulation backend.
  Preserved the original Torch solvers by default and retained the constructor and compute APIs.
  Newton workspace was initialized from compute inputs or supplied DiffIK limits; CUDA graph users must warm up
  before capture and recapture after changing joint counts.
* Documented the opt-in Newton float32 precision, OSC selection before inertia decoupling,
  six-joint decoupling requirement, and conditional mass weighting of null-space posture efforts.
  Existing tasks must validate gains and checkpoints before opting in; the default path retained
  the original behavior.

Changed
^^^^^^^

* **Breaking (Newton opt-in only):** Required differential IK joint limits before the first compute
  when joint-limit avoidance was enabled. Call ``set_joint_pos_limits()`` before computing with
  a positive ``joint_limit_avoidance_gain``; later limit updates remained supported in place.
  The default Lab backend retained its previous behavior.

Fixed
^^^^^

* Fixed variable joint-impedance gain clamping for robot batches with more than two joints.
* Fixed batched joint-impedance inertia compensation when the robot and joint counts differed.
* Fixed the original differential IK SVD solver for position-only and under-actuated tasks.
