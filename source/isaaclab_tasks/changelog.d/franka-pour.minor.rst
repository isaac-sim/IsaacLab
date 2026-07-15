Added
^^^^^

* Added the ``Isaac-Pour-Franka-v0`` contributed task, where a Franka pours
  granular MPM media between scene-owned cups using proxy-coupled MJWarp and
  implicit-MPM solvers.
* Added both staged procedural resets and a reset-dataset training preset. The
  latter uses offline rejection sampling, Newton IK and collision validation,
  adaptive competence-weighted replay, general fixed-weight rewards, and
  particle-based success.
* Added CUDA-graph-captured sparse-grid training with isolated MPM worlds,
  sparse-grid playback, visible MPM particles, video-friendly camera framing,
  and SpaceMouse teleoperation presets for Franka Pour.

Deprecated
^^^^^^^^^^

* Deprecated the experimental ``Reset-Mixture`` task, configuration, runner,
  and curriculum names in favor of their ``Reset-Dataset`` counterparts. The
  compatibility names do not make older Cartesian-IK policy checkpoints
  compatible with the new relative-joint policy.
