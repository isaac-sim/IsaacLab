Fixed
^^^^^

* Fixed :class:`~isaaclab_newton.envs.mdp.NewtonInverseKinematicsAction` prototype
  construction for nested articulation roots and shared collision geometry.
* Fixed the first CUDA-graph-enabled Newton IK action so it replays the captured
  solve before writing its first joint target.
* Cached the immutable fixed-base clone-orientation check so Newton IK no longer
  synchronizes the device on every action.

Added
^^^^^

* Added configurable per-body rigid-contact capacity to
  :class:`~isaaclab_newton.physics.VBDSolverCfg`.
* Added an isolated-articulation model option to keep Newton IK solves compact in
  coupled scenes with unrelated articulated deformables.
* Added optional CUDA-graph replay for fixed-shape Newton IK manager actions.
