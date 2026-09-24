Added
^^^^^

* Added :mod:`isaaclab_newton.controllers` with :class:`~isaaclab_newton.controllers.NewtonDifferentialIKController`,
  :class:`~isaaclab_newton.controllers.NewtonJointImpedanceController`, and
  :class:`~isaaclab_newton.controllers.NewtonOperationalSpaceController`. They wrap the model-free controllers in
  :mod:`newton.controllers` for any physics backend, and their configurations expose every Newton option,
  including differential IK posture control, joint-impedance Coriolis and acceleration feedforward, and
  operational-space selection frames and live gains.
* Added :class:`~isaaclab_newton.envs.mdp.NewtonDifferentialInverseKinematicsActionCfg` and
  :class:`~isaaclab_newton.envs.mdp.NewtonOperationalSpaceControllerActionCfg` to drive the Newton controllers
  from manager-based environments.

Removed
^^^^^^^

* **Breaking:** Moved :mod:`isaaclab_newton.ik` to :mod:`isaaclab_newton.controllers.ik` without an alias. Replace
  imports such as ``from isaaclab_newton.ik import NewtonIKSolver`` with
  ``from isaaclab_newton.controllers.ik import NewtonIKSolver``, and ``isaaclab_newton.ik.<module>`` with
  ``isaaclab_newton.controllers.ik.<module>``.
