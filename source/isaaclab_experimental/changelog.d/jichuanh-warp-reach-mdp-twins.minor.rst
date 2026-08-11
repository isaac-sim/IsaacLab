Added
^^^^^

* Added Warp MDP twins for :class:`~isaaclab.envs.mdp.is_terminated_term` and
  :func:`~isaaclab.envs.mdp.pose_command_success`, so ``Isaac-Reach-Franka`` and
  ``Isaac-Reach-UR10`` run under ``--frontend warp``.
* Added :meth:`~isaaclab_experimental.envs.frontend.WarpFrontend.check_compatibility`, which
  reports why an environment configuration cannot run on the Warp frontend instead of raising,
  so several configurations can be surveyed in one pass.
* Added :attr:`~isaaclab_experimental.managers.TerminationManager.term_dones_wp`, exposing the
  per-term done buffer to reward terms that aggregate a subset of termination terms.

Fixed
^^^^^

* Fixed the Warp :func:`~isaaclab_experimental.envs.mdp.pose_command_success` twin disagreeing
  with the stable term's orientation error for non-unit quaternions, by computing the angle as
  ``2*atan2(|xyz|, |w|)`` instead of ``2*acos(|w|)``. Both arguments scale with the quaternion,
  so the angle is now norm-invariant and environments near the configured
  ``orientation_success_threshold`` are classified the same way as on the stable path.
