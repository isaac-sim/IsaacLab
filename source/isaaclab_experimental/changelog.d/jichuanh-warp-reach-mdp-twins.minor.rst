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
