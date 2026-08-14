Added
^^^^^

* Added Warp MDP twins for :func:`~isaaclab_tasks.core.locomotion.mdp.terminated_penalty` and
  :class:`~isaaclab_tasks.core.locomotion.mdp.survival_success_rate`, so ``Isaac-Ant`` and
  ``Isaac-Humanoid`` run under ``--frontend warp``.

Fixed
^^^^^

* Fixed the Warp orientation-error twins reporting a smaller rotation error than the stable
  terms for non-unit quaternions, by computing the angle as ``2*atan2(|xyz|, |w|)`` instead of
  ``2*acos(|w|)``.
