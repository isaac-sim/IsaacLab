Added
^^^^^

* Added manager-based counterparts for the Allegro and Shadow cube
  reorientation tasks (state and OpenAI FF/LSTM observation variants), sharing
  the Direct tasks' scalar parameters and boolean success metrics through
  common MDP terms.
* Added opt-in domain randomization to the manager-based Allegro environment
  (``enable_domain_randomization``, disabled by default; enabling requires
  retraining).
* Added a Newton physics preset to the manager-based Allegro environment.

Deprecated
^^^^^^^^^^

* Deprecated the legacy manager-based reorientation configuration
  :class:`~isaaclab_tasks.core.reorient.reorient_manager_env_cfg.ReorientObjectEnvCfg`
  and the manager terms only it consumes (:func:`~isaaclab_tasks.core.reorient.mdp.success_bonus`,
  :func:`~isaaclab_tasks.core.reorient.mdp.track_pos_l2`,
  :func:`~isaaclab_tasks.core.reorient.mdp.track_orientation_inv_l2`,
  :func:`~isaaclab_tasks.core.reorient.mdp.max_consecutive_success`, and
  :func:`~isaaclab_tasks.core.reorient.mdp.object_away_from_goal`) in favor of
  the Direct-compatible manager configurations and terms (e.g.
  :class:`~isaaclab_tasks.core.reorient.mdp.ReorientReward` and
  :class:`~isaaclab_tasks.core.reorient.mdp.ReorientTimeout`).

Changed
^^^^^^^

* **Breaking:** Changed the manager-based Allegro reorientation environment to
  match the Direct observation, action, reward, reset, termination, success,
  asset, and benchmark contracts. Existing manager checkpoints are
  incompatible and must be retrained.
