Added
^^^^^

* Added an opt-in ``compute_final_obs`` flag (default ``False``) to
  :class:`~isaaclab.envs.DirectRLEnvCfg`, :class:`~isaaclab.envs.DirectMARLEnvCfg`, and
  :class:`~isaaclab.envs.ManagerBasedRLEnvCfg` that captures the terminal observation before a
  Same-Step autoreset and exposes it through ``extras["final_obs"]``. The captured observation has
  the same observation noise applied as the returned observation. When the flag is ``False`` the
  previous behavior is preserved (no capture, no extra observation computation).
* Declared the Same-Step autoreset mode in RL environment metadata.
