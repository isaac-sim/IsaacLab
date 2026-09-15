Fixed
^^^^^

* Fixed :meth:`~isaaclab.managers.TerminationManager.reset` reporting termination
  statistics computed over all environments even when resetting only a subset
  through ``env_ids``. Reported means now consider only the selected rows,
  consistent with the reward manager's subset logging.
