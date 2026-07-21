Added
^^^^^

* Added Warp-native uniform pose and velocity command terms with pointer-stable
  command arrays.
* Added :class:`~isaaclab_experimental.managers.CurriculumManager` and a
  boolean-mask reset path for Warp manager-based environments, retaining compact
  environment IDs only for legacy host consumers.
* Added Warp-native terrain curricula backed by cached zero-copy Warp views of
  terrain and scene origins.
* Added a boolean ``env_mask`` option to :meth:`ManagerBasedEnvWarp.reset_to`, the
  ``ISAACLAB_WARP_CAPTURE`` environment variable to force eager execution, and
  :attr:`WarpGraphCache.captured_stages` introspection for capture-coverage checks.

Changed
^^^^^^^

* **Breaking:** Removed the experimental ``ManagerCallSwitch`` and
  ``MANAGER_CALL_CONFIG`` interface. Warp environments now instantiate Warp
  managers directly and automatically keep managers with non-capturable terms
  on eager execution.

Fixed
^^^^^

* Fixed Warp event state leaking across environments and configurations, and
  corrected center-of-mass sampling and termination metrics for partial resets.
* Fixed captured velocity-command terms to read current root state on every
  replay instead of stale lazy-derived buffers.
* Fixed identity quaternion initialization for the Warp-native uniform pose
  command term.
