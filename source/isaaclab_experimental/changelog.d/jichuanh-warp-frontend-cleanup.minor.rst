Added
^^^^^

* Added Warp-native uniform pose and velocity command terms with pointer-stable
  command arrays.
* Added :class:`~isaaclab_experimental.managers.CurriculumManager` and a
  boolean-mask reset path for Warp manager-based environments, retaining compact
  environment IDs only for legacy host consumers.
* Added Warp-native terrain curricula backed by cached zero-copy Warp views of
  terrain and scene origins.

Changed
^^^^^^^

* Changed Warp manager stages to execute eagerly by default while stateful
  capture semantics are validated. Set
  ``MANAGER_CALL_CONFIG='{"default": 2}'`` to opt eligible manager stages into
  CUDA graph capture. Direct-environment stages remained eager.

Deprecated
^^^^^^^^^^

* Deprecated selecting stable managers inside ``ManagerBasedEnvWarp``. Use
  ``ManagerBasedEnv`` for Torch managers or mode ``1`` for the Warp frontend.

Fixed
^^^^^

* Fixed Warp event state leaking across environments and configurations, and
  corrected center-of-mass sampling and termination metrics for partial resets.
* Fixed captured velocity-command terms to read current root state on every
  replay instead of stale lazy-derived buffers.
