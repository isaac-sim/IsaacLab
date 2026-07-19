Added
^^^^^

* Added Warp-native uniform pose and velocity command terms with pointer-stable
  command arrays.
* Added :class:`~isaaclab_experimental.managers.CurriculumManager` and a
  boolean-mask reset path for Warp manager-based environments, retaining compact
  environment IDs only for legacy host consumers.

Changed
^^^^^^^

* Changed Warp manager and direct-environment stages to execute eagerly by
  default while stateful capture semantics are validated. Set
  ``MANAGER_CALL_CONFIG='{"default": 2}'`` for manager capture or
  ``ISAACLAB_WARP_DIRECT_CAPTURE=1`` for the legacy direct capture path.

Deprecated
^^^^^^^^^^

* Deprecated selecting stable managers inside ``ManagerBasedEnvWarp``. Use
  ``ManagerBasedEnv`` for Torch managers or mode ``1`` for the Warp frontend.

Fixed
^^^^^

* Fixed Warp event state leaking across environments and configurations, and
  corrected center-of-mass sampling and termination metrics for partial resets.
