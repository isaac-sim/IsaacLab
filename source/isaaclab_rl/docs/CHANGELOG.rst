Changelog
---------

0.1.5 (2025-06-17)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Relaxed upper range pin for protobuf python dependency for more permissive installation.

0.1.4 (2025-04-10)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added configurations for distillation implementation in RSL-RL.
* Added configuration for recurrent actor-critic in RSL-RL.


0.1.3 (2025-03-31)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed the location of :meth:`isaaclab_rl.rsl_rl.RslRlOnPolicyRunnerCfg._modify_action_space`
  to be called only after retrieving the dimensions of the environment, preventing errors
  related to accessing uninitialized attributes.


0.1.2 (2025-03-28)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added symmetry and curiosity-based exploration configurations for RSL-RL wrapper.


0.1.1 (2025-03-10)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added a parameter to clip the actions in the action space inside the RSL-RL wrapper.
  This parameter is set to None by default, which is the same as not clipping the actions.
* Added attribute :attr:`isaaclab_rl.rsl_rl.RslRlOnPolicyRunnerCfg.clip_actions` to set
  the clipping range for the actions in the RSL-RL on-policy runner.


0.1.0 (2024-12-27)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

Initial version of the extension.
This extension is split off from ``isaaclab_tasks`` to include the wrapper scripts for the supported RL libraries.

Supported RL libraries are:

* RL Games
* RSL RL
* SKRL
* Stable Baselines3
