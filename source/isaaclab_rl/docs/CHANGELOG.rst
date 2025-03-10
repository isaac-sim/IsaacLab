Changelog
---------

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
