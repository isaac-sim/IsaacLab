Changelog
---------

0.4.4 (2025-10-15)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Added onnxscript package to isaaclab_rl setup.py to fix onnxscript package missing issue in aarch64 platform.


0.4.3 (2025-10-15)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Isaac-Ant-v0's sb3_ppo_cfg default value, so it trains under reasonable amount of time.


0.4.2 (2025-10-14)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Updated opset version from 11 to 18 in RSL-RL OnnxPolicyExporter to avoid onnex downcast issue seen in aarch64.


0.4.1 (2025-09-09)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Made PBT a bit nicer by
* 1. added resume logic to allow wandb to continue on the same run_id
* 2. corrected broadcasting order in distributed setup
* 3. made score query general by using dotted keys to access dictionary of arbitrary depth


0.4.0 (2025-09-09)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Introduced PBT to rl-games.


0.3.0 (2025-09-03)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Enhanced rl-games wrapper to allow dict observation.


0.2.4 (2025-08-07)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Disallowed string values in ``sb3_ppo_cfg.yaml`` from being passed to ``eval()`` in
  :meth:`~isaaclab_rl.sb3.process_sb3_cfg`. This change prevents accidental or malicious
  code execution when loading configuration files, improving overall security and reliability.


0.2.3 (2025-06-29)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Support SB3 VecEnv wrapper to configure with composite observation spaces properly so that the cnn creation pipelines
  natively supported by sb3 can be automatically triggered


0.2.2 (2025-06-30)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Call :meth:`eval` during :meth:`forward`` RSL-RL OnnxPolicyExporter


0.2.1 (2025-06-26)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Relaxed upper range pin for protobuf python dependency for more permissive installation.


0.2.0 (2025-04-24)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Switched to a 3.11 compatible branch for rl-games as Isaac Sim 5.0 is now using Python 3.11.


0.1.5 (2025-04-11)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Optimized Stable-Baselines3 wrapper ``Sb3VecEnvWrapper`` (now 4x faster) by using Numpy buffers and only logging episode and truncation information by default.
* Upgraded minimum SB3 version to 2.6.0 and added optional dependencies for progress bar


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
