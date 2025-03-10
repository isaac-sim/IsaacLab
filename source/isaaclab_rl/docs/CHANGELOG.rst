Changelog
---------

0.2.0 (2025-03-05)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^
- Optimized Stable-Baselines3 wrapper ``Sb3VecEnvWrapper`` (now 4x faster) by using Numpy buffers and only logging episode and truncation information by default.

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
