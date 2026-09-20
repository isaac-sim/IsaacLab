Changed
^^^^^^^

* Unified the coding style of the core task packages: module docstrings, section banners, import style, gym
  registration layout, stub (``.pyi``) layout and ``ManagerTermBase`` constructor signatures now follow one convention.
* Moved the duplicated ``survival_success_rate``, ``terminated_penalty``, ``joint_pos_target_l2``,
  ``DifficultyScheduler`` and ``initial_final_interpolate_fn`` terms to :mod:`isaaclab.envs.mdp`. The task ``mdp``
  packages keep exposing them through their :mod:`isaaclab.envs.mdp` fallback, so ``mdp.<term>`` references in task
  configurations keep working. Import them from :mod:`isaaclab.envs.mdp` instead of the task packages.
* Moved the fourbar-pole ``joint_pos_cos`` and ``joint_pos_sin`` observation terms from ``mdp/rewards.py`` to
  ``mdp/observations.py``; they remain available as ``mdp.joint_pos_cos`` and ``mdp.joint_pos_sin``.
* Shared the physics, camera and asset presets of the direct and manager-based cartpole, Ant and Humanoid tasks
  through new ``cartpole_common``, ``ant_common`` and ``humanoid_common`` modules instead of duplicating them.
* Renamed the private ``_FrankaSoftSceneCfg`` and ``_FrankaSoftCameraSceneCfg`` scene configurations of the Franka
  soft-body tasks to the public ``FrankaSoftBaseSceneCfg`` and ``FrankaSoftBaseCameraSceneCfg``.
* Replaced the deprecated ``viewer`` settings of the handover and Franka soft-body tasks with
  ``sim.default_visualizer_cfg``.
* Registered a default agent for the ``Isaac-Shadow-Handover``, ``Isaac-Lift-Cable-Franka`` and
  ``Isaac-Lift-Cable-Franka-Camera`` tasks.

Fixed
^^^^^

* Fixed the lift ADR curriculum interpolating the point-cloud noise upper bound towards ``-0.01`` instead of ``0.01``.
* Fixed the ``LiftEnvCfg`` configuration class missing the ``@configclass`` decorator.
* Fixed the in-hand reorientation keypoint helpers rebuilding constant corner offsets on the device every step, and
  the lift deformable and cable out-of-bounds terminations allocating constant bound tensors every step.
* Fixed the lift, handover and reorientation tasks rebuilding per-step index and origin tensors with ``repeat`` where a
  broadcast suffices.
* Fixed docstrings stating a ``(w, x, y, z)`` quaternion order in the lift, handover, deploy and keyboard task
  packages; Isaac Lab uses ``(x, y, z, w)``.
