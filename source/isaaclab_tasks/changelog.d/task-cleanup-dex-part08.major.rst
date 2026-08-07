Added
^^^^^

* Added manager-based counterparts for the Shadow cube reorientation task and its
  OpenAI FF/LSTM variants, alongside the existing Allegro manager task.
* Added :class:`~isaaclab_tasks.core.reorient.mdp.reorient_timeout`, which restarts
  the episode timer on every goal reach so OpenAI-variant episodes extend across
  success streaks.
* Added ``enable_domain_randomization`` to the Allegro and Shadow manager
  environments.
* Added Newton and OvPhysX presets to the manager-based reorientation environments,
  selectable with ``physics=``.
* Added a Direct-versus-manager value-parity test covering timing, success tolerance,
  fall distance, and the consecutive-success cap.

Changed
^^^^^^^

* **Breaking:** Changed ``Metrics/success_rate`` on the manager-based reorientation
  tasks to a per-episode success bit drawn at
  ``ReorientCommandCfg.success_count_threshold``, matching the Direct environments,
  instead of a per-attempt ratio. Curves from earlier runs are not comparable.
* **Breaking:** Changed domain randomization to default off on the Allegro and Shadow
  manager tasks and on for the OpenAI variants, matching each task's Direct
  counterpart. Set ``enable_domain_randomization`` to restore it.
* **Breaking:** Changed the manager-based Allegro environment to match the Direct
  observation, action, reset, and termination contracts, and to use the same agent
  configurations. The observation space changes size, so existing manager checkpoints
  must be retrained. ``rl_games_manager_ppo_cfg.yaml``, ``skrl_manager_ppo_cfg.yaml``
  and ``AllegroCubePPORunnerCfg`` are gone; use ``rl_games_ppo_cfg.yaml``,
  ``skrl_ppo_cfg.yaml`` and ``AllegroHandPPORunnerCfg``.
* **Breaking:** Moved the Shadow Hand camera benchmark task to the contributed tasks
  as ``IsaacContrib-Reorient-Cube-Shadow-Camera-Benchmark-Direct``. The released
  ``Isaac-Reorient-Cube-Shadow-Camera-Benchmark-Direct`` identifier no longer
  resolves; use the contributed one.
* **Breaking:** Replaced the per-backend in-hand cube preset with one configuration
  per hand. Newton spawned a 54 mm articulation at ``z=0.535``; both backends now
  share the PhysX rigid body, a 60 mm cube at ``z=0.6``. Newton checkpoints must be
  retrained, and code resolving the object through
  :class:`~isaaclab.assets.Articulation` must use :class:`~isaaclab.assets.RigidObject`.
* Renamed the per-robot scene constants to name what they hold: ``ROBOT_CFG`` becomes
  ``SHADOW_HAND_ROBOT_CFG`` or ``ALLEGRO_HAND_ROBOT_CFG``, and ``OBJECT_CFG`` becomes
  ``CUBE_CFG``.
* Renamed the manager-based Allegro configurations after the robot rather than the object,
  matching the Shadow counterparts: ``AllegroCubeEnvCfg`` and ``AllegroCubeSceneCfg`` become
  ``AllegroHandManagerEnvCfg`` and ``AllegroHandManagerSceneCfg``. The
  ``Isaac-Reorient-Cube-Allegro`` task identifier is unchanged.

Removed
^^^^^^^

* Removed ``ReorientObjectEnvCfg`` and the shared reorientation observation, action,
  and command configurations. Each manager task declares its own; derive from
  :class:`~isaaclab.envs.ManagerBasedRLEnvCfg` directly.
* Removed ``reorient_common``. Its constants are declared by the tasks that use them,
  and the in-hand offset and goal-marker position are per-robot fields on the Direct
  configurations.
* Removed the handover ``EventCfg``, which was never wired into
  ``HandoverEnvCfg.events``.
* Removed ``isaaclab_tasks.core.utils``. Its helpers moved to
  ``isaaclab_tasks.core.reorient.utils``, which the hand-over task imports.
* Removed the ``clone_in_fabric`` settings from the reorientation scenes. The flag no
  longer reaches the replicator, so the value had no effect.

Fixed
^^^^^

* Fixed the manager-based reorientation tasks not reporting ``Metrics/success_rate``.
