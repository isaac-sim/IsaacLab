Added
^^^^^

* Added manager-based counterparts for the Shadow Hand cube reorientation task,
  ``Isaac-Reorient-Cube-Shadow`` and ``Isaac-Reorient-Cube-Shadow-Camera``, alongside the
  existing ``Isaac-Reorient-Cube-Allegro``.
* Added ``presets=asymmetric`` to ``Isaac-Reorient-Cube-Shadow``, which swaps the
  full-state actor for a reduced actor paired with a privileged critic and selects the
  matching ``rsl_rl`` observation groups.
* Added ``presets=randomized`` to the Allegro and Shadow manager tasks, which adds the
  domain-randomization events to the per-episode reset.
* Added Newton and OvPhysX presets to the manager-based reorientation environments,
  selectable with ``physics=``.

Changed
^^^^^^^

* **Breaking:** Moved the OpenAI Shadow Hand reorientation variants to the contributed
  tasks. Replace ``Isaac-Reorient-Cube-Shadow-OpenAI-FF-Direct`` with
  ``IsaacContrib-Reorient-Cube-Shadow-OpenAI-FF-Direct`` and
  ``Isaac-Reorient-Cube-Shadow-OpenAI-LSTM-Direct`` with
  ``IsaacContrib-Reorient-Cube-Shadow-OpenAI-LSTM-Direct``. The paper's
  training regime -- 20 Hz control, action and observation noise, and an episode budget
  spent per goal -- does not generalize, so it no longer ships in the core task.
* **Breaking:** Moved the Shadow Hand camera benchmark task to the contributed tasks as
  ``IsaacContrib-Reorient-Cube-Shadow-Camera-Benchmark-Direct``. The released
  ``Isaac-Reorient-Cube-Shadow-Camera-Benchmark-Direct`` identifier no longer resolves.
* **Breaking:** Changed ``Metrics/success_rate`` on the Direct reorientation tasks to the
  per-attempt success rate the manager-based tasks already report, ``goals reached /
  goals presented``. The Direct tasks previously reported a per-episode success bit, so the
  same metric name meant different things in the two workflows and their curves could not
  be compared. Direct curves from earlier runs are not comparable. Removed the
  ``success_count_threshold`` configuration field, which no longer has an effect.
* **Breaking:** Changed domain randomization to be opt-in through ``presets=randomized``
  on the Allegro and Shadow manager tasks, matching each task's Direct counterpart.
* **Breaking:** Changed the manager-based Allegro environment to match the Direct
  observation, action, reset and termination contracts, and to use the same agent
  configurations. The observation space changes size, so existing manager checkpoints must
  be retrained. ``rl_games_manager_ppo_cfg.yaml`` and ``skrl_manager_ppo_cfg.yaml`` are
  gone; use ``rl_games_ppo_cfg.yaml`` and ``skrl_ppo_cfg.yaml``.
* **Breaking:** Replaced the per-backend in-hand cube preset with one configuration per
  hand. Newton spawned a 54 mm articulation at ``z=0.535``; both backends now share the
  PhysX rigid body, a 60 mm cube at ``z=0.6``. Newton checkpoints must be retrained, and
  code resolving the object through :class:`~isaaclab.assets.Articulation` must use
  :class:`~isaaclab.assets.RigidObject`.
* Renamed the reorientation ``rsl_rl`` runner configurations after the workflow they
  serve: ``AllegroCubePPORunnerCfg`` becomes ``AllegroHandManagerPPORunnerCfg`` and the
  former ``AllegroHandPPORunnerCfg`` becomes ``AllegroHandDirectPPORunnerCfg``.
  ``ShadowHandPPORunnerCfg`` is unchanged and still serves the Direct task;
  ``ShadowHandManagerPPORunnerCfg`` adds the ``presets=asymmetric`` selection.
* Renamed the per-robot scene constants to name what they hold: ``ROBOT_CFG`` becomes
  ``SHADOW_HAND_ROBOT_CFG`` or ``ALLEGRO_HAND_ROBOT_CFG``, and ``OBJECT_CFG`` becomes
  ``CUBE_CFG``.
* Renamed the manager-based Allegro configurations after the robot rather than the object,
  matching the Shadow counterparts: ``AllegroCubeEnvCfg`` and ``AllegroCubeSceneCfg`` become
  ``AllegroHandManagerEnvCfg`` and ``AllegroHandManagerSceneCfg``. The
  ``Isaac-Reorient-Cube-Allegro`` task identifier is unchanged.

Removed
^^^^^^^

* Removed ``ReorientObjectEnvCfg``. The Shadow and Allegro manager tasks now share
  ``isaaclab_tasks.core.reorient.reorient_manager_env_cfg.ReorientManagerEnvBaseCfg``, which each hand
  specializes with its fingertip bodies, actuated joints, scene and control rate.
* Removed ``reorient_common``. Its constants are declared by the tasks that use them, and
  the in-hand offset and goal-marker position are per-robot fields on the Direct
  configurations.
* Removed the handover ``EventCfg``, which was never wired into ``HandoverEnvCfg.events``.
* Removed ``evaluate_reorient_success`` and ``reorient_reward`` from the reorientation
  ``mdp`` namespace. Neither is a manager MDP term; import them from
  ``isaaclab_tasks.core.reorient.mdp.rewards`` instead.
* Removed ``random_xy_rotation``. The manager reset uses the framework's
  :func:`~isaaclab.envs.mdp.reset_root_state_with_random_orientation`, leaving the helper
  without callers; the Direct and hand-over tasks compose their rotations with
  ``randomize_rotation`` directly.
* Removed the ``clone_in_fabric`` settings from the reorientation scenes. The flag no
  longer reaches the replicator, so the value had no effect.

Fixed
^^^^^

* Fixed the manager-based reorientation tasks not reporting ``Metrics/success_rate``.
