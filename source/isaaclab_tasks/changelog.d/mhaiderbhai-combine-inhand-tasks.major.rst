Changed
^^^^^^^

* **Breaking:** Consolidated the in-hand reorientation tasks into a single
  :mod:`isaaclab_tasks.core.reorient` package. The former ``isaaclab_tasks.core.allegro_hand``
  (direct) and ``isaaclab_tasks.core.inhand.config.allegro_hand`` (manager-based) tasks moved
  under :mod:`isaaclab_tasks.core.reorient.config.allegro_hand`, and ``isaaclab_tasks.core.shadow_hand``
  moved under :mod:`isaaclab_tasks.core.reorient.config.shadow_hand`. The shared direct base environment
  ``isaaclab_tasks.core.inhand_manipulation.inhand_manipulation_env.InHandManipulationEnv`` was
  renamed to :class:`isaaclab_tasks.core.reorient.reorient_direct_env.ReorientDirectEnv`, and the
  shared manager-based base configuration to
  :class:`isaaclab_tasks.core.reorient.reorient_manager_env_cfg.ReorientObjectEnvCfg`. Update imports
  such as ``from isaaclab_tasks.core.shadow_hand.shadow_hand_env_cfg import ShadowHandRobotCfg`` to
  ``from isaaclab_tasks.core.reorient.config.shadow_hand.shadow_hand_env_cfg import ShadowHandRobotCfg``.
* **Breaking:** Within :mod:`isaaclab_tasks.core.reorient.config.allegro_hand`, the workflow-specific
  config modules carry a ``_direct_`` / ``_manager_`` infix
  (:mod:`~isaaclab_tasks.core.reorient.config.allegro_hand.allegro_hand_direct_env_cfg` and
  :mod:`~isaaclab_tasks.core.reorient.config.allegro_hand.allegro_hand_manager_env_cfg`), and the colliding
  ``rl_games`` / ``skrl`` agent configs were renamed accordingly (e.g. ``rl_games_ppo_cfg.yaml`` →
  ``rl_games_direct_ppo_cfg.yaml`` / ``rl_games_manager_ppo_cfg.yaml``). The direct and
  manager-based ``rsl_rl`` runner configs (``AllegroHandPPORunnerCfg`` and ``AllegroCubePPORunnerCfg``)
  now live together in ``reorient.config.allegro_hand.agents.rsl_rl_ppo_cfg``.
* **Breaking:** Renamed the camera-based shadow hand task from ``Vision`` to ``Camera``. The env
  modules ``shadow_hand_vision_env`` / ``shadow_hand_vision_env_cfg`` became
  :mod:`~isaaclab_tasks.core.reorient.config.shadow_hand.shadow_hand_camera_env` /
  :mod:`~isaaclab_tasks.core.reorient.config.shadow_hand.shadow_hand_camera_env_cfg`, and the
  ``ShadowHandVision*`` classes (env, env cfg, runner cfg) became ``ShadowHandCamera*``.
* Promoted the task-local ``reset_joints_within_limits_range`` event term to the core
  :class:`~isaaclab.envs.mdp.events.reset_joints_within_limits_range`; it is still accessible via
  ``isaaclab_tasks.core.reorient.mdp`` through the core re-export.
* **Breaking:** Moved the multi-agent Shadow Hand Over task to the top-level
  :mod:`isaaclab_tasks.core.handover` package and renamed ``ShadowHandOverEnv`` /
  ``ShadowHandOverEnvCfg`` to :class:`~isaaclab_tasks.core.handover.handover_env.HandoverEnv` /
  :class:`~isaaclab_tasks.core.handover.handover_env_cfg.HandoverEnvCfg`.
* **Breaking:** Renamed the Gym environment IDs: dropped the ``-v0`` version suffix, renamed
  ``Repose-Cube`` to ``Reorient-Cube`` and the camera variants from ``Vision`` to ``Camera``. The
  manager-based workflow carries no workflow suffix while the direct workflow keeps ``-Direct``.
  Update ``gym.make`` / ``--task`` calls:

  * ``Isaac-Repose-Cube-Allegro-v0`` → ``Isaac-Reorient-Cube-Allegro``.
  * ``Isaac-Repose-Cube-Allegro-Play-v0`` → ``Isaac-Reorient-Cube-Allegro-Play``.
  * ``Isaac-Repose-Cube-Allegro-Direct-v0`` → ``Isaac-Reorient-Cube-Allegro-Direct``.
  * ``Isaac-Repose-Cube-Shadow-Direct-v0`` → ``Isaac-Reorient-Cube-Shadow-Direct``.
  * ``Isaac-Repose-Cube-Shadow-OpenAI-FF-Direct-v0`` → ``Isaac-Reorient-Cube-Shadow-OpenAI-FF-Direct``.
  * ``Isaac-Repose-Cube-Shadow-OpenAI-LSTM-Direct-v0`` → ``Isaac-Reorient-Cube-Shadow-OpenAI-LSTM-Direct``.
  * ``Isaac-Repose-Cube-Shadow-Vision-Direct-v0`` → ``Isaac-Reorient-Cube-Shadow-Camera-Direct``.
  * ``Isaac-Repose-Cube-Shadow-Vision-Direct-Play-v0`` → ``Isaac-Reorient-Cube-Shadow-Camera-Direct-Play``.
  * ``Isaac-Repose-Cube-Shadow-Vision-Benchmark-Direct-v0`` → ``Isaac-Reorient-Cube-Shadow-Camera-Benchmark-Direct``.
  * ``Isaac-Shadow-Hand-Over-Direct-v0`` → ``Isaac-Shadow-Handover-Direct``.
