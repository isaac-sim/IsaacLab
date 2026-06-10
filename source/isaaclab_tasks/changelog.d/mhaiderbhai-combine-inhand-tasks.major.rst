Changed
^^^^^^^

* **Breaking:** Consolidated the in-hand manipulation / reorientation tasks into a single
  :mod:`isaaclab_tasks.core.inhand` package. The former ``isaaclab_tasks.core.allegro_hand``
  (direct) and ``isaaclab_tasks.core.inhand.config.allegro_hand`` (manager-based) tasks moved
  under :mod:`isaaclab_tasks.core.inhand.allegro_hand`, and ``isaaclab_tasks.core.shadow_hand``
  moved under :mod:`isaaclab_tasks.core.inhand.shadow_hand`. The shared direct base environment
  ``isaaclab_tasks.core.inhand_manipulation.inhand_manipulation_env.InHandManipulationEnv`` was
  renamed to :class:`isaaclab_tasks.core.inhand.inhand_direct_env.InHandDirectEnv`, and the
  shared manager-based base configuration moved from ``inhand.inhand_env_cfg`` to
  :mod:`isaaclab_tasks.core.inhand.inhand_manager_env_cfg`. Update imports such as
  ``from isaaclab_tasks.core.shadow_hand.shadow_hand_env_cfg import ShadowHandRobotCfg`` to
  ``from isaaclab_tasks.core.inhand.shadow_hand.shadow_hand_env_cfg import ShadowHandRobotCfg``.
* **Breaking:** Within :mod:`isaaclab_tasks.core.inhand.allegro_hand`, the workflow-specific
  config modules carry a ``_direct_`` / ``_manager_`` infix
  (:mod:`~isaaclab_tasks.core.inhand.allegro_hand.allegro_hand_direct_env_cfg` and
  :mod:`~isaaclab_tasks.core.inhand.allegro_hand.allegro_hand_manager_env_cfg`), and the colliding
  ``rl_games`` / ``skrl`` agent configs were renamed accordingly (e.g. ``rl_games_ppo_cfg.yaml`` →
  ``rl_games_direct_ppo_cfg.yaml`` / ``rl_games_manager_ppo_cfg.yaml``). The direct and
  manager-based ``rsl_rl`` runner configs (``AllegroHandPPORunnerCfg`` and ``AllegroCubePPORunnerCfg``)
  now live together in ``inhand.allegro_hand.agents.rsl_rl_ppo_cfg``.
* **Breaking:** Moved the multi-agent Shadow Hand Over task to the top-level
  :mod:`isaaclab_tasks.core.hand_over` package and renamed ``ShadowHandOverEnv`` /
  ``ShadowHandOverEnvCfg`` to :class:`~isaaclab_tasks.core.hand_over.hand_over_env.HandOverEnv` /
  :class:`~isaaclab_tasks.core.hand_over.hand_over_env_cfg.HandOverEnvCfg`.
* **Breaking:** Renamed the in-hand Gym environment IDs to drop the ``-v0`` version suffix. The
  manager-based workflow carries no workflow suffix while the direct workflow keeps ``-Direct``.
  Update ``gym.make`` / ``--task`` calls:

  * ``Isaac-Repose-Cube-Allegro-v0`` → ``Isaac-Repose-Cube-Allegro``.
  * ``Isaac-Repose-Cube-Allegro-Play-v0`` → ``Isaac-Repose-Cube-Allegro-Play``.
  * ``Isaac-Repose-Cube-Allegro-Direct-v0`` → ``Isaac-Repose-Cube-Allegro-Direct``.
  * ``Isaac-Repose-Cube-Shadow-Direct-v0`` → ``Isaac-Repose-Cube-Shadow-Direct``.
  * ``Isaac-Repose-Cube-Shadow-OpenAI-FF-Direct-v0`` → ``Isaac-Repose-Cube-Shadow-OpenAI-FF-Direct``.
  * ``Isaac-Repose-Cube-Shadow-OpenAI-LSTM-Direct-v0`` → ``Isaac-Repose-Cube-Shadow-OpenAI-LSTM-Direct``.
  * ``Isaac-Repose-Cube-Shadow-Vision-Direct-v0`` → ``Isaac-Repose-Cube-Shadow-Vision-Direct``.
  * ``Isaac-Repose-Cube-Shadow-Vision-Direct-Play-v0`` → ``Isaac-Repose-Cube-Shadow-Vision-Direct-Play``.
  * ``Isaac-Repose-Cube-Shadow-Vision-Benchmark-Direct-v0`` → ``Isaac-Repose-Cube-Shadow-Vision-Benchmark-Direct``.
  * ``Isaac-Shadow-Hand-Over-Direct-v0`` → ``Isaac-Shadow-Hand-Over-Direct``.
