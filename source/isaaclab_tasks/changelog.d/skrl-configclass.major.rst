Changed
^^^^^^^

* **Breaking:** Changed core-task skrl agent configuration entry points to return
  ``SkrlRunnerCfg`` objects instead of dictionaries. Dictionary consumers must call
  ``skrl_cfg_to_dict`` before accessing the configuration. Hydra overrides must use
  ``gae_lambda`` instead of ``lambda`` and ``observation_preprocessor`` instead of
  ``state_preprocessor``; ``clip_predicted_values`` was removed.

  Custom YAML entry points remain supported and require no migration.
