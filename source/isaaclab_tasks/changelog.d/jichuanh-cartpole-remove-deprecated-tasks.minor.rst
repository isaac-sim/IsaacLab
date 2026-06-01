Changed
^^^^^^^

* **Breaking:** Removed 35 per-variant Cartpole task IDs (7 Direct camera,
  4 manager-based camera, 15 proprioceptive showcase, 9 camera-based
  showcase) and the ``deprecated`` ``gym.register`` kwarg that flagged
  them as aliases. Use the consolidated tasks
  (``Isaac-Cartpole-Camera-Direct``, ``Isaac-Cartpole-Camera``,
  ``Isaac-Cartpole-Showcase-Direct``,
  ``Isaac-Cartpole-Camera-Showcase-Direct``) with ``presets=<name>``
  to select the variant:

  * ``Isaac-Cartpole-{RGB,Depth,Albedo,SimpleShading-*}-Camera-Direct-v0``
    and ``Isaac-Cartpole-Camera-Presets-Direct-v0`` →
    ``--task=Isaac-Cartpole-Camera-Direct [presets=<rgb|depth|albedo|simple_shading_*>]``.
  * ``Isaac-Cartpole-{RGB,Depth}-v0`` →
    ``--task=Isaac-Cartpole-Camera [presets=<rgb|depth>]``.
  * ``Isaac-Cartpole-RGB-{ResNet18,TheiaTiny}-v0`` →
    ``--task=Isaac-Cartpole-Camera --agent=rl_games_feature_cfg_entry_point presets=<resnet18|theia_tiny>``.
  * ``Isaac-Cartpole-Showcase-<Obs>-<Action>-Direct-v0`` →
    ``--task=Isaac-Cartpole-Showcase-Direct [--agent=skrl_<obs>_<action>_cfg_entry_point] presets=<obs>_<action>``.
  * ``Isaac-Cartpole-Camera-Showcase-<Obs>-<Action>-Direct-v0`` →
    ``--task=Isaac-Cartpole-Camera-Showcase-Direct [--agent=skrl_<obs>_<action>_cfg_entry_point] presets=<obs>_<action>``.
