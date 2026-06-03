Added
^^^^^

* Added rsl_rl vision training configs for the cartpole camera tasks. The
  :obj:`Isaac-Cartpole-Camera-Direct` and :obj:`Isaac-Cartpole-Camera` tasks
  now expose an ``rsl_rl_cfg_entry_point`` using a CNN policy for the raw
  RGB/depth pipelines, and :obj:`Isaac-Cartpole-Camera` additionally exposes an
  ``rsl_rl_feature_cfg_entry_point`` with an MLP policy for the pretrained
  ResNet18 and Theia-Tiny feature pipelines.

Changed
^^^^^^^

* Consolidated the cartpole physics-backend presets into a single shared
  ``CartpolePhysicsCfg``. The direct camera task (:obj:`Isaac-Cartpole-Camera-Direct`)
  now uses the same tuned Newton solver settings as the other cartpole tasks
  instead of the solver defaults.
* **Breaking:** The cartpole camera tasks (:obj:`Isaac-Cartpole-Camera-Direct`
  and :obj:`Isaac-Cartpole-Camera`) now emit channel-first ``[C, H, W]`` image
  observations instead of channel-last ``[H, W, C]``, matching the layout
  expected by the rsl_rl CNN policy. The bundled rl_games and skrl configs were
  updated accordingly (``permute_input: False`` / no input permute); custom
  agent configs that assumed channel-last observations must drop their
  NHWC-to-NCHW permute.
