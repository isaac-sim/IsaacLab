Added
^^^^^

* Added ``FileCfg.visual_material_path=None`` to apply material inputs to authored shaders without
  replacing their textures or bindings. Fields set to ``None`` were left unchanged.
* Added OmniPBR metallic and texture-influence settings to ``PbrMdlCfg`` and shared
  ``modify_visual_material`` between material creation and prototype overrides.
