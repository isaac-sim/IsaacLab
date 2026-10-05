* Added ``FileCfg.visual_material_path=None`` to apply material inputs to connected surface shaders
  without replacing textures or bindings. Fields set to ``None`` were left unchanged.
* Added OmniPBR metallic and texture-influence settings to ``PbrMdlCfg`` and shared a private input
  writer between material creation and prototype overrides.
