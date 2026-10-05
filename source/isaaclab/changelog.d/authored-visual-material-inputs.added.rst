* Added ``FileCfg.visual_material_path=None`` to apply material inputs to compatible, connected surface
  shaders without replacing their textures or bindings. Other shader types, utility nodes, and fields
  set to ``None`` were left unchanged. Instanceable materials required ``make_uninstanceable=True``.
* Added OmniPBR metallic and texture-influence settings to ``PbrMdlCfg`` and shared a private input
  writer between material creation and prototype overrides.
