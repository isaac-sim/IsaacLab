Changed
^^^^^^^

* **Breaking:** Changed :class:`~isaaclab.assets.AssetBaseCfg` to construct an authoring-only
  :class:`~isaaclab.assets.Asset` by default, so every asset cfg now uses ``cfg.class_type(cfg)``.
  :attr:`~isaaclab.scene.InteractiveScene.extras` now contains these assets instead of their cfgs;
  access the original configuration through ``asset.cfg`` and the spawned prim through ``asset.prim``.
