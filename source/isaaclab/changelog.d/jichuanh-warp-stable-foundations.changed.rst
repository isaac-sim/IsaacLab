* **Breaking:** Changed :attr:`~isaaclab.scene.InteractiveScene.env_origins` and
  :attr:`~isaaclab.terrains.TerrainImporter.env_origins` to a :class:`~isaaclab.utils.warp.ProxyArray` of
  ``wp.vec3f``, so Warp code reads the origins without a conversion. Torch code reads ``env_origins.torch``, of shape
  ``(num_envs, 3)``. Indexing, arithmetic and torch functions on the ProxyArray itself still work with a one-time
  deprecation warning, but ``isinstance(..., torch.Tensor)``, ``.shape``, ``.dtype`` and ``.device`` now describe the
  Warp array, as for the position fields of the asset data.
