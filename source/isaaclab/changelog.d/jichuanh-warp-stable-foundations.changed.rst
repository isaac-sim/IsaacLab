* :attr:`~isaaclab.utils.warp.ProxyArray.warp` and :attr:`~isaaclab.utils.warp.ProxyArray.torch` are instance
  attributes, so reading them costs a dictionary lookup instead of a property call (21 ns instead of 53 ns and
  63 ns). The torch view is still created on its first read, and ``wp.quatf`` views still warn on every read. A
  subclass creates its torch view with :func:`warp.to_torch` instead of ``super().torch``.
* **Breaking:** Changed :attr:`~isaaclab.scene.InteractiveScene.env_origins` and
  :attr:`~isaaclab.terrains.TerrainImporter.env_origins` to a :class:`~isaaclab.utils.warp.ProxyArray` of
  ``wp.vec3f``, so Warp code reads the origins without a conversion. Torch code reads ``env_origins.torch``, of shape
  ``(num_envs, 3)``. Indexing, arithmetic and torch functions on the ProxyArray itself still work with a one-time
  deprecation warning, but ``isinstance(..., torch.Tensor)``, ``.shape``, ``.dtype`` and ``.device`` now describe the
  Warp array, as for the position fields of the asset data.
* Changed the ``frame`` of :class:`~isaaclab.sensors.RayCasterCamera` and
  :class:`~isaaclab.sensors.MultiMeshRayCasterCamera` to a :class:`~isaaclab.utils.warp.ProxyArray`, like
  :attr:`Camera.frame <isaaclab.sensors.Camera.frame>`. Torch code reads ``frame.torch``.
