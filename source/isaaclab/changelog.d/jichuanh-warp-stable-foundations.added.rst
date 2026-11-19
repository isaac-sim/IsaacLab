* Added the ``joint_ids_wp`` and ``body_ids_wp`` fields to :class:`~isaaclab.managers.SceneEntityCfg`: ``int32``
  Warp arrays of the resolved joint and body indices, in the order of ``joint_ids`` and ``body_ids``, for Warp terms.
  The new :meth:`~isaaclab.managers.SceneEntityCfg.resolve_warp_ids`, which the Warp managers call after
  :meth:`~isaaclab.managers.SceneEntityCfg.resolve`, fills them; :meth:`~isaaclab.managers.SceneEntityCfg.resolve`,
  ``joint_ids`` and ``body_ids`` are unchanged. Copies of the configuration, :func:`copy.deepcopy` included, share
  them; ``to_dict`` stores them as lists, and a pickled configuration leaves them out (``None``) until
  :meth:`~isaaclab.managers.SceneEntityCfg.resolve_warp_ids` runs again.
* Added :attr:`~isaaclab.scene.InteractiveScene.env_origins_wp` and
  :attr:`~isaaclab.terrains.TerrainImporter.env_origins_wp`, the ``wp.vec3f`` Warp array that holds the environment
  origins, for Warp code. ``env_origins`` stays a :class:`torch.Tensor`, now a view of that array: assigning
  ``TerrainImporter.env_origins`` writes the new origins into it, so an earlier view of either stays current.
