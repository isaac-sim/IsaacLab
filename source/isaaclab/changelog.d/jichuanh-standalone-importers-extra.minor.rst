Added
^^^^^

* Added an ``importers`` extra carrying the standalone URDF and MJCF importers, so conversion
  works without Isaac Sim. It cannot be combined with the ``isaacsim`` extra, which ships its own
  copies of the same importers.

* Added :attr:`~isaaclab.sim.converters.AssetConverterBaseCfg.physics_variant` to choose which
  ``"Physics"`` variant the URDF and MJCF converters select on the generated USD file.

Fixed
^^^^^

* Fixed URDF and MJCF conversion producing assets with no joints, articulation roots, or mass
  properties.

* Fixed MJCF conversion failing with ``Cannot find a valid schema for 'MjcSceneAPI'`` when another
  package queried a USD schema first.

* Fixed :meth:`~isaaclab.utils.dict.class_to_dict` expanding enum values into their internal
  members, which wrote unusable entries into serialized configurations.

Changed
^^^^^^^

* Changed :func:`~isaaclab.sim.utils.select_usd_variants` to raise for a variant set listed in
  :obj:`~isaaclab.sim.utils.REQUIRED_VARIANT_SETS` that is absent or does not offer the requested
  variant. Other variant sets still log a warning and continue.

* Changed ``./isaaclab.sh --install`` to reject extras that the root ``pyproject.toml`` declares
  conflicting, instead of installing a combination ``uv sync`` refuses.
