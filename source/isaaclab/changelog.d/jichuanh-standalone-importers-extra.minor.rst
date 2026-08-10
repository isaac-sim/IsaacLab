Added
^^^^^

* Added an ``importers`` extra carrying the standalone URDF and MJCF importers, so conversion works
  without Isaac Sim. It cannot be combined with the ``isaacsim`` extra.

* Added :attr:`~isaaclab.sim.converters.AssetConverterBaseCfg.physics_variant` to choose which
  ``"Physics"`` variant the URDF and MJCF converters select.

Fixed
^^^^^

* Fixed ``scripts/tools/convert_urdf.py`` and ``scripts/tools/convert_mjcf.py`` crashing when the
  converted asset was previewed with a kitless visualizer (``--viz newton``, ``--viz rerun``, or
  ``--viz viser``), by selecting the physics backend that matches the runtime.

* Fixed URDF and MJCF conversion producing assets with no joints, articulation roots, or mass
  properties.

* Fixed MJCF conversion failing with ``Cannot find a valid schema for 'MjcSceneAPI'``.

* Fixed installation failures caused by overlapping standalone USD providers by using
  ``usd-exchange`` on all supported platforms and installing required Newton mesh-processing
  packages directly.

* Fixed :meth:`~isaaclab.utils.dict.class_to_dict` expanding enum values into their internal
  members.

Changed
^^^^^^^

* Changed ``./isaaclab.sh --install`` to reject extras that ``pyproject.toml`` declares conflicting,
  and to reject one that conflicts with what the environment already has. Drop one of the tokens, or
  install into a fresh environment.
