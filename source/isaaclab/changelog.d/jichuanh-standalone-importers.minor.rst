Added
^^^^^

* Added the standalone URDF and MJCF importers as base dependencies, so conversion works without
  Isaac Sim and without an extra install step.

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

* Changed :meth:`~isaaclab.sim.utils.select_usd_variants` to raise when a variant set exists on the
  prim but does not offer the requested variant, which includes
  :attr:`~isaaclab.sim.UsdFileCfg.variants` at spawn time. USD accepts such a selection and composes
  the prim as if nothing were selected, so the asset used to spawn silently without what the variant
  carries. A variant set the prim does not have is still skipped with a warning. Set a variant the
  asset offers, or drop the entry.
