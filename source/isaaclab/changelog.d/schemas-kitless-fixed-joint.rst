Changed
^^^^^^^

* Changed :meth:`~isaaclab.sim.schemas.modify_articulation_root_properties` to author
  the fixed-root-link world joint directly with USD on every backend instead of calling
  ``omni.physx.scripts.utils.createJoint``. This removes the ``omni.physx`` dependency
  from the spawn path, which previously raised ``ModuleNotFoundError`` when spawning
  fixed-base articulations on kitless backends (e.g. Newton).
