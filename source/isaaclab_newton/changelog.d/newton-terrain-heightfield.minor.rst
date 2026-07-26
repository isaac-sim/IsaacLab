Added
^^^^^

* Added automatic conversion of height-field-tagged terrain collision meshes into Newton heightfield
  colliders when building the Newton model. A terrain mesh tagged by
  :class:`~isaaclab.terrains.TerrainImporter` is rasterized into a :class:`newton.Heightfield` through
  :meth:`newton.Heightfield.create_from_mesh` at the same horizontal resolution and skipped during USD
  import, so the MuJoCo solver compiles a heightfield instead of a multi-hundred-thousand-vertex mesh.
  This cuts solver-initialization time for terrain-based tasks (for ``Isaac-Velocity-Rough-AnymalD``
  the terrain's MuJoCo compile drops from ``~950 ms`` to ``~5 ms`` and solver initialization from
  ``~1.9 s`` to ``~0.85 s``).
