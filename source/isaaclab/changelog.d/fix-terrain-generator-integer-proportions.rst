Fixed
^^^^^

* Fixed :class:`~isaaclab.terrains.TerrainGenerator` raising a ``UFuncTypeError`` when every sub-terrain
  ``proportion`` is an integer. The proportions are now normalized as floats.
