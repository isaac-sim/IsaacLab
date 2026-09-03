Fixed
^^^^^

* Fixed :func:`~isaaclab.terrains.trimesh.mesh_terrains.star_terrain` failing with
  ``AttributeError: module 'numpy' has no attribute 'math'``. The function used the ``np.math``
  alias, which was removed in NumPy 2.0, and now uses the standard library ``math`` module instead.
