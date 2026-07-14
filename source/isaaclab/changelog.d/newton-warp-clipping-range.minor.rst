Added
^^^^^

* Added :func:`~isaaclab.utils.warp.warp_math.replace_background_depth_wp` to replace non-positive
  (background / beyond-far) depth values with a fill value, for ray tracers that use ``0.0`` rather
  than ``+inf`` as the depth background sentinel.
