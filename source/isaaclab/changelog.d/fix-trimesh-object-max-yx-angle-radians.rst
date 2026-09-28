Fixed
^^^^^

* Fixed :func:`~isaaclab.terrains.trimesh.utils.make_box`, :func:`~isaaclab.terrains.trimesh.utils.make_cylinder`
  and :func:`~isaaclab.terrains.trimesh.utils.make_cone` ignoring the ``max_yx_angle`` limit when it is given in
  radians (``degrees=False``). The random tilt could exceed the limit by a factor of pi, so objects capped at 30
  degrees could tip over by up to about 94 degrees.
