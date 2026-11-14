* Fixed :meth:`~isaaclab_mimic.locomanipulation_sdg.path_utils.ParameterizedPath.find_nearest` returning a
  distance along the path past a corner or before the start when the nearest point is a segment endpoint, which
  moved the locomanipulation SDG pure pursuit target too far ahead or held it at the path start.
