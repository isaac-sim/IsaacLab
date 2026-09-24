Fixed
^^^^^

* Fixed :func:`~isaaclab_tasks.contrib.forge.forge_utils.change_FT_frame` applying the inverse rotation and the
  wrong lever-arm sign when re-expressing a force/torque reading in another frame. The FORGE force observation
  is unchanged because the environment uses identity rotations and only consumes the force components.
