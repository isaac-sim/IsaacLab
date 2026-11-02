* Added a prototype ``is7`` visualizer backend (:class:`~isaaclab_visualizers.is7.IS7BackendCfg`) that shows an
  Isaac Lab simulation in the Isaac Sim 7 viewer application, with its stage tree, property and content panes.
  Isaac Lab keeps its own physics; the viewer host renders an export of the authored stage. A pose engine registered
  with the host writes Isaac Lab's body world matrices and a transform journal into the host's stage before each
  rendered frame (``IS7BackendCfg.pose_transport``). Instanceable prims are expanded within
  ``IS7BackendCfg.deinstance_max_points``, which makes per-frame pose updates about 2.6x faster on instanced assets.
