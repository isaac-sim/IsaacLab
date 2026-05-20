Fixed
^^^^^

* Updated ``configclass`` imports in :mod:`isaaclab_visualizers.kit`,
  :mod:`isaaclab_visualizers.newton`, :mod:`isaaclab_visualizers.rerun`, and
  :mod:`isaaclab_visualizers.viser` visualizer configs to import from
  :mod:`isaaclab.utils.configclass` directly, matching the lazy-import layout
  introduced in :mod:`isaaclab.utils`.
* Updated ``test_visualizer_cartpole_integration`` to read the tiled camera
  RGB output via the ``.torch`` accessor, matching the Warp-backed camera
  data API.
