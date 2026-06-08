Fixed
^^^^^

* Fixed Newton visualizer HUD dependency checks by requiring
  ``typing-extensions>=4.15.0`` for the Newton visualizer extra and failing
  integration tests when Newton reports that ``imgui_bundle`` could not be
  imported.

* Fixed Rerun and Viser visualizers rendering Newton infinite ground planes too
  small by expanding non-positive plane extents to the same large finite size
  used by Newton GL.

* Fixed Viser visualizer ground-grid flickering by reusing unchanged plane grid
  line segments instead of removing and re-adding them every frame.
