Fixed
^^^^^

* Fixed the Newton GL visualizer's "Pause Rendering" button not reflecting the paused
  state after pressing :kbd:`Space`. Both controls now toggle the same underlying flag, so
  the button label and :meth:`~isaaclab_visualizers.newton.newton_visualizer.NewtonViewerGL.is_rendering_paused`
  stay in sync regardless of whether rendering was paused via the button or the keyboard shortcut.
  Also clarified the on-screen control hint from "Space - Pause/Resume" to "Space - Pause/Resume
  Rendering".
