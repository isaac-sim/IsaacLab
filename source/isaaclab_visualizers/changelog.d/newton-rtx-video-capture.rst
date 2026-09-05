Fixed
^^^^^

* Fixed the Newton RTX visualizer dropping visualization markers, so both the interactive
  ``--viz newton_rtx`` viewer and videos captured through it now draw the goal poses, command
  arrows, and other debug markers the GL viewer already showed, sanitizing the marker group
  ids into valid USD prim paths the RTX stage accepts.
