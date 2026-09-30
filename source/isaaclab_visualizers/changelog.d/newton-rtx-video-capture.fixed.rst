* Fixed the Newton RTX visualizer dropping visualization markers, so both the interactive
  ``--viz newton_rtx`` viewer and videos captured through it now draw the goal poses, command
  arrows, and other debug markers the GL viewer already showed, sanitizing the marker group
  ids into valid USD prim paths the RTX stage accepts.
* Fixed the Newton RTX visualizer showing a black window and failing headless frame capture with
  ``ovrtx`` 0.5, which keys render outputs by prim path (``/Render/Vars/LdrColor``) while Newton's
  ``ViewerRTX`` looks up ``LdrColor``. The viewer now aliases each output under its short name.
