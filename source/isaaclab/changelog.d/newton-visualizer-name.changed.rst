* **Breaking:** Renamed the ``newton_gl`` visualizer selector to ``newton``. ``newton_gl`` still selects it, and
  ``newton_rtx`` is a deprecated alias that launches ``newton`` with ``initial_renderer="rtx"``, so the window opens
  in RTX rendering, with a :class:`FutureWarning` saying the option is removed in the next release. ``viz:<type>``
  video sources and ``visualizer=newton_rtx`` presets select ``newton`` with the default OpenGL renderer.
