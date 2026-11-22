* **Breaking:** Renamed the ``newton_gl`` visualizer selector to ``newton``. ``newton_gl`` still selects it, and
  ``newton_rtx`` stays a supported selector that launches ``newton`` with ``initial_renderer="rtx"``, so the window
  opens in RTX rendering with the Renderer toggle available. ``viz:<type>`` video sources select ``newton`` with the
  default OpenGL renderer.
