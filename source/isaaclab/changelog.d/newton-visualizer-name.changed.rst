* **Breaking:** Renamed the ``newton_gl`` visualizer selector to ``newton``. ``newton_gl`` still selects it, and
  ``newton_rtx`` now raises a ``ValueError`` that points to ``newton``, which supports RTX rendering. The same applies
  to ``viz:<type>`` video sources.
