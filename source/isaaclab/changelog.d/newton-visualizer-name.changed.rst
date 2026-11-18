* **Breaking:** Renamed the ``newton_gl`` visualizer selector to ``newton``. ``newton_gl`` still selects it, and
  ``newton_rtx`` is a deprecated alias that launches ``newton``, which supports RTX rendering, with a
  :class:`FutureWarning` saying the option is removed in the next release. The same applies to ``viz:<type>`` video
  sources and to ``visualizer=newton_rtx`` presets.
