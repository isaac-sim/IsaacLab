Added
^^^^^

* Added ``visualizer=NAME[,...]`` alongside physics and renderer selectors. Accepted ``--physics``,
  ``--renderer``, and ``--visualizer`` (also ``--viz``) as aliases before task composition without
  forwarding preset names to Kit. Explicit selections took precedence over script defaults;
  conflicting selections reported an error. Custom visualizers used existing ``PresetCfg`` alternatives.
