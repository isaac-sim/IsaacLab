Changed
^^^^^^^

* Clarified ``--video`` behavior when multiple video-capable visualizers are active:
  Gymnasium video recording captures one ``env.render()`` stream, with Kit taking
  priority over Newton.
