Changed
^^^^^^^

* Changed the Isaac Sim pin to ``6.1.0rc3+release.45488.8127a152.gl``, served by the
  internal Isaac Sim PyPI index. ``uv sync`` and ``uv run`` pick it up from the declared
  index; pip-based installs must add ``--extra-index-url`` for that index, since the
  release candidate is not published on ``pypi.nvidia.com``.
* Changed the OV runtime pins to ``ovphysx==0.5.10``, ``ovrtx>=0.4.1,<0.5.0``, and
  ``ovstage==0.1.1.355824``, with ``ovphysx`` and ``ovrtx`` routed through the internal
  Omniverse PyPI index. Reinstall the ``ov``, ``ovphysx``, or ``ovrtx`` extra to pick up
  the new wheels.
