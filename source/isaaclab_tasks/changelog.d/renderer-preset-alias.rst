Changed
^^^^^^^

* Renamed the renderer presets ``ovrtx_renderer`` -> ``ovrtx`` and
  ``isaacsim_rtx_renderer`` -> ``isaacsim_rtx`` to drop the redundant
  ``_renderer`` suffix (the ``renderer=`` selector already names the category).
  The old ``ovrtx_renderer`` / ``isaacsim_rtx_renderer`` spellings still resolve
  as deprecated aliases and emit a :class:`FutureWarning`; migrate to the
  suffix-less names, which will become the only accepted form in a future
  release.

Deprecated
^^^^^^^^^^

* Deprecated the renderer preset names ``ovrtx_renderer`` and
  ``isaacsim_rtx_renderer`` in favor of ``ovrtx`` and ``isaacsim_rtx``.
