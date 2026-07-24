Fixed
^^^^^

* Fixed the Franka arms demo asset URL and OVRTX camera scene localization.

* Fixed the heterogeneous-scene demo to use the current launcher arguments and
  registered Digit task identifier.

Changed
^^^^^^^

* **Breaking:** Renamed the ``newton`` renderer selector in the PPISP camera
  demo to ``newton_renderer``. Update commands to pass
  ``--renderer newton_renderer``.
