Added
^^^^^

* Added the packaged ``isaaclab demo <name>`` command so released demos can be listed and launched with
  ``uvx isaaclab demo <name>``.

Changed
^^^^^^^

* **Breaking:** Moved maintained demos from ``scripts/demos`` into the repository-level ``demos`` directory and
  updated the demo smoke suite to exercise the installed command. Use ``isaaclab demo list`` to discover names and
  replace direct script invocations with ``isaaclab demo <name>``.

Removed
^^^^^^^

* **Breaking:** Removed the temporary ``ppisp_camera_ovrtx.py`` QA demo. Use ``isaaclab demo ppisp-camera`` and
  select its ``newton_renderer`` renderer instead.
