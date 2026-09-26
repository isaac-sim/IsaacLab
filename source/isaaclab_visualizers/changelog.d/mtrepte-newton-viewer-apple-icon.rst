Fixed
^^^^^

* Fixed the Kit, ``newton_gl``, and ``newton_rtx`` visualizer windows showing a generic icon in
  Linux docks. Opening a visualizer window now writes a hidden desktop entry to
  ``$XDG_DATA_HOME/applications`` (default ``~/.local/share/applications``) that matches the
  window to its icon.
* Fixed the ``newton_rtx`` visualizer window not setting Newton's icon.
