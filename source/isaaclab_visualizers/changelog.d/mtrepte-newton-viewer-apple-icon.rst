Fixed
^^^^^

* Fixed the ``newton_rtx`` visualizer window showing a generic application icon.
  ``ViewerRTX`` creates its window directly and never set an icon, unlike ``ViewerGL``
  (used by ``newton_gl``), whose ``RendererGL`` already sets Newton's own apple icon by
  default. ``newton_rtx`` now reuses that same bundled Newton icon for consistency.
