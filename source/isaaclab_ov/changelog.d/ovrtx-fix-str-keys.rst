Fixed
^^^^^

* Fixed the OVRTX renderer producing string keys (e.g. ``"0"``, ``"1"``, ``"2"``) instead of integer keys
  in the non-colorized ``idToLabels`` and ``idToSemantics`` mappings for ``semantic_segmentation`` and
  ``instance_segmentation_fast``. Index ``camera.data.info[...]["idToLabels"]`` and
  ``camera.data.info[...]["idToSemantics"]`` with integer pixel/semantic IDs when
  ``colorize_semantic_segmentation=False`` or ``colorize_instance_segmentation=False``.
