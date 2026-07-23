Added
^^^^^

* Added :mod:`isaaclab.renderers.segmentation_colors`, a shared host + Warp implementation of the
  Replicator-compatible segmentation colorization (``random_color_from_id`` / ``color_hash`` /
  ``pack_rgba`` and the reserved ``BACKGROUND`` / ``UNLABELLED`` ids). It provides a single source of
  truth for the color palette used by the OVRTX and Newton Warp renderers so colorized segmentation
  outputs are visually consistent across backends.
