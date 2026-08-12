Changed
^^^^^^^

* Changed :func:`~isaaclab.utils.images.normalize_camera_output_for_display` motion-vector
  visualization to clamp UV channels to ``[-1, 1]`` instead of scaling by peak magnitude.
  Absolute motion stays comparable across frames; values outside ``[-1, 1]`` are saturated.
