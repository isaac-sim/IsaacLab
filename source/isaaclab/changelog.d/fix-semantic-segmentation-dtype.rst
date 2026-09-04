Fixed
^^^^^

* Fixed :func:`isaaclab.utils.images.normalize_camera_image` returning non-colorized
  ``"semantic_segmentation"`` unchanged. Such output is an ``int32`` label map on every renderer,
  and feeding it to a convolution raised ``Input type (int) and bias type (float) should be the
  same``. It is now cast to ``float32``; label ids carry no scale, so they are not rescaled.
  Colorized (``uint8`` RGBA) segmentation keeps its existing ``(x / 255) - mean`` normalization.
