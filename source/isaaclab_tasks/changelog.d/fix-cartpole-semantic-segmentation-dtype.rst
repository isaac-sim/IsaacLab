Fixed
^^^^^

* Fixed the Cartpole camera tasks crashing at the first training step with
  ``RuntimeError: Input type (unsigned char) and bias type (float) should be the same`` when the
  ``semantic_segmentation`` preset was selected. Both the manager-based observation term and the
  direct environment normalized only RGB-like and depth output, so segmentation reached the feature
  extractor as an integer tensor. Segmentation is now routed through
  :func:`isaaclab.utils.images.normalize_camera_image`, which keys on the tensor dtype and therefore
  handles both colorized (``uint8`` RGBA) and non-colorized (``int32`` label ids) output.
