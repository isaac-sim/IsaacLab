Fixed
^^^^^

* Fixed :class:`~isaaclab_ov.renderers.OVRTXRenderer` authoring only one pixel render var when a
  camera requested several data types, which left every other requested output empty. The render
  product now authors one render var per requested data type, so combinations such as ``rgb`` with
  ``normals``, ``albedo``, ``motion_vectors``, segmentation, and depth are rendered together.
* Fixed :class:`~isaaclab_ov.renderers.OVRTXRenderer` filling ``depth``,
  ``distance_to_image_plane``, and ``distance_to_camera`` from a single depth render var, which
  returned euclidean distance for the image-plane outputs (or the reverse) when they were requested
  together. Each output is now extracted from the source that measures it.

Changed
^^^^^^^

* **Breaking:** Changed :class:`~isaaclab_ov.renderers.OVRTXRenderer` to raise :class:`ValueError` when a camera
  requests ``rgb`` or ``rgba`` together with a ``simple_shading_*`` data type, or more than one
  ``simple_shading_*`` data type. These outputs all read the ``LdrColor`` render var and simple
  shading additionally requires the render product to be in RTX Minimal mode, so one render product
  cannot serve them. Previously the conflict was resolved silently and produced wrongly shaded or
  empty images. Request the conflicting outputs from separate cameras.
