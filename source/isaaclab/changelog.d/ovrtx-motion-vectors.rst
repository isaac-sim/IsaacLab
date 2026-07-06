Changed
^^^^^^^

* Changed :func:`~isaaclab.utils.images.normalize_camera_output_for_display` to map the
  ``motion_vectors`` data type, normalizing the per-pixel ``(u, v)`` offsets by their peak
  magnitude and packing them into an RGB image (``u`` -> red, ``v`` -> green) instead of
  scaling by ``255`` and returning a two-channel tensor. Callers that consumed the previous
  two-channel output should update to expect a three-channel ``[0, 1]`` image.
