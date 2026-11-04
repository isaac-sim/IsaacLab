* Fixed :meth:`~isaaclab_ov.renderers.OVRTXRenderer.log_mesh` failing with a USD type mismatch when the mesh color
  was a NumPy row or a sequence of NumPy scalars rather than a three-float tuple.
