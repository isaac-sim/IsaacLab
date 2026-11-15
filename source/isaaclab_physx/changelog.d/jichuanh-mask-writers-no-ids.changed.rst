* Changed the :class:`~isaaclab_physx.assets.Articulation` fixed and spatial tendon mask setters to write the
  tendon buffers through the masks instead of converting them to indices, removing a host wait per call.
