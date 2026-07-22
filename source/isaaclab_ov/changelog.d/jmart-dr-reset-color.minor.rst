Added
^^^^^

* Added :class:`~isaaclab_ov.visual.OVRTXVisualColorWriter`, the kit-less OVRTX backend for
  :class:`~isaaclab.envs.mdp.randomize_visual_color`. It injects one OmniPBR material per target and
  rebinds each mesh through the renderer's ``write_attribute`` channel, flushing the path-traced
  accumulator after each write. This replaces a mesh's asset-bundled material with OmniPBR, so
  non-color PBR inputs reset to OmniPBR defaults.
