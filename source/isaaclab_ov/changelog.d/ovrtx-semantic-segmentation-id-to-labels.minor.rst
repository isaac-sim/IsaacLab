Added
^^^^^

* Added semantic-segmentation label metadata to the OVRTX renderer. The
  ``SemanticIdMap`` render var is now decoded into
  ``camera.data.info["semantic_segmentation"]["idToLabels"]``, matching the
  Isaac RTX / Replicator contract (keys are semantic IDs or RGBA colors, values
  are ``{semantic_type: label}`` dicts, always including ``BACKGROUND`` and
  ``UNLABELLED``).
* Added :attr:`~isaaclab_ov.renderers.OVRTXRendererCfg.colorize_semantic_segmentation`
  to select between colorized RGBA (``uint8``) and raw ``int32`` semantic-ID
  output, at parity with the Isaac RTX renderer.

Fixed
^^^^^

* Fixed the OVRTX segmentation colorization hash to use 32-bit wraparound
  arithmetic (it previously widened to ``uint64``, changing the hashed bits).
  Colorized semantic and instance segmentation IDs now map to the same colors
  as ``omni.replicator`` / the Isaac RTX renderer.
