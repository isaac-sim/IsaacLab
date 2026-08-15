Changed
^^^^^^^

* Changed :class:`~isaaclab_ov.renderers.OVRTXRenderer` to suppress the OVRTX deprecation warnings
  emitted for the legacy stage API. Isaac Lab still drives that API until the ovstage path becomes
  the default, so the warnings were noise no user of this renderer could act on. The option is set
  only when the installed OVRTX build exposes it, so older wheels are unaffected.
