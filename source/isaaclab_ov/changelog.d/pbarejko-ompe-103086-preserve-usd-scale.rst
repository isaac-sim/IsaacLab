Fixed
^^^^^

* Fixed :class:`~isaaclab_ov.renderers.OVRTXRenderer` dropping authored USD scale when syncing
  Newton body transforms into OVRTX, which rendered scaled assets (for example Shadow Hand) at
  unit scale.
