Fixed
^^^^^

* Fixed deprecation warnings emitted during OVRTX rendering by replacing uses
  of the deprecated ``MappedRenderVar.tensor`` accessor with direct DLPack
  reads in :class:`~isaaclab_ov.renderers.OVRTXRenderer`.
