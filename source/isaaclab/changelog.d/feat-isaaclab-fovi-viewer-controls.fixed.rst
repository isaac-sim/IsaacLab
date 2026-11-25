* Fixed the ``foveated-camera`` example crashing while Kit started with ``renderer=isaacsim_rtx``: its top-level
  imports loaded USD before Kit, whose own USD then failed to load. It now imports USD once the simulation has
  launched.
