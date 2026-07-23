Added
^^^^^

* Added :func:`isaaclab.sim.enable_extension`, :func:`isaaclab.sim.disable_extension`, and
  :func:`isaaclab.sim.get_extension_path` for interacting with Kit extensions without Isaac Sim utility dependencies.

Changed
^^^^^^^

* Changed the stock Kit experiences to stop registering deprecated Isaac Sim extension aliases and deprecated extension
  search paths. Custom Kit experiences should depend on current ``isaacsim.*`` or ``omni.*`` extensions directly.

Fixed
^^^^^

* Fixed Pink IK controller initialization when the Isaac Sim experimental utilities extension was unavailable.
* Fixed mesh conversion initialization when the Isaac Sim experimental utilities extension was unavailable.
* Fixed MJCF conversion after removing the bundled Isaac Sim dependency extensions from application startup.
* Fixed GUI camera and XR applications missing the Replicator extension required by camera sensors.
* Fixed the GUI application failing to resolve its window icon path during startup.
* Fixed the documented wheel installation command missing the resolver overrides required by Isaac Sim.
* Fixed benchmark runs requesting OmniPerf output when using a Kit-less simulation preset.
* Fixed remaining scripts and deprecated XR utilities importing removed Isaac Sim extension helpers.
