Fixed
^^^^^

* Fixed first runs through the CLI failing out of the box on Linux aarch64 (e.g. DGX Spark):
  python subprocesses now preload the system ``libgomp`` automatically instead of exiting with
  a warning banner that asks the user to export ``LD_PRELOAD`` by hand. Also documented
  ``OMNI_KIT_ACCEPT_EULA=yes`` for non-interactive first launches and the aarch64 build path
  in the source installation guide.
