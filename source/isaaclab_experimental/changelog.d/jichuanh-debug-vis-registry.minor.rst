Changed
^^^^^^^

* Changed the Warp-frontend ``ActionTerm``, ``CommandTerm`` and ``DirectRLEnvWarp`` debug
  visualization toggles to register their callbacks through the simulation context's visualization
  marker registry instead of the deprecated Kit ``IApp.get_post_update_event_stream`` API, matching
  their non-experimental counterparts. Callbacks now run when a marker-capable visualizer
  dispatches them rather than on every Kit post-update tick.

Fixed
^^^^^

* Fixed debug visualization failing in kitless mode. Enabling it raised
  ``ModuleNotFoundError: No module named 'omni.kit'`` because ``omni.kit.app`` was imported inside
  the toggle. The registry path has no Kit dependency.
