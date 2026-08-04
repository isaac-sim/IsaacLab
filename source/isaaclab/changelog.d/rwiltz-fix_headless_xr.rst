Added
^^^^^

* Added :attr:`~isaaclab.app.AppLauncher.headless` so scripts can query whether the app was
  launched without a Kit window, replacing reads of the ``--headless`` CLI flag removed in 3.0.

Fixed
^^^^^

* Fixed XR teleoperation aborting at startup with ``Exiting app because of dependency solver
  failure`` wherever the Kit extension registry was unreachable. The XR experience depended on
  the ``omni.kit.xr.bundle.generic`` meta-extension, which Isaac Sim does not ship, so it
  resolved only by downloading from the registry. It now depends on the XR extensions shipped
  on disk, and starts with no registry access.
* Fixed ``--xr`` running non-headless when the task configuration declared a Kit visualizer.
  Enabling XR without an explicit ``--visualizer kit`` now always runs headless and auto-starts
  the XR session, since there is no viewport in which to click **Start XR**.
* Fixed ``AttributeError: 'Namespace' object has no attribute 'headless'`` in the teleoperation
  and demo recording scripts, which still read the ``--headless`` flag removed in 3.0.
