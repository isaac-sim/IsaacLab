Fixed
^^^^^

* Fixed :class:`~isaaclab.devices.Se3SpaceMouse` and
  :class:`~isaaclab.devices.Se2SpaceMouse` not detecting the 3Dconnexion
  SpaceNavigator. The device reports ``"SpaceNavigator for Notebooks"``, which
  was missing from the list of accepted product names, so it failed with
  ``No device found by SpaceMouse. Is the device connected?`` even though
  ``hid.enumerate()`` listed and opened it. It speaks the same HID report
  protocol as the already-supported devices, so only the name was missing.
