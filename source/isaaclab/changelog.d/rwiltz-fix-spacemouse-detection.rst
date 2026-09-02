Fixed
^^^^^

* Fixed :class:`~isaaclab.devices.Se2SpaceMouse` and :class:`~isaaclab.devices.Se3SpaceMouse` failing with
  ``No device found by SpaceMouse`` when a supported 3Dconnexion device was connected. Detection matched the
  HID product string exactly, which the ``libusb`` backend bundled in the ``hidapi`` wheels leaves empty
  unless the process may open the USB node. Directly attached devices are now matched by their USB vendor
  and product ids, with the product string kept as a fallback, and
  the errors raised when a device cannot be found or opened name the enumerated devices and the required
  permissions.
