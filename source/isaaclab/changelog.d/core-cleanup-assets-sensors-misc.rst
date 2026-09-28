Fixed
^^^^^

* Fixed :class:`~isaaclab.sensors.SensorBase` raising ``AttributeError`` on teardown when ``__init__`` failed
  before its simulation callbacks were registered.
* Fixed :class:`~isaaclab.devices.Se2SpaceMouse` raising on deletion when its listener thread was never started.
* Changed :class:`~isaaclab.devices.HaplyDevice` to report WebSocket receive errors through the module logger instead of printing them.
