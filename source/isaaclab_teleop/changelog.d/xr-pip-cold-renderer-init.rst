Fixed
^^^^^

* Fixed isolated XR camera PiP startup by acquiring the global scene-partition setting at bind,
  after renderer initialization and before creating panels. Camera configuration remained prepared
  before environment construction, while the global setting was restored to its bind-time value
  after the last bound session closed.
