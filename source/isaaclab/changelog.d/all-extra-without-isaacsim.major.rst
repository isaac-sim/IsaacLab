Changed
^^^^^^^

* **Breaking:** Changed the ``isaaclab[all]`` extra to exclude Isaac Sim. Install
  ``isaaclab[isaacsim]`` with the documented resolver overrides when Isaac Sim is required.
* **Breaking:** Moved the standalone URDF/MJCF importers from the base wheel to the
  ``isaaclab[importers]`` extra. Use the documented override command when installing it.

Fixed
^^^^^

* Fixed Newton actuator imports with the minimum Newton versions supported by the wheel.
