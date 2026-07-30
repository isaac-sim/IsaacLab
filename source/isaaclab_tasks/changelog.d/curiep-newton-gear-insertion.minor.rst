Added
^^^^^

* Added Newton point-SDF and hydroelastic-SDF physics presets and collision
  assets to the Flexiv Rizon 4s gear assembly task.
* Added a Newton inverse-kinematics task variant for task-space policies.

Fixed
^^^^^

* Fixed Newton gear reset, reward, and termination frames so selected gears
  target their shafts and physical grasps are measured at the fingertip
  midpoint without changing the PhysX defaults.
