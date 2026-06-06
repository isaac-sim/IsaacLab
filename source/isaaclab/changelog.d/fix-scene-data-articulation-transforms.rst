Added
^^^^^

* Added a scene-data backend hook for active ``InteractiveScene`` access so
  backends can source scene-owned entity transforms without relying on global
  rigid-body views, and visualizers can discover scene-owned contact sensors.
