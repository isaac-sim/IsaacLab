Fixed
^^^^^

* Fixed static child prims (visual and collision meshes) being drawn at the world origin
  after a stage is torn down and rebuilt in the same process. The rebuilt subtree was not
  flagged dirty, so the Fabric hierarchy never recomposed its world transforms.
