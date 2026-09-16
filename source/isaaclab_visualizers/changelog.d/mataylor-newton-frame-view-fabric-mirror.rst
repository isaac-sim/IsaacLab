Fixed
^^^^^

* Fixed black or misplaced generated Kit streaming-camera images with Newton physics by removing
  the redundant USD pose writes that reset the camera transform stack after its Fabric pose was updated.
  Centered the cartpole golden-test reset pose to keep the tilted poles inside the camera frame.
