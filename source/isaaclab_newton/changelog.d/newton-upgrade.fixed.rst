* Updated Newton to commit ``5ea82eef6f7911388025a8df6ccdb172e2fed286`` and MuJoCo/MuJoCo Warp to
  3.14, keeping workspace and wheel dependencies aligned.
* Updated cable segment reads and writes for Newton's explicit free root joint, preserving masked
  writes and CUDA graph capture.
* Fixed rigid-object hard resets by removing a redundant callback that accessed an invalidated view.
  Removed the same duplicate reset callback from cables and initialized their bindings and buffers once per data instance.
* Fixed contact-force debug visualization after Newton removed its deprecated sensor-transform alias.
