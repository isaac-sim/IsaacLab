* Updated Newton to commit ``5ea82eef6f7911388025a8df6ccdb172e2fed286`` and MuJoCo/MuJoCo Warp to
  3.14, keeping workspace and wheel dependencies aligned.
* Updated cable segment reads and writes for Newton's explicit free root joint, preserving masked
  writes and CUDA graph capture.
* Fixed asset hard resets by removing redundant callbacks that accessed invalidated views and clearing
  Newton's old view registry and per-model step hooks before rebuilding. Data and actuator buffers were
  initialized once per model generation. As before, hard resets recreated asset data; callers must
  reacquire ``asset.data`` and its arrays after ``sim.reset(soft=False)``.
* Fixed contact-force debug visualization after Newton removed its deprecated sensor-transform alias.
