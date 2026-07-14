Fixed
^^^^^

* Reduced direct locomotion step and reset overhead by staging actions once per environment step,
  avoiding duplicate articulation resets, redundant state copies, and separate joint-state writes.
* Reduced Ant Direct step overhead by fusing Newton post-processing and using device-mask resets.
