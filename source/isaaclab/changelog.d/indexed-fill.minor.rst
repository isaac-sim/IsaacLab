Added
^^^^^

* Added ``isaaclab.utils.index_fill_`` for in-place scalar fills with slices or integer indices,
  avoiding scalar uploads and CUDA synchronization when indices are already on the device.

Changed
^^^^^^^

* Used the shared fill operation in environment, manager, sensor, actuator, and buffer resets.
