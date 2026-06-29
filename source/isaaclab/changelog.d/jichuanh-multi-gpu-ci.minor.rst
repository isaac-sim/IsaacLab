Added
^^^^^

* Added :class:`isaaclab.test.utils.DeviceScope` and
  :func:`isaaclab.test.utils.test_devices` to parametrize unit tests over a
  device set resolved as ``scope ∩ runtime``. Named scopes cover common cases,
  string masks support custom combinations, and multi-GPU CI narrows the runtime
  to one non-default GPU per shard without changing single-GPU CI behavior.
