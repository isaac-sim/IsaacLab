Added
^^^^^

* Added :class:`isaaclab.test.utils.DeviceScope`,
  :func:`isaaclab.test.utils.test_devices`, and
  :func:`isaaclab.test.utils.resolve_test_sim_device` to parametrize unit tests
  and launch Kit from the same runtime device mask. Composable scopes cover
  common cases, string masks support custom combinations, and multi-GPU CI
  narrows the runtime to one non-default GPU per shard without changing
  single-GPU CI behavior.
