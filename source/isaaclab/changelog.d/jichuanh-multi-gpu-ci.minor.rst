Added
^^^^^

* Added :func:`isaaclab.testing.cuda_test_devices` for env-driven device
  parametrize in unit tests. ``ISAACLAB_TEST_DEVICES=001`` selects
  ``[cuda:1]`` for the multi-GPU CI runner; the default mask ``110``
  resolves to ``[cpu, cuda:0]`` and is a no-op on the single-GPU CI.

* Added ``ISAACLAB_SIM_DEVICE`` env var honored by
  :class:`isaaclab.app.AppLauncher` as the implicit-default device when
  the caller doesn't pass ``device=``. Lets the multi-GPU CI workflow
  boot Kit with ``active_gpu=1`` without editing every test's
  :class:`~isaaclab.app.AppLauncher` call site.
