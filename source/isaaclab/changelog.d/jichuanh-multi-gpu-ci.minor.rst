Added
^^^^^

* Added :func:`isaaclab.test.utils.test_devices` to parametrize unit tests over
  a device set resolved as ``scope ∩ budget``: ``scope`` is the call-site mask
  (the devices a test is valid on, e.g. ``"11X"`` for cpu + cuda:0 + any one
  non-default GPU), ``budget`` is the ``ISAACLAB_TEST_DEVICES`` env var (the
  devices a run may use, default ``"110"`` ⇒ cpu + cuda:0). A trailing ``X``
  means "any one non-default GPU", resolved to ``ISAACLAB_SIM_DEVICE`` when set.
  Single-GPU CI is unchanged; multi-GPU CI sets the budget to a non-default GPU.

* Added ``ISAACLAB_SIM_DEVICE`` env var honored by
  :class:`isaaclab.app.AppLauncher` as the implicit-default device when
  the caller doesn't pass ``device=``. Lets the multi-GPU CI workflow
  boot Kit on a non-default GPU without editing every test's
  :class:`~isaaclab.app.AppLauncher` call site.
