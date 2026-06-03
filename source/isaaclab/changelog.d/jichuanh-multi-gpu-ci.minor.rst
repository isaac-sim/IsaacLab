Added
^^^^^

* Added :func:`isaaclab.test.utils.test_devices` to parametrize unit tests over
  a device set resolved as ``scope ∩ runtime``: ``scope`` is the call-site mask
  of devices a test is valid on (default ``"11X"`` ⇒ cpu + cuda:0 + the
  non-default GPUs), and the runtime is the ``ISAACLAB_TEST_DEVICES`` env var of
  devices a run may use (default ``"110"`` ⇒ cpu + cuda:0). A trailing ``X``
  includes the remaining devices. Single-GPU CI is unchanged; multi-GPU CI sets
  the runtime to one non-default GPU per shard.

* Added ``ISAACLAB_SIM_DEVICE`` env var honored by
  :class:`isaaclab.app.AppLauncher` as the implicit-default device when
  the caller doesn't pass ``device=``. Lets the multi-GPU CI workflow
  boot Kit on a non-default GPU without editing every test's
  :class:`~isaaclab.app.AppLauncher` call site.

* Added ``py-spy`` + ``gdb`` stack capture in ``tools/conftest.py`` on
  ``shutdown_hang`` / ``startup_hang`` / ``timeout`` detection. Walks the test
  subprocess's process group (cap 8 pids), captures both Python and C++ frames
  before ``SIGKILL`` erases them, attaches the output to the JUnit error
  report. Makes Kit binary hangs observable in CI logs; safe no-op when
  ``py-spy``/``gdb`` are missing. Workflow side adds ``--cap-add=SYS_PTRACE``
  on the per-shard ``docker run`` (required to attach) and adds ``py-spy`` to
  the in-container ``pip install`` list.
