Fixed
^^^^^

* Fixed a ``SIGSEGV`` crash during Kit startup caused by NumPy's bundled
  OpenBLAS ``pthread_atfork`` handler.  When ``import torch`` (or any
  transitive NumPy import) runs before :class:`AppLauncher` creates the
  :class:`~isaacsim.SimulationApp`, OpenBLAS spawns worker threads and
  registers ``blas_thread_shutdown_`` as a child-side ``atfork`` handler.
  Kit's ``libomni.platforminfo.plugin`` then calls ``fork()`` during
  startup; in the child process the handler tries to ``pthread_join``
  threads that no longer exist, causing a segmentation fault.  The fix
  sets ``OPENBLAS_NUM_THREADS=1`` (via ``setdefault``) before the library
  is loaded so that no worker threads are created and the handler is a
  safe no-op.  Both :mod:`app_launcher` (for standalone scripts) and
  ``tools/conftest.py`` (for CI test subprocesses) are patched.
