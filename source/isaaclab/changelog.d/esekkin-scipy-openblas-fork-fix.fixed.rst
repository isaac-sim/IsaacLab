* Fixed crashes and hangs in ``fork()`` when several threads fork at once by raising the ``scipy``
  dependency floor to 1.18.0 and overriding the ``scipy==1.17.0`` pin from Isaac Sim 6.2. The OpenBLAS
  bundled with SciPy 1.17.0 could run its pre-fork thread-pool shutdown concurrently and join freed threads.
