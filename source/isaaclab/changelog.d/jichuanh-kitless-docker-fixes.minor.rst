Added
^^^^^

* Added OVPhysX and the Viser and Rerun visualizers to the kit-less Docker image. ``physics=ovphysx``
  and ``--viz newton,viser,rerun`` now work in that image without any additional installation.

Fixed
^^^^^

* Fixed ``docker/container.py`` reporting success when a Docker command failed. A failing build,
  start, stop, or config printed ``Finished building...`` and exited ``0``; it now reports the failing
  command and exits non-zero.
* Fixed the kit-less Docker image missing ``libxrender1``, which made ``--viz newton`` fail with
  ``AttributeError: 'NoneType' object has no attribute 'XRenderFindVisualFormat'``.
