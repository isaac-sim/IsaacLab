Added
^^^^^

* Added support for ``./docker/container.py <profile> <command>`` in addition to the existing
  ``<command> <profile>`` order, so stepping a container through ``build``, ``start``, ``enter``,
  and ``stop`` only changes the last argument. Both orders are accepted.

Changed
^^^^^^^

* Changed ``./docker/container.py enter`` to start the container when it is not already running,
  instead of failing with ``The container '<name>' is not running.``. This also resolves the
  ``X11 forwarding is enabled but the temporary .xauth file does not exist`` error that followed a
  ``stop``, since the file is recreated as part of starting.
