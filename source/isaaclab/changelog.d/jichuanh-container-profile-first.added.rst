* Added support for ``./docker/container.py <profile> <command>`` in addition to the existing
  ``<command> <profile>`` order, so stepping a container through ``build``, ``start``, ``enter``,
  and ``stop`` only changes the last argument. Both orders are accepted.
