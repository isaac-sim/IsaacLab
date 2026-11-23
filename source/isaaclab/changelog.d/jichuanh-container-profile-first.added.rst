* Added a ``-p``/``--profile`` option to ``./docker/container.py``, accepted before or after the
  command, so stepping a container through ``build``, ``start``, ``enter``, and ``stop`` only changes
  the last argument (``./docker/container.py -p kitless start``). The positional
  ``<command> <profile>`` form is unchanged.
