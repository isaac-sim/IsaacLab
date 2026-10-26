* Fixed :func:`~isaaclab.utils.configure_logging` leaving the root-logger handlers it replaces open. It now
  closes them, as :func:`logging.basicConfig` does with ``force=True``, so a log file opened by an earlier
  call is released and can be deleted on Windows.
