* Fixed :class:`~isaaclab.utils.datasets.HDF5DatasetFileHandler` dataset files staying locked after closing when a
  child process was started while they were open, which made reopening them fail with ``BlockingIOError``.
