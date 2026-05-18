Fixed
^^^^^

* Fixed a Windows fatal exception (``0xc0000139``) that crashed the process on startup when
  running any :class:`~isaaclab.envs.ManagerBasedRLEnv`-based script (e.g. ``skrl/play.py``)
  on machines where ``h5py``'s native DLL could not be loaded. The ``import h5py`` in
  :class:`~isaaclab.utils.datasets.HDF5DatasetFileHandler` was a top-level statement that
  executed unconditionally at import time. It is now deferred to the individual methods that
  open or create HDF5 files, so the DLL is only loaded when dataset recording is actually used.
