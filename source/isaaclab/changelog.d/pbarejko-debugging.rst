Fixed
^^^^^

* Fixed a startup crash in :class:`~isaaclab.app.AppLauncher` when launching with a CUDA device.
  Setting the current torch CUDA device used to happen before ``SimulationApp`` was created, which
  imported ``torch`` (and transitively NumPy/OpenBLAS) prior to Kit's platform-info fork. On systems
  where OpenBLAS's at-fork handlers were not yet safe, that fork could crash. The
  ``torch.cuda.set_device`` call is now deferred until after ``SimulationApp`` starts.
