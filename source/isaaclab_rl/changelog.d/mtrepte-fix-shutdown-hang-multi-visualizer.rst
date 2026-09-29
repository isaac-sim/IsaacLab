Fixed
^^^^^

* Fixed train/play entrypoints leaving environments open after setup failures or interrupts.
  Cleanup closed the final environment wrapper and ignored further Ctrl+C presses during teardown.
  Random/zero agents used the same cleanup and propagated interrupts instead of reporting success.
