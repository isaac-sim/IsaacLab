* Fixed ``--- Logging error ---`` / ``I/O operation on closed file`` tracebacks on Windows and macOS when a
  logging handler created during the loading screen wrote after startup finished. The loading screen now
  points such handlers back at the real ``sys.stdout`` / ``sys.stderr`` when it restores the console.
