* Fixed CLI runs on Linux aarch64 from a virtual environment exiting at ``isaacsim`` import with
  the ``LD_PRELOAD`` banner. The CLI now preloads the system ``libgomp.so.1`` by its full path,
  which is the form Isaac Sim's check accepts.
