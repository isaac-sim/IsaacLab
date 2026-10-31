* Added an OVXR application adapter and session bootstrap that borrow the simulation-owned
  OVRTX renderer and OVStage, publish scene updates, and coordinate render-thread ownership
  and shutdown with Isaac Capture.
  Publication failures close admission, and failed lease retirement retains borrowed resources.
