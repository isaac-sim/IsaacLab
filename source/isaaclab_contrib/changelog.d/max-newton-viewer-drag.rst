Fixed
^^^^^

* Fixed interactive Newton visualizer support for VBD by including startup
  force callbacks in its first CUDA graph capture. Late callback changes use
  eager execution because VBD graph re-capture is unavailable.
