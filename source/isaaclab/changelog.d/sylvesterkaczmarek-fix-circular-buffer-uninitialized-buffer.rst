Fixed
^^^^^

* Fixed ``CircularBuffer.buffer`` raising an implementation-level PyTorch error when accessed before the first
  ``append()`` by reporting a clear ``RuntimeError`` instead.
