Fixed
^^^^^

* Kept asynchronous OVRTX renders owned through delivery failures and reset, and isolated each
  camera's output publication. Transform writes converted directly into retained input buffers.
