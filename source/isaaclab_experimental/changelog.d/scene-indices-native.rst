Changed
^^^^^^^

* Built Warp selection arrays and masks from resolved Torch tensors without host readback.
  Converted selectors to Python lists only when exporting name metadata.
