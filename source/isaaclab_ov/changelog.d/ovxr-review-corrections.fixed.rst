* Reused camera readback staging buffers while preserving stream ordering and retaining
  storage when its consumers cannot be drained.
* Rejected ordinary stage commits while leased and invalidated pairs after borrower publication failures.
* Validated camera devices before admission and retained positional configuration bindings.
