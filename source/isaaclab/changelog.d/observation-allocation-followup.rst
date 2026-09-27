Added
^^^^^

* Added the optional keyword-only ``out=None`` callable contract for manager-allocated observation outputs.
  RGB observations used this interface automatically to avoid copying normalized uint8 images,
  including stacked frames, while preserving independent observation snapshots.

Fixed
^^^^^

* Preserved observation snapshots and modifier state during post-processing, and removed redundant
  history allocations.
* Restored LEAPP action tracing while retaining in-place joint-action offset and clipping operations.
