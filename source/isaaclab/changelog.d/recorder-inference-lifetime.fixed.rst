* Kept retained recorder allocations writable after leaving PyTorch inference mode, including
  grown buffers and schema changes, so subsequent reset and export can finalize those episodes.
