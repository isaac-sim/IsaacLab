Added
^^^^^

* Separated manager-based episode completion from starting replacements through environment
  lifecycle hooks. Finite evaluators can now drain active episodes without starting replacements,
  including when the episode count is smaller than the environment batch. The default retained
  continuous automatic resets. Inactive environments continued physics and manager computations while
  excluding their completion signals, rewards, and trajectory records.
* Added optional environment IDs to recorder manager step callbacks, preserving the full-batch
  output contract of recorder terms. No migration is required for existing callers.
