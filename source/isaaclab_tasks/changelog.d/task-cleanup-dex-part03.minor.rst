Added
^^^^^

* Added behavioral-success metrics and threshold-independent episode-error
  diagnostics to the dexterous reorientation environments.

Fixed
^^^^^

* Fixed dexterous hand resets that could initialize joints below their lower
  position limits. Reset joint positions now sample uniformly across the full
  joint range; previously the distribution was biased toward the lower half of
  the range.
