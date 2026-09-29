Fixed
^^^^^

* Prevented repeated Ctrl+C presses from interrupting Kit shutdown, restoring the previous signal
  handler when shutdown returned.
