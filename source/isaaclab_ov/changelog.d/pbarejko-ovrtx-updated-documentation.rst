Fixed
^^^^^

* Fixed the OVRTX renderer to raise a clear, actionable error when the optional
  ``ovrtx`` runtime wheel is not installed, pointing users to
  ``./isaaclab.sh -i 'ov[ovrtx]'`` instead of a cryptic ``No module named 'ovrtx'``.
