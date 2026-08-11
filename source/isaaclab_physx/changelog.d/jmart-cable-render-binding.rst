Fixed
^^^^^

* Fixed the Isaac RTX sensor pump refreshing rigid-body transforms only. Newton writes cable curve
  points and particle points into Fabric from the ``sync_*_to_usd`` calls that only ``pre_render()``
  makes, so a camera read that skipped them drew bodies at their current pose and cables frozen at
  their spawn pose — a stable, entirely plausible image of the wrong thing.
