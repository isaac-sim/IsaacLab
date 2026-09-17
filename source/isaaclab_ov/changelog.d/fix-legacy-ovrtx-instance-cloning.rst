Changed
^^^^^^^

* Changed OVRTX to use ovstage scene ownership by default, preserving USD instancing and avoiding
  legacy clone expansion costs. Set ``ISAAC_LAB_OVRTX_USE_OVSTAGE=0`` to retain the legacy path.

Fixed
^^^^^

* Fixed missing cloned visuals in the legacy OVRTX renderer by expanding nested USD instances
  in its exported clone sources. Simulation stages and the ovstage renderer path retained
  their original instancing.
