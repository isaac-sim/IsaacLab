Fixed
^^^^^

* Fixed the Isaac RTX sensor pump refreshing rigid-body transforms only. Newton writes cable curve
  points and particle points into Fabric from the ``sync_*_to_usd`` calls that only ``pre_render()``
  makes, so a camera read that skipped them drew bodies at their current pose and cables frozen at
  their spawn pose — a stable, entirely plausible image of the wrong thing.
* Fixed :class:`~isaaclab_physx.renderers.IsaacRtxRenderer` launching its tiled-reshape kernel over
  an empty annotator buffer. The launch is dimensioned by the destination, so every thread read past
  the end of a zero-length source and took the process down with an illegal memory access. A frame
  with no data now reports a blank frame and a warning.
