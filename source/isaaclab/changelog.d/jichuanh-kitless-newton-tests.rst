Fixed
^^^^^

* Removed the unconditional ``from omni.physx.scripts import utils`` in
  :func:`isaaclab.sim.schemas.modify_articulation_root_properties` by inlining
  the single-selection ``Fixed`` joint creation via :mod:`pxr.UsdPhysics`
  directly. The previous code path broke any kitless newton run that needed
  to anchor a fixed-base articulation to the world.
