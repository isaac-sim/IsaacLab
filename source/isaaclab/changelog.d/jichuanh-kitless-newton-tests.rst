Fixed
^^^^^

* Removed the unconditional ``from omni.physx.scripts import utils`` in
  :func:`isaaclab.sim.schemas.modify_articulation_root_properties` by inlining
  the single-selection ``Fixed`` joint creation via :mod:`pxr.UsdPhysics`
  directly. The previous code path broke any kitless newton run that needed
  to anchor a fixed-base articulation to the world.

Changed
^^^^^^^

* The ``test_articulation.py`` newton integration tests no longer boot Kit at
  module level: :func:`AppLauncher` is dropped and the existing kitless
  branch in :class:`~isaaclab.sim.SimulationContext` (already gated by
  :func:`~isaaclab.utils.version.has_kit`) carries the test. Avoids the Kit
  lifecycle ``SIGHUP`` / shutdown-hang under concurrent multi-GPU CI.
