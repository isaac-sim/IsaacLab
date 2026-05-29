Fixed
^^^^^

* Fixed module-import-time ``from pxr import …`` in :mod:`isaaclab.sim.simulation_context`,
  :mod:`isaaclab.assets.asset_base`, :mod:`isaaclab.scene.interactive_scene`,
  :mod:`isaaclab.sim.spawners.from_files.from_files`, :mod:`isaaclab.sim.utils.prims`,
  :mod:`isaaclab.sim.utils.queries`, :mod:`isaaclab.sim.utils.semantics`,
  :mod:`isaaclab.sim.utils.stage`, and :mod:`isaaclab.sim.utils.transforms`.  Previously,
  ``from isaaclab.assets import Articulation`` or ``from isaaclab.sim import SimulationContext``
  forced ``pxr`` (USD) into ``sys.modules`` before :class:`~isaaclab.app.AppLauncher`,
  which broke Kit's USD binding registration with a cascade of
  ``TfNotice`` / ``UsdAPISchemaBase`` / ``GfVec3f`` converter errors during
  ``SimulationApp.startup``.  Kit-less env-cfg parsing followed by ``--visualizer kit``
  now succeeds without any pxr modules preloaded.  Each affected module keeps its
  type hints under ``TYPE_CHECKING`` and defers the runtime ``from pxr import …`` into
  the function bodies that use it.
