isaaclab.sim.usd_export
=======================

.. automodule:: isaaclab.sim.usd_export

Exports one initialized environment as USD. The file keeps the scene's USD, with assets as references to
their source files, and adds the physics values of the active backend that the USD does not already state.
Use :meth:`~isaaclab.scene.InteractiveScene.export_to_usd` or
:attr:`~isaaclab.scene.InteractiveSceneCfg.export_usd_path`; a Newton export loads back with
:meth:`isaaclab_newton.sim.usd_export.UsdExporter.load_usd`.

.. currentmodule:: isaaclab.sim.usd_export

.. autosummary::
   :nosignatures:

   UsdExporter
   BaseUsdExporter
   UsdEdit

Exporters
---------

.. autoclass:: UsdExporter
   :show-inheritance:

.. autoclass:: BaseUsdExporter
   :members:
   :show-inheritance:

.. autoclass:: UsdEdit
   :members:
