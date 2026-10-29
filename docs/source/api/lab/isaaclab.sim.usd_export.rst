isaaclab.sim.usd_export
=======================

.. automodule:: isaaclab.sim.usd_export

Exports one initialized environment as USD. The file keeps the scene's USD, with assets as references to
their source files, and adds the physics values of the active backend that an import of that USD would not
reproduce. Use :meth:`~isaaclab.scene.InteractiveScene.export_to_usd` or
:attr:`~isaaclab.scene.InteractiveSceneCfg.export_usd_path`. The file is meant for each backend's own import:
Isaac Sim and OVPhysX open it, and Newton reads it with :meth:`newton.ModelBuilder.add_usd`.

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
