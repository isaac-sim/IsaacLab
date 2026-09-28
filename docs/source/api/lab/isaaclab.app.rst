isaaclab.app
============

.. automodule:: isaaclab.app

   .. rubric:: Classes

   .. autosummary::

      AppLauncher
      LoadingScreen
      Scan
      SimulationLauncher

   .. rubric:: Functions

   .. autosummary::

      add_launcher_args
      launch_simulation
      make_physics_cfg
      report_activity
      scan


Simulation Launcher
-------------------

.. autofunction:: launch_simulation

.. autofunction:: add_launcher_args

.. autoclass:: SimulationLauncher
   :members:

.. autoclass:: AppLauncher
   :members:

.. autofunction:: make_physics_cfg

.. autofunction:: scan

.. autoclass:: Scan
   :members:



Additional Public Classes
-------------------------

The following classes are part of the public :mod:`isaaclab.app` API.

.. currentmodule:: isaaclab.app

.. autosummary::
   :nosignatures:

   LoadingScreen
   SettingsManager

.. autoclass:: LoadingScreen
   :show-inheritance:

.. autoclass:: SettingsManager
   :show-inheritance:
