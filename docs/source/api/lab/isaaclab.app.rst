isaaclab.app
============

.. automodule:: isaaclab.app

   .. rubric:: Classes

   .. autosummary::

      AppLauncher
      LoadingScreen
      SimulationLauncher

   .. rubric:: Functions

   .. autosummary::

      add_launcher_args
      launch_simulation
      validate_launch
      report_activity


Simulation Launcher
-------------------

.. autofunction:: launch_simulation

.. autofunction:: validate_launch

.. autofunction:: add_launcher_args

.. autoclass:: SimulationLauncher
   :members:

.. autoclass:: AppLauncher
   :members:


Seasonal loading greetings
--------------------------

The loading screen includes bundled default greetings that work offline. Seasonal greetings are
selected by ``{ISAACLAB_NUCLEUS_DIR}/Greetings/manifest.json`` on the asset server. Greetings use
the same asset root and region profile as robot assets, including ``ISAACSIM_ASSET_ROOT`` and
``ISAACSIM_ASSET_REGION_PROFILE``. Each greeting pairs 24 by 12 and 74 by 12 terminal cell variants;
resizing keeps the selected artwork family. The loading screen displays the first frame of each variant.

Startup reuses validated cached greetings immediately and refreshes them in the background. The cache
lives under ``isaaclab/greetings/`` in ``XDG_CACHE_HOME``, then ``LOCALAPPDATA`` if the former is unset,
or ``~/.cache`` otherwise, with separate entries for each asset root. Unavailable or invalid remote
data leaves bundled defaults or usable cached greetings in place. Selecting the ``default`` collection
uses bundled artwork without downloading remote greeting assets.

Maintainers can change the calendar without an Isaac Lab release by editing ``Greetings/manifest.json``
on the asset server. Dates use UTC globally, include both endpoints, and are reevaluated at each startup,
including when using a cached catalog. Event IDs must be unique, collections must exist in the catalog,
and date windows must not overlap. Outside those windows, ``active_collection`` applies; keep it set to
``default`` for the bundled greetings. For example, the catalog's ``schedule`` object can contain:

.. code-block:: json

   {
     "timezone": "UTC",
     "events": [
       {
         "id": "christmas-2027",
         "collection": "christmas",
         "start": "2027-12-20",
         "end": "2027-12-27"
       }
     ]
   }


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
