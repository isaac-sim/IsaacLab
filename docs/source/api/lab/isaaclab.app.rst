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
selected by ``greetings/manifest.json`` in the
`nvidia/isaaclab-docs-assets dataset <https://huggingface.co/datasets/nvidia/isaaclab-docs-assets/tree/main/greetings>`__
on Hugging Face.
Each greeting pairs 24 by 12 and 74 by 12 terminal cell variants;
resizing keeps the selected artwork family. Each variant contains one static ANSI ``frame`` string,
with ``cols`` and ``rows`` specifying its dimensions.

Startup reuses validated cached greetings immediately and refreshes them in the background. The cache
lives under ``isaaclab/greetings/`` in ``XDG_CACHE_HOME``, then ``LOCALAPPDATA`` if the former is unset,
or ``~/.cache`` otherwise, with separate entries for each greeting source URL. If no cache directory can
be resolved, remote greetings are disabled and bundled artwork is used. Unavailable or invalid remote
data leaves bundled defaults or usable cached greetings in place. Selecting the ``default`` collection
uses bundled artwork without downloading remote greeting assets.

Set ``ISAACLAB_REMOTE_GREETINGS=0`` to use bundled artwork without remote requests or greeting cache
access.

Maintainers can change the calendar without an Isaac Lab release by editing ``greetings/manifest.json``
in the dataset's ``main`` branch. Dates use UTC globally, include both endpoints, and are reevaluated at
each startup, including when using a cached catalog. Event IDs must be unique, collections must exist
in the catalog, and date windows must not overlap. Outside those windows, ``active_collection`` applies;
keep it set to ``default`` for the bundled greetings. For example, the catalog's ``schedule`` object can contain:

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
