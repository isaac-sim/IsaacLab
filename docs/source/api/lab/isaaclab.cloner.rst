isaaclab.cloner
===============

.. automodule:: isaaclab.cloner

   .. Rubric:: Submodules

   .. autosummary::

      path
      query

   .. Rubric:: Classes

   .. autosummary::

      ClonePlan
      CloneCfg
      InclusionSet
      ReplicateSession
      UsdReplicateContext

   .. Rubric:: Functions

   .. autosummary::

      clone_plan_from_env_0
      make_clone_plan
      make_valid_clone_combinations
      num_spawn_variants
      grid_transforms
      replicate
      queue_replication
      usd_replicate
      filter_collisions

Clone plan
~~~~~~~~~~

.. automodule:: isaaclab.cloner.clone_plan
   :members:
   :show-inheritance:

Path
~~~~

.. automodule:: isaaclab.cloner.path
   :members:

Query
~~~~~

.. automodule:: isaaclab.cloner.query
   :members:

Additional Public Classes
-------------------------

The following classes are part of the public :mod:`isaaclab.cloner` API.

.. currentmodule:: isaaclab.cloner

.. autosummary::
   :nosignatures:

   CloneCfg
   InclusionSet
   ReplicateSession
   UsdReplicateContext

.. autoclass:: CloneCfg
   :show-inheritance:

.. autoclass:: InclusionSet
   :show-inheritance:

.. autoclass:: ReplicateSession
   :show-inheritance:

.. autoclass:: UsdReplicateContext
   :show-inheritance:
