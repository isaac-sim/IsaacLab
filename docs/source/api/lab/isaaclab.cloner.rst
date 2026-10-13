isaaclab.cloner
===============

.. automodule:: isaaclab.cloner

   .. Rubric:: Function namespaces

   .. autosummary::

      path
      query

   .. Rubric:: Classes

   .. autosummary::

      ClonePlan
      PrototypeWorldTopology
      CloneCfg
      InclusionSet
      ReplicateSession
      UsdReplicateContext

   .. Rubric:: Functions

   .. autosummary::

      clone_plan_from_env_0
      make_clone_plan
      to_warp
      make_valid_clone_combinations
      num_spawn_variants
      grid_transforms
      replicate
      usd_replicate
      filter_collisions

Clone plan
~~~~~~~~~~

.. currentmodule:: isaaclab.cloner

.. autoclass:: ClonePlan
   :members:

.. autoclass:: PrototypeWorldTopology
   :members:

.. autoclass:: isaaclab.cloner.clone_plan.TemplateMatch
   :members:

.. autofunction:: make_clone_plan

.. autofunction:: grid_transforms

.. autofunction:: to_warp

Path
~~~~

.. autoclass:: isaaclab.cloner.path
   :members:

Query
~~~~~

.. autoclass:: isaaclab.cloner.query
   :members:

Additional Public Classes
-------------------------

.. autoclass:: CloneCfg
   :show-inheritance:

.. autoclass:: InclusionSet
   :show-inheritance:

.. autoclass:: ReplicateSession
   :show-inheritance:

.. autoclass:: UsdReplicateContext
   :show-inheritance:
