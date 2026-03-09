Curriculum Utilities
====================

.. currentmodule:: isaaclab.managers

This guide walks through the common curriculum helper functions and terms that can be used to create flexible curricula
for RL environments in Isaac Lab. These utilities can be passed to a :class:`~isaaclab.managers.CurriculumTermCfg`
object to enable dynamic modification of reward weights and environment parameters during training.

.. note::

   We cover three utilities in this guide:
   - The simple function modifies reward :func:`modify_reward_weight`
   - The term modify any environment parameters :class:`modify_env_param`
   - The term modify term_cfg :class:`modify_term_cfg`

.. dropdown:: Full source for curriculum utilities
   :icon: code

   .. literalinclude:: ../../../source/isaaclab/isaaclab/envs/mdp/curriculums.py
      :language: python


Modifying Reward Weights
------------------------

The function :func:`modify_reward_weight` updates the weight of a reward term after a specified number of simulation
steps. This can be passed directly as the ``func`` in a ``CurriculumTermCfg``.

.. literalinclude:: ../../../source/isaaclab/isaaclab/envs/mdp/curriculums.py
   :language: python
   :pyobject: modify_reward_weight

**Usage example**:

.. code-block:: python

   from isaaclab.managers import CurriculumTermCfg
   import isaaclab.managers.mdp as mdp

   # After 100k steps, set the "sparse_reward" term weight to 0.5
   sparse_reward_schedule = CurriculumTermCfg(
       func=mdp.modify_reward_weight,
       params={
           "term_name": "sparse_reward",
           "weight": 0.5,
           "num_steps": 100_000,
       }
   )


Dynamically Modifying Environment Parameters
--------------------------------------------

The class :class:`modify_env_param` is a :class:`~isaaclab.managers.ManagerTermBase` subclass that lets you target any
dotted attribute path in the environment and apply a user-supplied function to compute a new value at runtime. It
handles nested attributes, dictionary keys, list or tuple indexing, and respects a ``NO_CHANGE`` sentinel if no update
is desired.

.. literalinclude:: ../../../source/isaaclab/isaaclab/envs/mdp/curriculums.py
   :language: python
   :pyobject: modify_env_param

**Usage example**:

.. code-block:: python

   import torch
   from isaaclab.managers import CurriculumTermCfg
   import isaaclab.managers.mdp as mdp

   def resample_friction(env, env_ids, old_value, low, high, num_steps):
       # After num_steps, sample a new friction coefficient uniformly
       if env.common_step_counter > num_steps:
           return torch.empty((len(env_ids),), device="cpu").uniform_(low, high)
       return mdp.modify_env_param.NO_CHANGE

   friction_curriculum = CurriculumTermCfg(
       func=mdp.modify_env_param,
       params={
           "address": "event_manager.cfg.object_physics_material.func.material_buckets",
           "modify_fn": resample_friction,
           "modify_params": {
               "low": 0.3,
               "high": 1.0,
               "num_steps": 120_000,
           }
       }
   )


Modify Term Configuration
-------------------------

The subclass :class:`modify_term_cfg` provides a more concise style address syntax, using consistent with hydra config
syntax. It otherwise behaves identically to :class:`modify_env_param`.

.. literalinclude:: ../../../source/isaaclab/isaaclab/envs/mdp/curriculums.py
   :language: python
   :pyobject: modify_term_cfg

**Usage example**:

.. code-block:: python

   def override_command_range(env, env_ids, old_value, value, num_steps):
       # Override after num_steps
       if env.common_step_counter > num_steps:
           return value
       return mdp.modify_term_cfg.NO_CHANGE

   range_override = CurriculumTermCfg(
       func=mdp.modify_term_cfg,
       params={
           "address": "commands.object_pose.ranges.pos_x",
           "modify_fn": override_command_range,
           "modify_params": {
               "value": (-0.75, -0.25),
               "num_steps": 12_000,
           }
       }
   )
