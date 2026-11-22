isaaclab_newton.envs.mdp
========================

.. automodule:: isaaclab_newton.envs.mdp

Events
------

.. automodule:: isaaclab_newton.envs.mdp.events
    :members:

Terminations
------------

Configure ``solver_fault`` as a termination and ``zero_reward_on_solver_fault`` as the last nonzero
reward term. Both terms query the backend directly. Detection runs lazily once per state
timestamp and is refreshed after a physics step or completed FK/reset. The reward term clears complete
reward rows before summation and episode accumulation and does not require a named termination term.

.. code-block:: python

   from isaaclab.managers import RewardTermCfg, TerminationTermCfg
   from isaaclab_newton.envs.mdp import solver_fault, zero_reward_on_solver_fault
   from isaaclab.utils import configclass

   @configclass
   class TerminationsCfg:
       solver_fault = TerminationTermCfg(func=solver_fault)

   @configclass
   class RewardsCfg:
       # Declare task reward terms above this final term.
       solver_fault_reward = RewardTermCfg(func=zero_reward_on_solver_fault, weight=1.0)

To replace nonfinite observation values with zero, configure each observation term with
``tensor_processors=[TensorProcessorCfg(func=replace_non_finite)]``, importing both names from
:mod:`isaaclab.utils.tensor_processors`. This processes ordinary and terminal observations before
history storage. It leaves source state untouched so solver fault detection remains effective.

Detection includes Newton body, joint and particle state. MuJoCo additionally checks its private
persistent buffers. MJWarp checks and resets these buffers per world. CPU MuJoCo uses global solver
history, so a fault in that history terminates every environment.

.. automodule:: isaaclab_newton.envs.mdp.terminations
    :members:

Rewards
-------

.. automodule:: isaaclab_newton.envs.mdp.rewards
    :members:


Classes
-------

The following classes are part of the public :mod:`isaaclab_newton.envs.mdp` API.

.. currentmodule:: isaaclab_newton.envs.mdp

.. autosummary::
   :nosignatures:

   NewtonDifferentialInverseKinematicsAction
   NewtonDifferentialInverseKinematicsActionCfg
   NewtonInverseKinematicsAction
   NewtonInverseKinematicsActionCfg
   NewtonOperationalSpaceControllerAction
   NewtonOperationalSpaceControllerActionCfg

.. autoclass:: NewtonDifferentialInverseKinematicsAction
   :show-inheritance:

.. autoclass:: NewtonDifferentialInverseKinematicsActionCfg
   :show-inheritance:

.. autoclass:: NewtonInverseKinematicsAction
   :show-inheritance:

.. autoclass:: NewtonInverseKinematicsActionCfg
   :show-inheritance:

.. autoclass:: NewtonOperationalSpaceControllerAction
   :show-inheritance:

.. autoclass:: NewtonOperationalSpaceControllerActionCfg
   :show-inheritance:
