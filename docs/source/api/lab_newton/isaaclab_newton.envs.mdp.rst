isaaclab_newton.envs.mdp
========================

.. automodule:: isaaclab_newton.envs.mdp

Events
------

.. automodule:: isaaclab_newton.envs.mdp.events
    :members:

Terminations
------------

Configure a physics fault as an invalid transition to reset the affected environments and zero their
reward contributions and terminal observations before they reach the learner.

.. code-block:: python

   from isaaclab.managers import TerminationTermCfg
   from isaaclab_newton.envs.mdp import solver_fault

   solver_fault_term = TerminationTermCfg(func=solver_fault, invalid_transition=True)

Detection includes Newton body, joint and particle state. MuJoCo additionally checks its private
persistent buffers. MJWarp checks and resets these buffers per world. CPU MuJoCo uses global solver
history, so a fault in that history terminates every environment.


.. automodule:: isaaclab_newton.envs.mdp.terminations
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
