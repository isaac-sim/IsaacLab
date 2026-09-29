isaaclab.actuators
==================

.. automodule:: isaaclab.actuators

  .. rubric:: Classes

  .. autosummary::

    ActuatorBase
    ActuatorBaseCfg
    ActuatorTargetCommand
    ActuatorCollection
    ActuatorControl
    ActuatorOutputCommand
    ImplicitActuator
    ImplicitActuatorCfg
    IdealPDActuator
    IdealPDActuatorCfg
    DCMotor
    DCMotorCfg
    DelayedPDActuator
    DelayedPDActuatorCfg
    RemotizedPDActuator
    RemotizedPDActuatorCfg
    ActuatorNetMLP
    ActuatorNetMLPCfg
    ActuatorNetLSTM
    ActuatorNetLSTMCfg

  .. rubric:: Functions

  .. autosummary::

    resolve_joint_parameter

Actuator Base
-------------

.. autoclass:: ActuatorBase
  :members:
  :inherited-members:

.. autofunction:: resolve_joint_parameter

.. autoclass:: ActuatorBaseCfg
  :members:
  :inherited-members:
  :exclude-members: __init__, class_type

Actuator Collection
-------------------

.. autoclass:: ActuatorCollection
  :members:
  :inherited-members:

.. autoclass:: ActuatorTargetCommand
  :members:

.. autoclass:: ActuatorOutputCommand
  :members:

Actuator Control
----------------

.. autoclass:: ActuatorControl
  :members:
  :inherited-members:

Implicit Actuator
-----------------

.. autoclass:: ImplicitActuator
  :members:
  :inherited-members:
  :show-inheritance:

.. autoclass:: ImplicitActuatorCfg
  :members:
  :inherited-members:
  :show-inheritance:
  :exclude-members: __init__, class_type

Ideal PD Actuator
-----------------

.. autoclass:: IdealPDActuator
  :members:
  :inherited-members:
  :show-inheritance:

.. autoclass:: IdealPDActuatorCfg
  :members:
  :inherited-members:
  :show-inheritance:
  :exclude-members: __init__, class_type

DC Motor Actuator
-----------------

.. autoclass:: DCMotor
  :members:
  :inherited-members:
  :show-inheritance:

.. autoclass:: DCMotorCfg
  :members:
  :inherited-members:
  :show-inheritance:
  :exclude-members: __init__, class_type

Delayed PD Actuator
-------------------

.. autoclass:: DelayedPDActuator
  :members:
  :inherited-members:
  :show-inheritance:

.. autoclass:: DelayedPDActuatorCfg
  :members:
  :inherited-members:
  :show-inheritance:
  :exclude-members: __init__, class_type

Remotized PD Actuator
---------------------

.. autoclass:: RemotizedPDActuator
  :members:
  :inherited-members:
  :show-inheritance:

.. autoclass:: RemotizedPDActuatorCfg
  :members:
  :inherited-members:
  :show-inheritance:
  :exclude-members: __init__, class_type

MLP Network Actuator
---------------------

.. autoclass:: ActuatorNetMLP
  :members:
  :inherited-members:
  :show-inheritance:

.. autoclass:: ActuatorNetMLPCfg
  :members:
  :inherited-members:
  :show-inheritance:
  :exclude-members: __init__, class_type

LSTM Network Actuator
---------------------

.. autoclass:: ActuatorNetLSTM
  :members:
  :inherited-members:
  :show-inheritance:

.. autoclass:: ActuatorNetLSTMCfg
  :members:
  :inherited-members:
  :show-inheritance:
  :exclude-members: __init__, class_type

Newton Actuator Access
----------------------

.. automodule:: isaaclab.actuators.newton

  .. rubric:: Functions

  .. autosummary::

    read_group_parameter
    write_group_parameter

.. autofunction:: isaaclab.actuators.newton.read_group_parameter

.. autofunction:: isaaclab.actuators.newton.write_group_parameter
