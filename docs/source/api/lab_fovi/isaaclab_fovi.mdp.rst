isaaclab_fovi.mdp
=================

.. automodule:: isaaclab_fovi.mdp

  .. rubric:: Classes

  .. autosummary::

    GazeAction
    GazeActionCfg
    gaze_teacher_labels

  .. rubric:: Functions

  .. autosummary::

    gaze_state
    motor_action_l2

Actions
-------

.. autoclass:: GazeAction
    :members:
    :show-inheritance:

.. autoclass:: GazeActionCfg
    :members:
    :inherited-members:
    :show-inheritance:
    :exclude-members: __init__, class_type

Observations
------------

.. autofunction:: gaze_state

.. autoclass:: gaze_teacher_labels
    :show-inheritance:

Rewards
-------

.. autofunction:: motor_action_l2
