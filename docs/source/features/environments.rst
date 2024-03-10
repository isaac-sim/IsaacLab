Environments
============

The following lists comprises of all the RL tasks implementations that are available in Orbit.
While we try to keep this list up-to-date, you can always get the latest list of environments by
running the following command:

.. code-block:: bash

    ./orbit.sh -p source/standalone/environments/list_envs.py

We are actively working on adding more environments to the list. If you have any environments that
you would like to add to Orbit, please feel free to open a pull request!

Classic
-------

Classic environments that are based on IsaacGymEnvs implementation of MuJoCo-style environments.


.. table::
    :widths: 33 37 30

    +------------------+-----------------------------+-------------------------------------------------------------------------+
    | World            | Environment ID              | Description                                                             |
    +==================+=============================+=========================================================================+
    | |humanoid|       | |humanoid-link|             | Move towards a direction with the MuJoCo humanoid robot                 |
    +------------------+-----------------------------+-------------------------------------------------------------------------+
    | |ant|            | |ant-link|                  | Move towards a direction with the MuJoCo ant robot                      |
    +------------------+-----------------------------+-------------------------------------------------------------------------+
    | |cartpole|       | |cartpole-link|             | Move the cart to keep the pole upwards in the classic cartpole control  |
    +------------------+-----------------------------+-------------------------------------------------------------------------+

.. |humanoid| image:: ../_static/tasks/classic/humanoid.jpg
.. |ant| image:: ../_static/tasks/classic/ant.jpg
.. |cartpole| image:: ../_static/tasks/classic/cartpole.jpg

.. |humanoid-link| replace:: `Isaac-Humanoid-v0 <https://github.com/NVIDIA-Omniverse/orbit/blob/main/source/extensions/omni.isaac.orbit_tasks/omni/isaac/orbit_tasks/classic/humanoid/humanoid_env_cfg.py>`__
.. |ant-link| replace:: `Isaac-Ant-v0 <https://github.com/NVIDIA-Omniverse/orbit/blob/main/source/extensions/omni.isaac.orbit_tasks/omni/isaac/orbit_tasks/classic/ant/ant_env_cfg.py>`__
.. |cartpole-link| replace:: `Isaac-Cartpole-v0 <https://github.com/NVIDIA-Omniverse/orbit/blob/main/source/extensions/omni.isaac.orbit_tasks/omni/isaac/orbit_tasks/classic/cartpole/cartpole_env_cfg.py>`__


Manipulation
------------

Environments based on fixed-arm manipulation tasks.

For many of these tasks, we include configurations with different arm action spaces. For example,
for the reach environment:

* |lift-cube-link|: Franka arm with joint position control
* |lift-cube-ik-abs-link|: Franka arm with absolute IK control
* |lift-cube-ik-rel-link|: Franka arm with relative IK control

.. table::
    :widths: 33 37 30

    +----------------+---------------------+-----------------------------------------------------------------------------+
    | World          | Environment ID      | Description                                                                 |
    +================+=====================+=============================================================================+
    | |reach-franka| | |reach-franka-link| | Move the end-effector to a sampled target pose with the Franka robot        |
    +----------------+---------------------+-----------------------------------------------------------------------------+
    | |reach-ur10|   | |reach-ur10-link|   | Move the end-effector to a sampled target pose with the UR10 robot          |
    +----------------+---------------------+-----------------------------------------------------------------------------+
    | |lift-cube|    | |lift-cube-link|    | Pick a cube and bring it to a sampled target position with the Franka robot |
    +----------------+---------------------+-----------------------------------------------------------------------------+
    | |cabi-franka|  | |cabi-franka-link|  | Grasp the handle of a cabinet's drawer and open it with the Franka robot    |
    +----------------+---------------------+-----------------------------------------------------------------------------+

.. |reach-franka| image:: ../_static/tasks/manipulation/franka_reach.jpg
.. |reach-ur10| image:: ../_static/tasks/manipulation/ur10_reach.jpg
.. |lift-cube| image:: ../_static/tasks/manipulation/franka_lift.jpg
.. |cabi-franka| image:: ../_static/tasks/manipulation/franka_open_drawer.jpg

.. |reach-franka-link| replace:: `Isaac-Reach-Franka-v0 <https://github.com/NVIDIA-Omniverse/orbit/blob/main/source/extensions/omni.isaac.orbit_tasks/omni/isaac/orbit_tasks/manipulation/reach/config/franka/joint_pos_env_cfg.py>`__
.. |reach-ur10-link| replace:: `Isaac-Reach-UR10-v0 <https://github.com/NVIDIA-Omniverse/orbit/blob/main/source/extensions/omni.isaac.orbit_tasks/omni/isaac/orbit_tasks/manipulation/reach/config/ur_10/joint_pos_env_cfg.py>`__
.. |lift-cube-link| replace:: `Isaac-Lift-Cube-Franka-v0 <https://github.com/NVIDIA-Omniverse/orbit/blob/main/source/extensions/omni.isaac.orbit_tasks/omni/isaac/orbit_tasks/manipulation/lift/config/franka/joint_pos_env_cfg.py>`__
.. |lift-cube-ik-abs-link| replace:: `Isaac-Lift-Cube-Franka-IK-Abs-v0 <https://github.com/NVIDIA-Omniverse/orbit/blob/main/source/extensions/omni.isaac.orbit_tasks/omni/isaac/orbit_tasks/manipulation/lift/config/franka/ik_abs_env_cfg.py>`__
.. |lift-cube-ik-rel-link| replace:: `Isaac-Lift-Cube-Franka-IK-Rel-v0 <https://github.com/NVIDIA-Omniverse/orbit/blob/main/source/extensions/omni.isaac.orbit_tasks/omni/isaac/orbit_tasks/manipulation/lift/config/franka/ik_rel_env_cfg.py>`__
.. |cabi-franka-link| replace:: `Isaac-Open-Drawer-Franka-v0 <https://github.com/NVIDIA-Omniverse/orbit/blob/main/source/extensions/omni.isaac.orbit_tasks/omni/isaac/orbit_tasks/manipulation/cabinet/config/franka/joint_pos_env_cfg.py>`__


Locomotion
----------

Environments based on legged locomotion tasks.

.. table::
    :widths: 33 37 30

    +------------------------------+----------------------------------------------+-------------------------------------------------------------------------+
    | World                        | Environment ID                               | Description                                                             |
    +==============================+==============================================+=========================================================================+
    | |velocity-flat-anymal-b|     | |velocity-flat-anymal-b-link|                | Track a velocity command on flat terrain with the Anymal B robot        |
    +------------------------------+----------------------------------------------+-------------------------------------------------------------------------+
    | |velocity-rough-anymal-b|    | |velocity-rough-anymal-b-link|               | Track a velocity command on rough terrain with the Anymal B robot       |
    +------------------------------+----------------------------------------------+-------------------------------------------------------------------------+
    | |velocity-flat-anymal-c|     | |velocity-flat-anymal-c-link|                | Track a velocity command on flat terrain with the Anymal C robot        |
    +------------------------------+----------------------------------------------+-------------------------------------------------------------------------+
    | |velocity-rough-anymal-c|    | |velocity-rough-anymal-c-link|               | Track a velocity command on rough terrain with the Anymal C robot       |
    +------------------------------+----------------------------------------------+-------------------------------------------------------------------------+
    | |velocity-flat-anymal-d|     | |velocity-flat-anymal-d-link|                | Track a velocity command on flat terrain with the Anymal D robot        |
    +------------------------------+----------------------------------------------+-------------------------------------------------------------------------+
    | |velocity-rough-anymal-d|    | |velocity-rough-anymal-d-link|               | Track a velocity command on rough terrain with the Anymal D robot       |
    +------------------------------+----------------------------------------------+-------------------------------------------------------------------------+
    | |velocity-flat-unitree-a1|   | |velocity-flat-unitree-a1-link|              | Track a velocity command on flat terrain with the Unitree A1 robot      |
    +------------------------------+----------------------------------------------+-------------------------------------------------------------------------+
    | |velocity-rough-unitree-a1|  | |velocity-rough-unitree-a1-link|             | Track a velocity command on rough terrain with the Unitree A1 robot     |
    +------------------------------+----------------------------------------------+-------------------------------------------------------------------------+
    | |velocity-flat-unitree-go1|  | |velocity-flat-unitree-go1-link|             | Track a velocity command on flat terrain with the Unitree Go1 robot     |
    +------------------------------+----------------------------------------------+-------------------------------------------------------------------------+
    | |velocity-rough-unitree-go1| | |velocity-rough-unitree-go1-link|            | Track a velocity command on rough terrain with the Unitree Go1 robot    |
    +------------------------------+----------------------------------------------+-------------------------------------------------------------------------+
    | |velocity-flat-unitree-go2|  | |velocity-flat-unitree-go2-link|             | Track a velocity command on flat terrain with the Unitree Go2 robot     |
    +------------------------------+----------------------------------------------+-------------------------------------------------------------------------+
    | |velocity-rough-unitree-go2| | |velocity-rough-unitree-go2-link|            | Track a velocity command on rough terrain with the Unitree Go2 robot    |
    +------------------------------+----------------------------------------------+-------------------------------------------------------------------------+

.. |velocity-flat-anymal-b-link| replace:: `Isaac-Velocity-Flat-Anymal-B-v0 <https://github.com/NVIDIA-Omniverse/orbit/blob/main/source/extensions/omni.isaac.orbit_tasks/omni/isaac/orbit_tasks/locomotion/velocity/config/anymal_b/flat_env_cfg.py>`__
.. |velocity-rough-anymal-b-link| replace:: `Isaac-Velocity-Rough-Anymal-B-v0 <https://github.com/NVIDIA-Omniverse/orbit/blob/main/source/extensions/omni.isaac.orbit_tasks/omni/isaac/orbit_tasks/locomotion/velocity/config/anymal_b/rough_env_cfg.py>`__

.. |velocity-flat-anymal-c-link| replace:: `Isaac-Velocity-Flat-Anymal-C-v0 <https://github.com/NVIDIA-Omniverse/orbit/blob/main/source/extensions/omni.isaac.orbit_tasks/omni/isaac/orbit_tasks/locomotion/velocity/config/anymal_c/flat_env_cfg.py>`__
.. |velocity-rough-anymal-c-link| replace:: `Isaac-Velocity-Rough-Anymal-C-v0 <https://github.com/NVIDIA-Omniverse/orbit/blob/main/source/extensions/omni.isaac.orbit_tasks/omni/isaac/orbit_tasks/locomotion/velocity/config/anymal_c/rough_env_cfg.py>`__

.. |velocity-flat-anymal-d-link| replace:: `Isaac-Velocity-Flat-Anymal-D-v0 <https://github.com/NVIDIA-Omniverse/orbit/blob/main/source/extensions/omni.isaac.orbit_tasks/omni/isaac/orbit_tasks/locomotion/velocity/config/anymal_d/flat_env_cfg.py>`__
.. |velocity-rough-anymal-d-link| replace:: `Isaac-Velocity-Rough-Anymal-D-v0 <https://github.com/NVIDIA-Omniverse/orbit/blob/main/source/extensions/omni.isaac.orbit_tasks/omni/isaac/orbit_tasks/locomotion/velocity/config/anymal_d/rough_env_cfg.py>`__

.. |velocity-flat-unitree-a1-link| replace:: `Isaac-Velocity-Flat-Unitree-A1-v0 <https://github.com/NVIDIA-Omniverse/orbit/blob/main/source/extensions/omni.isaac.orbit_tasks/omni/isaac/orbit_tasks/locomotion/velocity/config/unitree_a1/flat_env_cfg.py>`__
.. |velocity-rough-unitree-a1-link| replace:: `Isaac-Velocity-Rough-Unitree-A1-v0 <https://github.com/NVIDIA-Omniverse/orbit/blob/main/source/extensions/omni.isaac.orbit_tasks/omni/isaac/orbit_tasks/locomotion/velocity/config/unitree_a1/rough_env_cfg.py>`__

.. |velocity-flat-unitree-go1-link| replace:: `Isaac-Velocity-Flat-Unitree-Go1-v0 <https://github.com/NVIDIA-Omniverse/orbit/blob/main/source/extensions/omni.isaac.orbit_tasks/omni/isaac/orbit_tasks/locomotion/velocity/config/unitree_go1/flat_env_cfg.py>`__
.. |velocity-rough-unitree-go1-link| replace:: `Isaac-Velocity-Rough-Unitree-Go1-v0 <https://github.com/NVIDIA-Omniverse/orbit/blob/main/source/extensions/omni.isaac.orbit_tasks/omni/isaac/orbit_tasks/locomotion/velocity/config/unitree_go1/rough_env_cfg.py>`__

.. |velocity-flat-unitree-go2-link| replace:: `Isaac-Velocity-Flat-Unitree-Go2-v0 <https://github.com/NVIDIA-Omniverse/orbit/blob/main/source/extensions/omni.isaac.orbit_tasks/omni/isaac/orbit_tasks/locomotion/velocity/config/unitree_go2/flat_env_cfg.py>`__
.. |velocity-rough-unitree-go2-link| replace:: `Isaac-Velocity-Rough-Unitree-Go2-v0 <https://github.com/NVIDIA-Omniverse/orbit/blob/main/source/extensions/omni.isaac.orbit_tasks/omni/isaac/orbit_tasks/locomotion/velocity/config/unitree_go2/rough_env_cfg.py>`__


.. |velocity-flat-anymal-b| image:: ../_static/tasks/locomotion/anymal_b_flat.jpg
.. |velocity-rough-anymal-b| image:: ../_static/tasks/locomotion/anymal_b_rough.jpg
.. |velocity-flat-anymal-c| image:: ../_static/tasks/locomotion/anymal_c_flat.jpg
.. |velocity-rough-anymal-c| image:: ../_static/tasks/locomotion/anymal_c_rough.jpg
.. |velocity-flat-anymal-d| image:: ../_static/tasks/locomotion/anymal_d_flat.jpg
.. |velocity-rough-anymal-d| image:: ../_static/tasks/locomotion/anymal_d_rough.jpg
.. |velocity-flat-unitree-a1| image:: ../_static/tasks/locomotion/a1_flat.jpg
.. |velocity-rough-unitree-a1| image:: ../_static/tasks/locomotion/a1_rough.jpg
.. |velocity-flat-unitree-go1| image:: ../_static/tasks/locomotion/go1_flat.jpg
.. |velocity-rough-unitree-go1| image:: ../_static/tasks/locomotion/go1_rough.jpg
.. |velocity-flat-unitree-go2| image:: ../_static/tasks/locomotion/go2_flat.jpg
.. |velocity-rough-unitree-go2| image:: ../_static/tasks/locomotion/go2_rough.jpg
