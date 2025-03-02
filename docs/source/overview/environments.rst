.. _environments:

Available Environments
======================

The following lists comprises of all the RL tasks implementations that are available in Isaac Lab.
While we try to keep this list up-to-date, you can always get the latest list of environments by
running the following command:

.. tab-set::
   :sync-group: os

   .. tab-item:: :icon:`fa-brands fa-linux` Linux
      :sync: linux

      .. code:: bash

         ./isaaclab.sh -p scripts/environments/list_envs.py

   .. tab-item:: :icon:`fa-brands fa-windows` Windows
      :sync: windows

      .. code:: batch

         isaaclab.bat -p scripts\environments\list_envs.py

We are actively working on adding more environments to the list. If you have any environments that
you would like to add to Isaac Lab, please feel free to open a pull request!

Single-agent
------------

Classic
~~~~~~~

Classic environments that are based on IsaacGymEnvs implementation of MuJoCo-style environments.

.. table::
    :widths: 33 37 30

    +------------------+-----------------------------+-------------------------------------------------------------------------+
    | World            | Environment ID              | Description                                                             |
    +==================+=============================+=========================================================================+
    | |humanoid|       | |humanoid-link|             | Move towards a direction with the MuJoCo humanoid robot                 |
    |                  |                             |                                                                         |
    |                  | |humanoid-direct-link|      |                                                                         |
    +------------------+-----------------------------+-------------------------------------------------------------------------+
    | |ant|            | |ant-link|                  | Move towards a direction with the MuJoCo ant robot                      |
    |                  |                             |                                                                         |
    |                  | |ant-direct-link|           |                                                                         |
    +------------------+-----------------------------+-------------------------------------------------------------------------+
    | |cartpole|       | |cartpole-link|             | Move the cart to keep the pole upwards in the classic cartpole control  |
    |                  |                             |                                                                         |
    |                  | |cartpole-direct-link|      |                                                                         |
    +------------------+-----------------------------+-------------------------------------------------------------------------+
    | |cartpole|       | |cartpole-rgb-link|         | Move the cart to keep the pole upwards in the classic cartpole control  |
    |                  |                             | and perceptive inputs                                                   |
    |                  | |cartpole-depth-link|       |                                                                         |
    |                  |                             |                                                                         |
    |                  | |cartpole-rgb-direct-link|  |                                                                         |
    |                  |                             |                                                                         |
    |                  | |cartpole-depth-direct-link||                                                                         |
    +------------------+-----------------------------+-------------------------------------------------------------------------+
    | |cartpole|       | |cartpole-resnet-link|      | Move the cart to keep the pole upwards in the classic cartpole control  |
    |                  |                             | based off of features extracted from perceptive inputs with pre-trained |
    |                  | |cartpole-theia-link|       | frozen vision encoders                                                  |
    +------------------+-----------------------------+-------------------------------------------------------------------------+

.. |humanoid| image:: ../_static/tasks/classic/humanoid.jpg
.. |ant| image:: ../_static/tasks/classic/ant.jpg
.. |cartpole| image:: ../_static/tasks/classic/cartpole.jpg

.. |humanoid-link| replace:: `Isaac-Humanoid-v0 <https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/humanoid/humanoid_env_cfg.py>`__
.. |ant-link| replace:: `Isaac-Ant-v0 <https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant/ant_env_cfg.py>`__
.. |cartpole-link| replace:: `Isaac-Cartpole-v0 <https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/cartpole/cartpole_env_cfg.py>`__
.. |cartpole-rgb-link| replace:: `Isaac-Cartpole-RGB-v0 <https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/cartpole/cartpole_camera_env_cfg.py>`__
.. |cartpole-depth-link| replace:: `Isaac-Cartpole-Depth-v0 <https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/cartpole/cartpole_camera_env_cfg.py>`__
.. |cartpole-resnet-link| replace:: `Isaac-Cartpole-RGB-ResNet18-v0 <https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/cartpole/cartpole_camera_env_cfg.py>`__
.. |cartpole-theia-link| replace:: `Isaac-Cartpole-RGB-TheiaTiny-v0 <https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/cartpole/cartpole_camera_env_cfg.py>`__


.. |humanoid-direct-link| replace:: `Isaac-Humanoid-Direct-v0 <https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/direct/humanoid/humanoid_env.py>`__
.. |ant-direct-link| replace:: `Isaac-Ant-Direct-v0 <https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/direct/ant/ant_env.py>`__
.. |cartpole-direct-link| replace:: `Isaac-Cartpole-Direct-v0 <https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/direct/cartpole/cartpole_env.py>`__
.. |cartpole-rgb-direct-link| replace:: `Isaac-Cartpole-RGB-Camera-Direct-v0 <https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/direct/cartpole/cartpole_camera_env.py>`__
.. |cartpole-depth-direct-link| replace:: `Isaac-Cartpole-Depth-Camera-Direct-v0 <https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/direct/cartpole/cartpole_camera_env.py>`__

Manipulation
~~~~~~~~~~~~

Environments based on fixed-arm manipulation tasks.

For many of these tasks, we include configurations with different arm action spaces. For example,
for the lift-cube environment:

* |lift-cube-link|: Franka arm with joint position control
* |lift-cube-ik-abs-link|: Franka arm with absolute IK control
* |lift-cube-ik-rel-link|: Franka arm with relative IK control

.. table::
    :widths: 33 37 30

    +--------------------+-------------------------+-----------------------------------------------------------------------------+
    | World              | Environment ID          | Description                                                                 |
    +====================+=========================+=============================================================================+
    | |reach-franka|     | |reach-franka-link|     | Move the end-effector to a sampled target pose with the Franka robot        |
    +--------------------+-------------------------+-----------------------------------------------------------------------------+
    | |reach-ur10|       | |reach-ur10-link|       | Move the end-effector to a sampled target pose with the UR10 robot          |
    +--------------------+-------------------------+-----------------------------------------------------------------------------+
    | |lift-cube|        | |lift-cube-link|        | Pick a cube and bring it to a sampled target position with the Franka robot |
    +--------------------+-------------------------+-----------------------------------------------------------------------------+
    | |stack-cube|       | |stack-cube-link|       | Stack three cubes (bottom to top: blue, red, green) with the Franka robot   |
    +--------------------+-------------------------+-----------------------------------------------------------------------------+
    | |cabi-franka|      | |cabi-franka-link|      | Grasp the handle of a cabinet's drawer and open it with the Franka robot    |
    |                    |                         |                                                                             |
    |                    | |franka-direct-link|    |                                                                             |
    +--------------------+-------------------------+-----------------------------------------------------------------------------+
    | |cube-allegro|     | |cube-allegro-link|     | In-hand reorientation of a cube using Allegro hand                          |
    |                    |                         |                                                                             |
    |                    | |allegro-direct-link|   |                                                                             |
    +--------------------+-------------------------+-----------------------------------------------------------------------------+
    | |cube-shadow|      | |cube-shadow-link|      | In-hand reorientation of a cube using Shadow hand                           |
    |                    |                         |                                                                             |
    |                    | |cube-shadow-ff-link|   |                                                                             |
    |                    |                         |                                                                             |
    |                    | |cube-shadow-lstm-link| |                                                                             |
    +--------------------+-------------------------+-----------------------------------------------------------------------------+
    | |cube-shadow|      | |cube-shadow-vis-link|  | In-hand reorientation of a cube using Shadow hand using perceptive inputs   |
    +--------------------+-------------------------+-----------------------------------------------------------------------------+

.. |reach-franka| image:: ../_static/tasks/manipulation/franka_reach.jpg
.. |reach-ur10| image:: ../_static/tasks/manipulation/ur10_reach.jpg
.. |lift-cube| image:: ../_static/tasks/manipulation/franka_lift.jpg
.. |cabi-franka| image:: ../_static/tasks/manipulation/franka_open_drawer.jpg
.. |cube-allegro| image:: ../_static/tasks/manipulation/allegro_cube.jpg
.. |cube-shadow| image:: ../_static/tasks/manipulation/shadow_cube.jpg
.. |stack-cube| image:: ../_static/tasks/manipulation/franka_stack.jpg

.. |reach-franka-link| replace:: `Isaac-Reach-Franka-v0 <https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/reach/config/franka/joint_pos_env_cfg.py>`__
.. |reach-ur10-link| replace:: `Isaac-Reach-UR10-v0 <https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/reach/config/ur_10/joint_pos_env_cfg.py>`__
.. |lift-cube-link| replace:: `Isaac-Lift-Cube-Franka-v0 <https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/config/franka/joint_pos_env_cfg.py>`__
.. |lift-cube-ik-abs-link| replace:: `Isaac-Lift-Cube-Franka-IK-Abs-v0 <https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/config/franka/ik_abs_env_cfg.py>`__
.. |lift-cube-ik-rel-link| replace:: `Isaac-Lift-Cube-Franka-IK-Rel-v0 <https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/config/franka/ik_rel_env_cfg.py>`__
.. |cabi-franka-link| replace:: `Isaac-Open-Drawer-Franka-v0 <https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/cabinet/config/franka/joint_pos_env_cfg.py>`__
.. |franka-direct-link| replace:: `Isaac-Franka-Cabinet-Direct-v0 <https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/direct/franka_cabinet/franka_cabinet_env.py>`__
.. |cube-allegro-link| replace:: `Isaac-Repose-Cube-Allegro-v0 <https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/inhand/config/allegro_hand/allegro_env_cfg.py>`__
.. |allegro-direct-link| replace:: `Isaac-Repose-Cube-Allegro-Direct-v0 <https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/direct/allegro_hand/allegro_hand_env_cfg.py>`__
.. |stack-cube-link| replace:: `Isaac-Stack-Cube-Franka-v0 <https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/stack/config/franka/stack_joint_pos_env_cfg.py>`__

.. |cube-shadow-link| replace:: `Isaac-Repose-Cube-Shadow-Direct-v0 <https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/direct/shadow_hand/shadow_hand_env_cfg.py>`__
.. |cube-shadow-ff-link| replace:: `Isaac-Repose-Cube-Shadow-OpenAI-FF-Direct-v0 <https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/direct/shadow_hand/shadow_hand_env_cfg.py>`__
.. |cube-shadow-lstm-link| replace:: `Isaac-Repose-Cube-Shadow-OpenAI-LSTM-Direct-v0 <https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/direct/shadow_hand/shadow_hand_env_cfg.py>`__
.. |cube-shadow-vis-link| replace:: `Isaac-Repose-Cube-Shadow-Vision-Direct-v0 <https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/direct/shadow_hand/shadow_hand_vision_env.py>`__

Contact-rich Manipulation
~~~~~~~~~~~~~~~~~~~~~~~~~

Environments based on contact-rich manipulation tasks such as peg insertion, gear meshing and nut-bolt fastening.

These tasks share the same task configurations and control options. You can switch between them by specifying the task name.
For example:

* |factory-peg-link|: Peg insertion with the Franka arm
* |factory-gear-link|: Gear meshing with the Franka arm
* |factory-nut-link|: Nut-Bolt fastening with the Franka arm

.. table::
    :widths: 33 37 30

    +--------------------+-------------------------+-----------------------------------------------------------------------------+
    | World              | Environment ID          | Description                                                                 |
    +====================+=========================+=============================================================================+
    | |factory-peg|      | |factory-peg-link|      | Insert peg into the socket with the Franka robot                            |
    +--------------------+-------------------------+-----------------------------------------------------------------------------+
    | |factory-gear|     | |factory-gear-link|     | Insert and mesh gear into the base with other gears, using the Franka robot |
    +--------------------+-------------------------+-----------------------------------------------------------------------------+
    | |factory-nut|      | |factory-nut-link|      | Thread the nut onto the first 2 threads of the bolt, using the Franka robot |
    +--------------------+-------------------------+-----------------------------------------------------------------------------+

.. |factory-peg| image:: ../_static/tasks/factory/peg_insert.jpg
.. |factory-gear| image:: ../_static/tasks/factory/gear_mesh.jpg
.. |factory-nut| image:: ../_static/tasks/factory/nut_thread.jpg

.. |factory-peg-link| replace:: `Isaac-Factory-PegInsert-Direct-v0 <https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/direct/factory/factory_env_cfg.py>`__
.. |factory-gear-link| replace:: `Isaac-Factory-GearMesh-Direct-v0 <https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/direct/factory/factory_env_cfg.py>`__
.. |factory-nut-link| replace:: `Isaac-Factory-NutThread-Direct-v0 <https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/direct/factory/factory_env_cfg.py>`__

Locomotion
~~~~~~~~~~

Environments based on legged locomotion tasks.

.. table::
    :widths: 33 37 30

    +------------------------------+----------------------------------------------+------------------------------------------------------------------------------+
    | World                        | Environment ID                               | Description                                                                  |
    +==============================+==============================================+==============================================================================+
    | |velocity-flat-anymal-b|     | |velocity-flat-anymal-b-link|                | Track a velocity command on flat terrain with the Anymal B robot             |
    +------------------------------+----------------------------------------------+------------------------------------------------------------------------------+
    | |velocity-rough-anymal-b|    | |velocity-rough-anymal-b-link|               | Track a velocity command on rough terrain with the Anymal B robot            |
    +------------------------------+----------------------------------------------+------------------------------------------------------------------------------+
    | |velocity-flat-anymal-c|     | |velocity-flat-anymal-c-link|                | Track a velocity command on flat terrain with the Anymal C robot             |
    |                              |                                              |                                                                              |
    |                              | |velocity-flat-anymal-c-direct-link|         |                                                                              |
    +------------------------------+----------------------------------------------+------------------------------------------------------------------------------+
    | |velocity-rough-anymal-c|    | |velocity-rough-anymal-c-link|               | Track a velocity command on rough terrain with the Anymal C robot            |
    |                              |                                              |                                                                              |
    |                              | |velocity-rough-anymal-c-direct-link|        |                                                                              |
    +------------------------------+----------------------------------------------+------------------------------------------------------------------------------+
    | |velocity-flat-anymal-d|     | |velocity-flat-anymal-d-link|                | Track a velocity command on flat terrain with the Anymal D robot             |
    +------------------------------+----------------------------------------------+------------------------------------------------------------------------------+
    | |velocity-rough-anymal-d|    | |velocity-rough-anymal-d-link|               | Track a velocity command on rough terrain with the Anymal D robot            |
    +------------------------------+----------------------------------------------+------------------------------------------------------------------------------+
    | |velocity-flat-unitree-a1|   | |velocity-flat-unitree-a1-link|              | Track a velocity command on flat terrain with the Unitree A1 robot           |
    +------------------------------+----------------------------------------------+------------------------------------------------------------------------------+
    | |velocity-rough-unitree-a1|  | |velocity-rough-unitree-a1-link|             | Track a velocity command on rough terrain with the Unitree A1 robot          |
    +------------------------------+----------------------------------------------+------------------------------------------------------------------------------+
    | |velocity-flat-unitree-go1|  | |velocity-flat-unitree-go1-link|             | Track a velocity command on flat terrain with the Unitree Go1 robot          |
    +------------------------------+----------------------------------------------+------------------------------------------------------------------------------+
    | |velocity-rough-unitree-go1| | |velocity-rough-unitree-go1-link|            | Track a velocity command on rough terrain with the Unitree Go1 robot         |
    +------------------------------+----------------------------------------------+------------------------------------------------------------------------------+
    | |velocity-flat-unitree-go2|  | |velocity-flat-unitree-go2-link|             | Track a velocity command on flat terrain with the Unitree Go2 robot          |
    +------------------------------+----------------------------------------------+------------------------------------------------------------------------------+
    | |velocity-rough-unitree-go2| | |velocity-rough-unitree-go2-link|            | Track a velocity command on rough terrain with the Unitree Go2 robot         |
    +------------------------------+----------------------------------------------+------------------------------------------------------------------------------+
    | |velocity-flat-spot|         | |velocity-flat-spot-link|                    | Track a velocity command on flat terrain with the Boston Dynamics Spot robot |
    +------------------------------+----------------------------------------------+------------------------------------------------------------------------------+
    | |velocity-flat-h1|           | |velocity-flat-h1-link|                      | Track a velocity command on flat terrain with the Unitree H1 robot           |
    +------------------------------+----------------------------------------------+------------------------------------------------------------------------------+
    | |velocity-rough-h1|          | |velocity-rough-h1-link|                     | Track a velocity command on rough terrain with the Unitree H1 robot          |
    +------------------------------+----------------------------------------------+------------------------------------------------------------------------------+
    | |velocity-flat-g1|           | |velocity-flat-g1-link|                      | Track a velocity command on flat terrain with the Unitree G1 robot           |
    +------------------------------+----------------------------------------------+------------------------------------------------------------------------------+
    | |velocity-rough-g1|          | |velocity-rough-g1-link|                     | Track a velocity command on rough terrain with the Unitree G1 robot          |
    +------------------------------+----------------------------------------------+------------------------------------------------------------------------------+

.. |velocity-flat-anymal-b-link| replace:: `Isaac-Velocity-Flat-Anymal-B-v0 <https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/anymal_b/flat_env_cfg.py>`__
.. |velocity-rough-anymal-b-link| replace:: `Isaac-Velocity-Rough-Anymal-B-v0 <https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/anymal_b/rough_env_cfg.py>`__

.. |velocity-flat-anymal-c-link| replace:: `Isaac-Velocity-Flat-Anymal-C-v0 <https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/anymal_c/flat_env_cfg.py>`__
.. |velocity-rough-anymal-c-link| replace:: `Isaac-Velocity-Rough-Anymal-C-v0 <https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/anymal_c/rough_env_cfg.py>`__

.. |velocity-flat-anymal-c-direct-link| replace:: `Isaac-Velocity-Flat-Anymal-C-Direct-v0 <https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/direct/anymal_c/anymal_c_env.py>`__
.. |velocity-rough-anymal-c-direct-link| replace:: `Isaac-Velocity-Rough-Anymal-C-Direct-v0 <https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/direct/anymal_c/anymal_c_env.py>`__

.. |velocity-flat-anymal-d-link| replace:: `Isaac-Velocity-Flat-Anymal-D-v0 <https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/anymal_d/flat_env_cfg.py>`__
.. |velocity-rough-anymal-d-link| replace:: `Isaac-Velocity-Rough-Anymal-D-v0 <https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/anymal_d/rough_env_cfg.py>`__

.. |velocity-flat-unitree-a1-link| replace:: `Isaac-Velocity-Flat-Unitree-A1-v0 <https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/a1/flat_env_cfg.py>`__
.. |velocity-rough-unitree-a1-link| replace:: `Isaac-Velocity-Rough-Unitree-A1-v0 <https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/a1/rough_env_cfg.py>`__

.. |velocity-flat-unitree-go1-link| replace:: `Isaac-Velocity-Flat-Unitree-Go1-v0 <https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go1/flat_env_cfg.py>`__
.. |velocity-rough-unitree-go1-link| replace:: `Isaac-Velocity-Rough-Unitree-Go1-v0 <https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go1/rough_env_cfg.py>`__

.. |velocity-flat-unitree-go2-link| replace:: `Isaac-Velocity-Flat-Unitree-Go2-v0 <https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2/flat_env_cfg.py>`__
.. |velocity-rough-unitree-go2-link| replace:: `Isaac-Velocity-Rough-Unitree-Go2-v0 <https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2/rough_env_cfg.py>`__

.. |velocity-flat-spot-link| replace:: `Isaac-Velocity-Flat-Spot-v0 <https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/spot/flat_env_cfg.py>`__

.. |velocity-flat-h1-link| replace:: `Isaac-Velocity-Flat-H1-v0 <https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/h1/flat_env_cfg.py>`__
.. |velocity-rough-h1-link| replace:: `Isaac-Velocity-Rough-H1-v0 <https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/h1/rough_env_cfg.py>`__

.. |velocity-flat-g1-link| replace:: `Isaac-Velocity-Flat-G1-v0 <https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/flat_env_cfg.py>`__
.. |velocity-rough-g1-link| replace:: `Isaac-Velocity-Rough-G1-v0 <https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/rough_env_cfg.py>`__


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
.. |velocity-flat-spot| image:: ../_static/tasks/locomotion/spot_flat.jpg
.. |velocity-flat-h1| image:: ../_static/tasks/locomotion/h1_flat.jpg
.. |velocity-rough-h1| image:: ../_static/tasks/locomotion/h1_rough.jpg
.. |velocity-flat-g1| image:: ../_static/tasks/locomotion/g1_flat.jpg
.. |velocity-rough-g1| image:: ../_static/tasks/locomotion/g1_rough.jpg

Navigation
~~~~~~~~~~

.. table::
    :widths: 33 37 30

    +----------------+---------------------+-----------------------------------------------------------------------------+
    | World          | Environment ID      | Description                                                                 |
    +================+=====================+=============================================================================+
    | |anymal_c_nav| | |anymal_c_nav-link| | Navigate towards a target x-y position and heading with the ANYmal C robot. |
    +----------------+---------------------+-----------------------------------------------------------------------------+

.. |anymal_c_nav-link| replace:: `Isaac-Navigation-Flat-Anymal-C-v0 <https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/navigation/config/anymal_c/navigation_env_cfg.py>`__

.. |anymal_c_nav| image:: ../_static/tasks/navigation/anymal_c_nav.jpg


Others
~~~~~~

.. note::

    Adversarial Motion Priors (AMP) training is only available with the `skrl` library, as it is the only one of the currently
    integrated libraries that supports it out-of-the-box (for the other libraries, it is necessary to implement the algorithm and architectures).
    See the `skrl's AMP Documentation <https://skrl.readthedocs.io/en/latest/api/agents/amp.html>`_ for more information.
    The AMP algorithm can be activated by adding the command line input ``--algorithm AMP`` to the train/play script.

    For evaluation, the play script's command line input ``--real-time`` allows the interaction loop between the environment and the agent to run in real time, if possible.

.. table::
    :widths: 33 37 30

    +----------------+---------------------------+-----------------------------------------------------------------------------+
    | World          | Environment ID            | Description                                                                 |
    +================+===========================+=============================================================================+
    | |quadcopter|   | |quadcopter-link|         | Fly and hover the Crazyflie copter at a goal point by applying thrust.      |
    +----------------+---------------------------+-----------------------------------------------------------------------------+
    | |humanoid_amp| | |humanoid_amp_dance-link| | Move a humanoid robot by imitating different pre-recorded human animations  |
    |                |                           | (Adversarial Motion Priors).                                                |
    |                | |humanoid_amp_run-link|   |                                                                             |
    |                |                           |                                                                             |
    |                | |humanoid_amp_walk-link|  |                                                                             |
    +----------------+---------------------------+-----------------------------------------------------------------------------+

.. |quadcopter-link| replace:: `Isaac-Quadcopter-Direct-v0 <https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/direct/quadcopter/quadcopter_env.py>`__
.. |humanoid_amp_dance-link| replace:: `Isaac-Humanoid-AMP-Dance-Direct-v0 <https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/direct/humanoid_amp/humanoid_amp_env_cfg.py>`__
.. |humanoid_amp_run-link| replace:: `Isaac-Humanoid-AMP-Run-Direct-v0 <https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/direct/humanoid_amp/humanoid_amp_env_cfg.py>`__
.. |humanoid_amp_walk-link| replace:: `Isaac-Humanoid-AMP-Walk-Direct-v0 <https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/direct/humanoid_amp/humanoid_amp_env_cfg.py>`__

.. |quadcopter| image:: ../_static/tasks/others/quadcopter.jpg
.. |humanoid_amp| image:: ../_static/tasks/others/humanoid_amp.jpg


Multi-agent
------------

.. note::

    True mutli-agent training is only available with the `skrl` library, see the `Multi-Agents Documentation <https://skrl.readthedocs.io/en/latest/api/multi_agents.html>`_ for more information.
    It supports the `IPPO` and `MAPPO` algorithms, which can be activated by adding the command line input ``--algorithm IPPO`` or ``--algorithm MAPPO`` to the train/play script.
    If these environments are run with other libraries or without the `IPPO` or `MAPPO` flags, they will be converted to single-agent environments under the hood.


Classic
~~~~~~~

.. table::
    :widths: 33 37 30

    +------------------------+------------------------------------+-----------------------------------------------------------------------------------------------------------------------+
    | World                  | Environment ID                     | Description                                                                                                           |
    +========================+====================================+=======================================================================================================================+
    | |cart-double-pendulum| | |cart-double-pendulum-direct-link| | Move the cart and the pendulum to keep the last one upwards in the classic inverted double pendulum on a cart control |
    +------------------------+------------------------------------+-----------------------------------------------------------------------------------------------------------------------+

.. |cart-double-pendulum| image:: ../_static/tasks/classic/cart_double_pendulum.jpg

.. |cart-double-pendulum-direct-link| replace:: `Isaac-Cart-Double-Pendulum-Direct-v0 <https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/direct/cart_double_pendulum/cart_double_pendulum_env.py>`__

Manipulation
~~~~~~~~~~~~

Environments based on fixed-arm manipulation tasks.

.. table::
    :widths: 33 37 30

    +----------------------+--------------------------------+--------------------------------------------------------+
    | World                | Environment ID                 | Description                                            |
    +======================+================================+========================================================+
    | |shadow-hand-over|   | |shadow-hand-over-direct-link| | Passing an object from one hand over to the other hand |
    +----------------------+--------------------------------+--------------------------------------------------------+

.. |shadow-hand-over| image:: ../_static/tasks/manipulation/shadow_hand_over.jpg

.. |shadow-hand-over-direct-link| replace:: `Isaac-Shadow-Hand-Over-Direct-v0 <https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/direct/shadow_hand_over/shadow_hand_over_env.py>`__

|

Comprehensive List of Environments
==================================

.. list-table::
    :widths: 33 25 19 25

    * - **Task Name**
      - **Inference Task Name**
      - **Workflow**
      - **RL Library**
    * - Isaac-Ant-Direct-v0
      -
      - Direct
      - **rl_games** (PPO), **rsl_rl** (PPO), **skrl** (PPO)
    * - Isaac-Ant-v0
      -
      - Manager Based
      - **rsl_rl** (PPO), **rl_games** (PPO), **skrl** (PPO), **sb3** (PPO)
    * - Isaac-Cart-Double-Pendulum-Direct-v0
      -
      - Direct
      - **rl_games** (PPO), **skrl** (IPPO, PPO, MAPPO)
    * - Isaac-Cartpole-Depth-Camera-Direct-v0
      -
      - Direct
      - **rl_games** (PPO), **skrl** (PPO)
    * - Isaac-Cartpole-Depth-v0
      -
      - Manager Based
      - **rl_games** (PPO)
    * - Isaac-Cartpole-Direct-v0
      -
      - Direct
      - **rl_games** (PPO), **rsl_rl** (PPO), **skrl** (PPO), **sb3** (PPO)
    * - Isaac-Cartpole-RGB-Camera-Direct-v0
      -
      - Direct
      - **rl_games** (PPO), **skrl** (PPO)
    * - Isaac-Cartpole-RGB-ResNet18-v0
      -
      - Manager Based
      - **rl_games** (PPO)
    * - Isaac-Cartpole-RGB-TheiaTiny-v0
      -
      - Manager Based
      - **rl_games** (PPO)
    * - Isaac-Cartpole-RGB-v0
      -
      - Manager Based
      - **rl_games** (PPO)
    * - Isaac-Cartpole-v0
      -
      - Manager Based
      - **rl_games** (PPO), **rsl_rl** (PPO), **skrl** (PPO), **sb3** (PPO)
    * - Isaac-Factory-GearMesh-Direct-v0
      -
      - Direct
      - **rl_games** (PPO)
    * - Isaac-Factory-NutThread-Direct-v0
      -
      - Direct
      - **rl_games** (PPO)
    * - Isaac-Factory-PegInsert-Direct-v0
      -
      - Direct
      - **rl_games** (PPO)
    * - Isaac-Franka-Cabinet-Direct-v0
      -
      - Direct
      - **rl_games** (PPO), **rsl_rl** (PPO), **skrl** (PPO)
    * - Isaac-Humanoid-AMP-Dance-Direct-v0
      -
      - Direct
      - **skrl** (AMP)
    * - Isaac-Humanoid-AMP-Run-Direct-v0
      -
      - Direct
      - **skrl** (AMP)
    * - Isaac-Humanoid-AMP-Walk-Direct-v0
      -
      - Direct
      - **skrl** (AMP)
    * - Isaac-Humanoid-Direct-v0
      -
      - Direct
      - **rl_games** (PPO), **rsl_rl** (PPO), **skrl** (PPO)
    * - Isaac-Humanoid-v0
      -
      - Manager Based
      - **rsl_rl** (PPO), **rl_games** (PPO), **skrl** (PPO), **sb3** (PPO)
    * - Isaac-Lift-Cube-Franka-IK-Abs-v0
      -
      - Manager Based
      -
    * - Isaac-Lift-Cube-Franka-IK-Rel-v0
      -
      - Manager Based
      -
    * - Isaac-Lift-Cube-Franka-v0
      - Isaac-Lift-Cube-Franka-Play-v0
      - Manager Based
      - **rsl_rl** (PPO), **skrl** (PPO), **rl_games** (PPO), **sb3** (PPO)
    * - Isaac-Lift-Teddy-Bear-Franka-IK-Abs-v0
      -
      - Manager Based
      -
    * - Isaac-Navigation-Flat-Anymal-C-v0
      - Isaac-Navigation-Flat-Anymal-C-Play-v0
      - Manager Based
      - **rsl_rl** (PPO), **skrl** (PPO)
    * - Isaac-Open-Drawer-Franka-IK-Abs-v0
      -
      - Manager Based
      -
    * - Isaac-Open-Drawer-Franka-IK-Rel-v0
      -
      - Manager Based
      -
    * - Isaac-Open-Drawer-Franka-v0
      - Isaac-Open-Drawer-Franka-Play-v0
      - Manager Based
      - **rsl_rl** (PPO), **rl_games** (PPO), **skrl** (PPO)
    * - Isaac-Quadcopter-Direct-v0
      -
      - Direct
      - **rl_games** (PPO), **rsl_rl** (PPO), **skrl** (PPO)
    * - Isaac-Reach-Franka-IK-Abs-v0
      -
      - Manager Based
      -
    * - Isaac-Reach-Franka-IK-Rel-v0
      -
      - Manager Based
      -
    * - Isaac-Reach-Franka-OSC-v0
      - Isaac-Reach-Franka-OSC-Play-v0
      - Manager Based
      - **rsl_rl** (PPO)
    * - Isaac-Reach-Franka-v0
      - Isaac-Reach-Franka-Play-v0
      - Manager Based
      - **rl_games** (PPO), **rsl_rl** (PPO), **skrl** (PPO)
    * - Isaac-Reach-UR10-v0
      - Isaac-Reach-UR10-Play-v0
      - Manager Based
      - **rl_games** (PPO), **rsl_rl** (PPO), **skrl** (PPO)
    * - Isaac-Repose-Cube-Allegro-Direct-v0
      -
      - Direct
      - **rl_games** (PPO), **rsl_rl** (PPO), **skrl** (PPO)
    * - Isaac-Repose-Cube-Allegro-NoVelObs-v0
      - Isaac-Repose-Cube-Allegro-NoVelObs-Play-v0
      - Manager Based
      - **rsl_rl** (PPO), **rl_games** (PPO), **skrl** (PPO)
    * - Isaac-Repose-Cube-Allegro-v0
      - Isaac-Repose-Cube-Allegro-Play-v0
      - Manager Based
      - **rsl_rl** (PPO), **rl_games** (PPO), **skrl** (PPO)
    * - Isaac-Repose-Cube-Shadow-Direct-v0
      -
      - Direct
      - **rl_games** (PPO), **rsl_rl** (PPO), **skrl** (PPO)
    * - Isaac-Repose-Cube-Shadow-OpenAI-FF-Direct-v0
      -
      - Direct
      - **rl_games** (FF), **rsl_rl** (PPO), **skrl** (PPO)
    * - Isaac-Repose-Cube-Shadow-OpenAI-LSTM-Direct-v0
      -
      - Direct
      - **rl_games** (LSTM)
    * - Isaac-Repose-Cube-Shadow-Vision-Direct-v0
      - Isaac-Repose-Cube-Shadow-Vision-Direct-Play-v0
      - Direct
      - **rsl_rl** (PPO), **rl_games** (VISION)
    * - Isaac-Shadow-Hand-Over-Direct-v0
      -
      - Direct
      - **rl_games** (PPO), **skrl** (IPPO, PPO, MAPPO)
    * - Isaac-Stack-Cube-Franka-IK-Rel-v0
      -
      - Manager Based
      -
    * - Isaac-Stack-Cube-Franka-v0
      -
      - Manager Based
      -
    * - Isaac-Stack-Cube-Instance-Randomize-Franka-IK-Rel-v0
      -
      - Manager Based
      -
    * - Isaac-Stack-Cube-Instance-Randomize-Franka-v0
      -
      - Manager Based
      -
    * - Isaac-Velocity-Flat-Anymal-B-v0
      - Isaac-Velocity-Flat-Anymal-B-Play-v0
      - Manager Based
      - **rsl_rl** (PPO), **skrl** (PPO)
    * - Isaac-Velocity-Flat-Anymal-C-Direct-v0
      -
      - Direct
      - **rl_games** (PPO), **rsl_rl** (PPO), **skrl** (PPO)
    * - Isaac-Velocity-Flat-Anymal-C-v0
      - Isaac-Velocity-Flat-Anymal-C-Play-v0
      - Manager Based
      - **rsl_rl** (PPO), **rl_games** (PPO), **skrl** (PPO)
    * - Isaac-Velocity-Flat-Anymal-D-v0
      - Isaac-Velocity-Flat-Anymal-D-Play-v0
      - Manager Based
      - **rsl_rl** (PPO), **skrl** (PPO)
    * - Isaac-Velocity-Flat-Cassie-v0
      - Isaac-Velocity-Flat-Cassie-Play-v0
      - Manager Based
      - **rsl_rl** (PPO), **skrl** (PPO)
    * - Isaac-Velocity-Flat-G1-v0
      - Isaac-Velocity-Flat-G1-Play-v0
      - Manager Based
      - **rsl_rl** (PPO), **skrl** (PPO)
    * - Isaac-Velocity-Flat-H1-v0
      - Isaac-Velocity-Flat-H1-Play-v0
      - Manager Based
      - **rsl_rl** (PPO), **skrl** (PPO)
    * - Isaac-Velocity-Flat-Spot-v0
      - Isaac-Velocity-Flat-Spot-Play-v0
      - Manager Based
      - **rsl_rl** (PPO), **skrl** (PPO)
    * - Isaac-Velocity-Flat-Unitree-A1-v0
      - Isaac-Velocity-Flat-Unitree-A1-Play-v0
      - Manager Based
      - **rsl_rl** (PPO), **skrl** (PPO)
    * - Isaac-Velocity-Flat-Unitree-Go1-v0
      - Isaac-Velocity-Flat-Unitree-Go1-Play-v0
      - Manager Based
      - **rsl_rl** (PPO), **skrl** (PPO)
    * - Isaac-Velocity-Flat-Unitree-Go2-v0
      - Isaac-Velocity-Flat-Unitree-Go2-Play-v0
      - Manager Based
      - **rsl_rl** (PPO), **skrl** (PPO)
    * - Isaac-Velocity-Rough-Anymal-B-v0
      - Isaac-Velocity-Rough-Anymal-B-Play-v0
      - Manager Based
      - **rsl_rl** (PPO), **skrl** (PPO)
    * - Isaac-Velocity-Rough-Anymal-C-Direct-v0
      -
      - Direct
      - **rl_games** (PPO), **rsl_rl** (PPO), **skrl** (PPO)
    * - Isaac-Velocity-Rough-Anymal-C-v0
      - Isaac-Velocity-Rough-Anymal-C-Play-v0
      - Manager Based
      - **rl_games** (PPO), **rsl_rl** (PPO), **skrl** (PPO)
    * - Isaac-Velocity-Rough-Anymal-D-v0
      - Isaac-Velocity-Rough-Anymal-D-Play-v0
      - Manager Based
      - **rsl_rl** (PPO), **skrl** (PPO)
    * - Isaac-Velocity-Rough-Cassie-v0
      - Isaac-Velocity-Rough-Cassie-Play-v0
      - Manager Based
      - **rsl_rl** (PPO), **skrl** (PPO)
    * - Isaac-Velocity-Rough-G1-v0
      - Isaac-Velocity-Rough-G1-Play-v0
      - Manager Based
      - **rsl_rl** (PPO), **skrl** (PPO)
    * - Isaac-Velocity-Rough-H1-v0
      - Isaac-Velocity-Rough-H1-Play-v0
      - Manager Based
      - **rsl_rl** (PPO), **skrl** (PPO)
    * - Isaac-Velocity-Rough-Unitree-A1-v0
      - Isaac-Velocity-Rough-Unitree-A1-Play-v0
      - Manager Based
      - **rsl_rl** (PPO), **skrl** (PPO)
    * - Isaac-Velocity-Rough-Unitree-Go1-v0
      - Isaac-Velocity-Rough-Unitree-Go1-Play-v0
      - Manager Based
      - **rsl_rl** (PPO), **skrl** (PPO)
    * - Isaac-Velocity-Rough-Unitree-Go2-v0
      - Isaac-Velocity-Rough-Unitree-Go2-Play-v0
      - Manager Based
      - **rsl_rl** (PPO), **skrl** (PPO)
