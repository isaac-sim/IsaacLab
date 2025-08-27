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
    |                  |                             | and perceptive inputs. Requires running with ``--enable_cameras``.      |
    |                  | |cartpole-depth-link|       |                                                                         |
    |                  |                             |                                                                         |
    |                  | |cartpole-rgb-direct-link|  |                                                                         |
    |                  |                             |                                                                         |
    |                  | |cartpole-depth-direct-link||                                                                         |
    +------------------+-----------------------------+-------------------------------------------------------------------------+
    | |cartpole|       | |cartpole-resnet-link|      | Move the cart to keep the pole upwards in the classic cartpole control  |
    |                  |                             | based off of features extracted from perceptive inputs with pre-trained |
    |                  | |cartpole-theia-link|       | frozen vision encoders. Requires running with ``--enable_cameras``.     |
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
    | |stack-cube|       | |stack-cube-link|       | Stack three cubes (bottom to top: blue, red, green) with the Franka robot.  |
    |                    |                         | Blueprint env used for the NVIDIA Isaac GR00T blueprint for synthetic       |
    |                    | |stack-cube-bp-link|    | manipulation motion generation                                              |
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
    | |cube-shadow|      | |cube-shadow-vis-link|  | In-hand reorientation of a cube using Shadow hand using perceptive inputs.  |
    |                    |                         | Requires running with ``--enable_cameras``.                                 |
    +--------------------+-------------------------+-----------------------------------------------------------------------------+
    | |gr1_pick_place|   | |gr1_pick_place-link|   | Pick up and place an object in a basket with a GR-1 humanoid robot          |
    +--------------------+-------------------------+-----------------------------------------------------------------------------+
    | |gr1_pp_waist|     | |gr1_pp_waist-link|     | Pick up and place an object in a basket with a GR-1 humanoid robot          |
    |                    |                         | with waist degrees-of-freedom enables that provides a wider reach space.    |
    +--------------------+-------------------------+-----------------------------------------------------------------------------+

.. |reach-franka| image:: ../_static/tasks/manipulation/franka_reach.jpg
.. |reach-ur10| image:: ../_static/tasks/manipulation/ur10_reach.jpg
.. |lift-cube| image:: ../_static/tasks/manipulation/franka_lift.jpg
.. |cabi-franka| image:: ../_static/tasks/manipulation/franka_open_drawer.jpg
.. |cube-allegro| image:: ../_static/tasks/manipulation/allegro_cube.jpg
.. |cube-shadow| image:: ../_static/tasks/manipulation/shadow_cube.jpg
.. |stack-cube| image:: ../_static/tasks/manipulation/franka_stack.jpg
.. |gr1_pick_place| image:: ../_static/tasks/manipulation/gr-1_pick_place.jpg
.. |gr1_pp_waist| image:: ../_static/tasks/manipulation/gr-1_pick_place_waist.jpg

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
.. |stack-cube-bp-link| replace:: `Isaac-Stack-Cube-Franka-IK-Rel-Blueprint-v0 <https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/stack/config/franka/stack_ik_rel_blueprint_env_cfg.py>`__
.. |gr1_pick_place-link| replace:: `Isaac-PickPlace-GR1T2-Abs-v0 <https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/pick_place/pickplace_gr1t2_env_cfg.py>`__
.. |gr1_pp_waist-link| replace:: `Isaac-PickPlace-GR1T2-WaistEnabled-Abs-v0 <https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/pick_place/pickplace_gr1t2_waist_enabled_env_cfg.py>`__

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

AutoMate
~~~~~~~~

Environments based on 100 diverse assembly tasks, each involving the insertion of a plug into a socket. These tasks share a common configuration and differ by th geometry and properties of the parts.

You can switch between tasks by specifying the corresponding asset ID. Available asset IDs include:

'00004', '00007', '00014', '00015', '00016', '00021', '00028', '00030', '00032', '00042', '00062', '00074', '00077', '00078', '00081', '00083', '00103', '00110', '00117', '00133', '00138', '00141', '00143', '00163', '00175', '00186', '00187', '00190', '00192', '00210', '00211', '00213', '00255', '00256', '00271', '00293', '00296', '00301', '00308', '00318', '00319', '00320', '00329', '00340', '00345', '00346', '00360', '00388', '00410', '00417', '00422', '00426', '00437', '00444', '00446', '00470', '00471', '00480', '00486', '00499', '00506', '00514', '00537', '00553', '00559', '00581', '00597', '00614', '00615', '00638', '00648', '00649', '00652', '00659', '00681', '00686', '00700', '00703', '00726', '00731', '00741', '00755', '00768', '00783', '00831', '00855', '00860', '00863', '01026', '01029', '01036', '01041', '01053', '01079', '01092', '01102', '01125', '01129', '01132', '01136'.

We provide environments for both disassembly and assembly.

.. attention::

  CUDA is required for running the AutoMate environments.
  Follow the below steps to install CUDA 12.8:

  .. code-block:: bash

      wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda_12.8.0_570.86.10_linux.run
      sudo sh cuda_12.8.0_570.86.10_linux.run

  When using conda, cuda toolkit can be installed with:

  .. code-block:: bash

      conda install cudatoolkit

  For addition instructions and Windows installation, please refer to the `CUDA installation page <https://developer.nvidia.com/cuda-12-8-0-download-archive>`_.

* |disassembly-link|: The plug starts inserted in the socket. A low-level controller lifts the plug out and moves it to a random position. This process is purely scripted and does not involve any learned policy. Therefore, it does not require policy training or evaluation. The resulting trajectories serve as demonstrations for the reverse process, i.e., learning to assemble. To run disassembly for a specific task: ``python source/isaaclab_tasks/isaaclab_tasks/direct/automate/run_disassembly_w_id.py --assembly_id=ASSEMBLY_ID --disassembly_dir=DISASSEMBLY_DIR``. All generated trajectories are saved to a local directory ``DISASSEMBLY_DIR``.
* |assembly-link|: The goal is to insert the plug into the socket. You can use this environment to train a policy via reinforcement learning or evaluate a pre-trained checkpoint.

  * To train an assembly policy, we run the command ``python source/isaaclab_tasks/isaaclab_tasks/direct/automate/run_w_id.py --assembly_id=ASSEMBLY_ID --train``. We can customize the training process using the optional flags: ``--headless`` to run without opening the GUI windows, ``--max_iterations=MAX_ITERATIONS`` to set the number of training iterations, ``--num_envs=NUM_ENVS`` to set the number of parallel environments during training, ``--seed=SEED`` to assign the random seed. The policy checkpoints will be saved automatically during training in the directory ``logs/rl_games/Assembly/test``.
  * To evaluate an assembly policy, we run the command ``python source/isaaclab_tasks/isaaclab_tasks/direct/automate/run_w_id.py --assembly_id=ASSEMBLY_ID --checkpoint=CHECKPOINT --log_eval``. The evaluation results are stored in ``evaluation_{ASSEMBLY_ID}.h5``.

.. table::
    :widths: 33 37 30

    +--------------------+-------------------------+-----------------------------------------------------------------------------+
    | World              | Environment ID          | Description                                                                 |
    +====================+=========================+=============================================================================+
    | |disassembly|      | |disassembly-link|      | Lift a plug out of the socket with the Franka robot                         |
    +--------------------+-------------------------+-----------------------------------------------------------------------------+
    | |assembly|         | |assembly-link|         | Insert a plug into its corresponding socket with the Franka robot           |
    +--------------------+-------------------------+-----------------------------------------------------------------------------+

.. |assembly| image:: ../_static/tasks/automate/00004.jpg
.. |disassembly| image:: ../_static/tasks/automate/01053_disassembly.jpg

.. |assembly-link| replace:: `Isaac-AutoMate-Assembly-Direct-v0 <https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/direct/automate/assembly_env_cfg.py>`__
.. |disassembly-link| replace:: `Isaac-AutoMate-Disassembly-Direct-v0 <https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/direct/automate/disassembly_env_cfg.py>`__

FORGE
~~~~~~~~

FORGE environments extend Factory environments with:

* Force sensing: Add observations for force experienced by the end-effector.
* Excessive force penalty: Add an option to penalize the agent for excessive contact forces.
* Dynamics randomization: Randomize controller gains, asset properties (friction, mass), and dead-zone.
* Success prediction: Add an extra action that predicts task success.

These tasks share the same task configurations and control options. You can switch between them by specifying the task name.

* |forge-peg-link|: Peg insertion with the Franka arm
* |forge-gear-link|: Gear meshing with the Franka arm
* |forge-nut-link|: Nut-Bolt fastening with the Franka arm

.. table::
    :widths: 33 37 30

    +--------------------+-------------------------+-----------------------------------------------------------------------------+
    | World              | Environment ID          | Description                                                                 |
    +====================+=========================+=============================================================================+
    | |forge-peg|        | |forge-peg-link|        | Insert peg into the socket with the Franka robot                            |
    +--------------------+-------------------------+-----------------------------------------------------------------------------+
    | |forge-gear|       | |forge-gear-link|       | Insert and mesh gear into the base with other gears, using the Franka robot |
    +--------------------+-------------------------+-----------------------------------------------------------------------------+
    | |forge-nut|        | |forge-nut-link|        | Thread the nut onto the first 2 threads of the bolt, using the Franka robot |
    +--------------------+-------------------------+-----------------------------------------------------------------------------+

.. |forge-peg| image:: ../_static/tasks/factory/peg_insert.jpg
.. |forge-gear| image:: ../_static/tasks/factory/gear_mesh.jpg
.. |forge-nut| image:: ../_static/tasks/factory/nut_thread.jpg

.. |forge-peg-link| replace:: `Isaac-Forge-PegInsert-Direct-v0 <https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/direct/forge/forge_env_cfg.py>`__
.. |forge-gear-link| replace:: `Isaac-Forge-GearMesh-Direct-v0 <https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/direct/forge/forge_env_cfg.py>`__
.. |forge-nut-link| replace:: `Isaac-Forge-NutThread-Direct-v0 <https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/direct/forge/forge_env_cfg.py>`__


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
    | |velocity-flat-digit|        | |velocity-flat-digit-link|                   | Track a velocity command on flat terrain with the Agility Digit robot        |
    +------------------------------+----------------------------------------------+------------------------------------------------------------------------------+
    | |velocity-rough-digit|       | |velocity-rough-digit-link|                  | Track a velocity command on rough terrain with the Agility Digit robot       |
    +------------------------------+----------------------------------------------+------------------------------------------------------------------------------+
    | |tracking-loco-manip-digit|  | |tracking-loco-manip-digit-link|             | Track a root velocity and hand pose command with the Agility Digit robot     |
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

.. |velocity-flat-digit-link| replace:: `Isaac-Velocity-Flat-Digit-v0 <https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/digit/flat_env_cfg.py>`__
.. |velocity-rough-digit-link| replace:: `Isaac-Velocity-Rough-Digit-v0 <https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/digit/rough_env_cfg.py>`__
.. |tracking-loco-manip-digit-link| replace:: `Isaac-Tracking-Flat-Digit-v0 <https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/tracking/config/digit/loco_manip_env_cfg.py>`__

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
.. |velocity-flat-digit| image:: ../_static/tasks/locomotion/agility_digit_flat.jpg
.. |velocity-rough-digit| image:: ../_static/tasks/locomotion/agility_digit_rough.jpg
.. |tracking-loco-manip-digit| image:: ../_static/tasks/locomotion/agility_digit_loco_manip.jpg

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

Spaces showcase
~~~~~~~~~~~~~~~

The |cartpole_showcase| folder contains showcase tasks (based on the *Cartpole* and *Cartpole-Camera* Direct tasks)
for the definition/use of the various Gymnasium observation and action spaces supported in Isaac Lab.

.. |cartpole_showcase| replace:: `cartpole_showcase <https://github.com/isaac-sim/IsaacLab/tree/main/source/isaaclab_tasks/isaaclab_tasks/direct/cartpole_showcase>`__

.. note::

    Currently, only Isaac Lab's Direct workflow supports the definition of observation and action spaces other than ``Box``.
    See Direct workflow's :py:obj:`~isaaclab.envs.DirectRLEnvCfg.observation_space` / :py:obj:`~isaaclab.envs.DirectRLEnvCfg.action_space`
    documentation for more details.

The following tables summarize the different pairs of showcased spaces for the *Cartpole* and *Cartpole-Camera* tasks.
Replace ``<OBSERVATION>`` and ``<ACTION>`` with the observation and action spaces to be explored in the task names for training and evaluation.

.. raw:: html

    <table class="showcase-table">
    <caption>
      <p>Showcase spaces for the <strong>Cartpole</strong> task</p>
      <p><code>Isaac-Cartpole-Showcase-&lt;OBSERVATION&gt;-&lt;ACTION&gt;-Direct-v0</code></p>
    </caption>
    <tbody>
      <tr>
        <td colspan="2" rowspan="2"></td>
        <td colspan="5" class="center">action space</td>
      </tr>
      <tr>
        <td><strong>&nbsp;Box</strong></td>
        <td><strong>&nbsp;Discrete</strong></td>
        <td><strong>&nbsp;MultiDiscrete</strong></td>
      </tr>
      <tr>
        <td rowspan="5" class="rot90 center"><p>observation</p><p>space</p></td>
        <td><strong>&nbsp;Box</strong></td>
        <td class="center">x</td>
        <td class="center">x</td>
        <td class="center">x</td>
      </tr>
      <tr>
        <td><strong>&nbsp;Discrete</strong></td>
        <td class="center">x</td>
        <td class="center">x</td>
        <td class="center">x</td>
      </tr>
      <tr>
        <td><strong>&nbsp;MultiDiscrete</strong></td>
        <td class="center">x</td>
        <td class="center">x</td>
        <td class="center">x</td>
      </tr>
      <tr>
        <td><strong>&nbsp;Dict</strong></td>
        <td class="center">x</td>
        <td class="center">x</td>
        <td class="center">x</td>
      </tr>
      <tr>
        <td><strong>&nbsp;Tuple</strong></td>
        <td class="center">x</td>
        <td class="center">x</td>
        <td class="center">x</td>
      </tr>
    </tbody>
    </table>
    <br>
    <table class="showcase-table">
    <caption>
        <p>Showcase spaces for the <strong>Cartpole-Camera</strong> task</p>
        <p><code>Isaac-Cartpole-Camera-Showcase-&lt;OBSERVATION&gt;-&lt;ACTION&gt;-Direct-v0</code></p>
    </caption>
    <tbody>
      <tr>
        <td colspan="2" rowspan="2"></td>
        <td colspan="5" class="center">action space</td>
      </tr>
      <tr>
        <td><strong>&nbsp;Box</strong></td>
        <td><strong>&nbsp;Discrete</strong></td>
        <td><strong>&nbsp;MultiDiscrete</strong></td>
      </tr>
      <tr>
        <td rowspan="5" class="rot90 center"><p>observation</p><p>space</p></td>
        <td><strong>&nbsp;Box</strong></td>
        <td class="center">x</td>
        <td class="center">x</td>
        <td class="center">x</td>
      </tr>
      <tr>
        <td><strong>&nbsp;Discrete</strong></td>
        <td class="center">-</td>
        <td class="center">-</td>
        <td class="center">-</td>
      </tr>
      <tr>
        <td><strong>&nbsp;MultiDiscrete</strong></td>
        <td class="center">-</td>
        <td class="center">-</td>
        <td class="center">-</td>
      </tr>
      <tr>
        <td><strong>&nbsp;Dict</strong></td>
        <td class="center">x</td>
        <td class="center">x</td>
        <td class="center">x</td>
      </tr>
      <tr>
        <td><strong>&nbsp;Tuple</strong></td>
        <td class="center">x</td>
        <td class="center">x</td>
        <td class="center">x</td>
      </tr>
    </tbody></table>

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

For environments that have a different task name listed under ``Inference Task Name``, please use the Inference Task Name
provided when running ``play.py`` or any inferencing workflows. These tasks provide more suitable configurations for
inferencing, including reading from an already trained checkpoint and disabling runtime perturbations used for training.

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
    * - Isaac-Cartpole-Camera-Showcase-Box-Box-Direct-v0 (Requires running with ``--enable_cameras``)
      -
      - Direct
      - **skrl** (PPO)
    * - Isaac-Cartpole-Camera-Showcase-Box-Discrete-Direct-v0 (Requires running with ``--enable_cameras``)
      -
      - Direct
      - **skrl** (PPO)
    * - Isaac-Cartpole-Camera-Showcase-Box-MultiDiscrete-Direct-v0 (Requires running with ``--enable_cameras``)
      -
      - Direct
      - **skrl** (PPO)
    * - Isaac-Cartpole-Camera-Showcase-Dict-Box-Direct-v0 (Requires running with ``--enable_cameras``)
      -
      - Direct
      - **skrl** (PPO)
    * - Isaac-Cartpole-Camera-Showcase-Dict-Discrete-Direct-v0 (Requires running with ``--enable_cameras``)
      -
      - Direct
      - **skrl** (PPO)
    * - Isaac-Cartpole-Camera-Showcase-Dict-MultiDiscrete-Direct-v0 (Requires running with ``--enable_cameras``)
      -
      - Direct
      - **skrl** (PPO)
    * - Isaac-Cartpole-Camera-Showcase-Tuple-Box-Direct-v0 (Requires running with ``--enable_cameras``)
      -
      - Direct
      - **skrl** (PPO)
    * - Isaac-Cartpole-Camera-Showcase-Tuple-Discrete-Direct-v0 (Requires running with ``--enable_cameras``)
      -
      - Direct
      - **skrl** (PPO)
    * - Isaac-Cartpole-Camera-Showcase-Tuple-MultiDiscrete-Direct-v0 (Requires running with ``--enable_cameras``)
      -
      - Direct
      - **skrl** (PPO)
    * - Isaac-Cartpole-Depth-Camera-Direct-v0 (Requires running with ``--enable_cameras``)
      -
      - Direct
      - **rl_games** (PPO), **skrl** (PPO)
    * - Isaac-Cartpole-Depth-v0 (Requires running with ``--enable_cameras``)
      -
      - Manager Based
      - **rl_games** (PPO)
    * - Isaac-Cartpole-Direct-v0
      -
      - Direct
      - **rl_games** (PPO), **rsl_rl** (PPO), **skrl** (PPO), **sb3** (PPO)
    * - Isaac-Cartpole-RGB-Camera-Direct-v0 (Requires running with ``--enable_cameras``)
      -
      - Direct
      - **rl_games** (PPO), **skrl** (PPO)
    * - Isaac-Cartpole-RGB-ResNet18-v0 (Requires running with ``--enable_cameras``)
      -
      - Manager Based
      - **rl_games** (PPO)
    * - Isaac-Cartpole-RGB-TheiaTiny-v0 (Requires running with ``--enable_cameras``)
      -
      - Manager Based
      - **rl_games** (PPO)
    * - Isaac-Cartpole-RGB-v0 (Requires running with ``--enable_cameras``)
      -
      - Manager Based
      - **rl_games** (PPO)
    * - Isaac-Cartpole-Showcase-Box-Box-Direct-v0
      -
      - Direct
      - **skrl** (PPO)
    * - Isaac-Cartpole-Showcase-Box-Discrete-Direct-v0
      -
      - Direct
      - **skrl** (PPO)
    * - Isaac-Cartpole-Showcase-Box-MultiDiscrete-Direct-v0
      -
      - Direct
      - **skrl** (PPO)
    * - Isaac-Cartpole-Showcase-Dict-Box-Direct-v0
      -
      - Direct
      - **skrl** (PPO)
    * - Isaac-Cartpole-Showcase-Dict-Discrete-Direct-v0
      -
      - Direct
      - **skrl** (PPO)
    * - Isaac-Cartpole-Showcase-Dict-MultiDiscrete-Direct-v0
      -
      - Direct
      - **skrl** (PPO)
    * - Isaac-Cartpole-Showcase-Discrete-Box-Direct-v0
      -
      - Direct
      - **skrl** (PPO)
    * - Isaac-Cartpole-Showcase-Discrete-Discrete-Direct-v0
      -
      - Direct
      - **skrl** (PPO)
    * - Isaac-Cartpole-Showcase-Discrete-MultiDiscrete-Direct-v0
      -
      - Direct
      - **skrl** (PPO)
    * - Isaac-Cartpole-Showcase-MultiDiscrete-Box-Direct-v0
      -
      - Direct
      - **skrl** (PPO)
    * - Isaac-Cartpole-Showcase-MultiDiscrete-Discrete-Direct-v0
      -
      - Direct
      - **skrl** (PPO)
    * - Isaac-Cartpole-Showcase-MultiDiscrete-MultiDiscrete-Direct-v0
      -
      - Direct
      - **skrl** (PPO)
    * - Isaac-Cartpole-Showcase-Tuple-Box-Direct-v0
      -
      - Direct
      - **skrl** (PPO)
    * - Isaac-Cartpole-Showcase-Tuple-Discrete-Direct-v0
      -
      - Direct
      - **skrl** (PPO)
    * - Isaac-Cartpole-Showcase-Tuple-MultiDiscrete-Direct-v0
      -
      - Direct
      - **skrl** (PPO)
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
    * - Isaac-AutoMate-Assembly-Direct-v0
      -
      - Direct
      - **rl_games** (PPO)
    * - Isaac-AutoMate-Disassembly-Direct-v0
      -
      - Direct
      -
    * - Isaac-Forge-GearMesh-Direct-v0
      -
      - Direct
      - **rl_games** (PPO)
    * - Isaac-Forge-NutThread-Direct-v0
      -
      - Direct
      - **rl_games** (PPO)
    * - Isaac-Forge-PegInsert-Direct-v0
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
    * - Isaac-Tracking-LocoManip-Digit-v0
      - Isaac-Tracking-LocoManip-Digit-Play-v0
      - Manager Based
      - **rsl_rl** (PPO)
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
    * - Isaac-Repose-Cube-Shadow-Vision-Direct-v0 (Requires running with ``--enable_cameras``)
      - Isaac-Repose-Cube-Shadow-Vision-Direct-Play-v0 (Requires running with ``--enable_cameras``)
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
    * - Isaac-Velocity-Flat-Digit-v0
      - Isaac-Velocity-Flat-Digit-Play-v0
      - Manager Based
      - **rsl_rl** (PPO)
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
      - **rsl_rl** (PPO), **skrl** (PPO), **sb3** (PPO)
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
    * - Isaac-Velocity-Rough-Digit-v0
      - Isaac-Velocity-Rough-Digit-Play-v0
      - Manager Based
      - **rsl_rl** (PPO)
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
      - **rsl_rl** (PPO), **skrl** (PPO), **sb3** (PPO)
    * - Isaac-Velocity-Rough-Unitree-Go1-v0
      - Isaac-Velocity-Rough-Unitree-Go1-Play-v0
      - Manager Based
      - **rsl_rl** (PPO), **skrl** (PPO)
    * - Isaac-Velocity-Rough-Unitree-Go2-v0
      - Isaac-Velocity-Rough-Unitree-Go2-Play-v0
      - Manager Based
      - **rsl_rl** (PPO), **skrl** (PPO)
