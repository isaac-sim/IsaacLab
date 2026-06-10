.. _environments:

Available Environments
======================

The following lists comprises of all the RL and IL tasks implementations that are available in Isaac Lab.
While we try to keep this list up-to-date, you can always get the latest list of environments by
running the following command:

.. tab-set::
   :sync-group: os

   .. tab-item:: :icon:`fa-brands fa-linux` Linux
      :sync: linux

      .. note::
         Use ``--keyword <search_term>`` (optional) to filter environments by keyword.

      .. code:: bash

         ./isaaclab.sh -p scripts/environments/list_envs.py --keyword <search_term>

   .. tab-item:: :icon:`fa-brands fa-windows` Windows
      :sync: windows

      .. note::
         Use ``--keyword <search_term>`` (optional) to filter environments by keyword.

      .. code:: batch

         isaaclab.bat -p scripts\environments\list_envs.py --keyword <search_term>


To also see the available presets for each environment, pass ``--show_presets``:

.. tab-set::
   :sync-group: os

   .. tab-item:: :icon:`fa-brands fa-linux` Linux
      :sync: linux

      .. code:: bash

         ./isaaclab.sh -p scripts/environments/list_envs.py --show_presets

   .. tab-item:: :icon:`fa-brands fa-windows` Windows
      :sync: windows

      .. code:: batch

         isaaclab.bat -p scripts\environments\list_envs.py --show_presets


The **RL Library** and **Presets** columns in the :ref:`comprehensive-environment-list` below are
generated automatically from the Gym registry.

We are actively working on adding more environments to the list. If you have any environments that
you would like to add to Isaac Lab, please feel free to open a pull request!


Preset Selectors
----------------

Many environments support multiple physics backends, rendering backends, and observation
modes. The **Presets** column in each table below is divided into three labeled groups:

* **physics=** — physics-backend name passed as ``physics=NAME``
  (e.g. ``physx``, ``newton_mjwarp``, ``newton_kamino``, ``ovphysx``,
  ``newton_mjwarp_vbd``)
* **renderer=** — renderer-backend name passed as ``renderer=NAME``
  (e.g. ``isaacsim_rtx_renderer``, ``newton_renderer``, ``ovrtx_renderer``)
* **presets=** — environment-specific (domain) preset name passed as
  ``presets=NAME[,NAME,...]``
  (e.g. ``rgb``, ``depth``, ``single_camera``, ``duo_camera``)

Pass ``--task=<task-name> --help`` to a training script to see all available
preset names grouped by selector type at the command line, or run
``./isaaclab.sh -p scripts/environments/list_envs.py --show_presets``
to list presets for every registered environment.

See :doc:`/source/features/hydra` for the full preset system documentation,
including all available backend names and how the typed selectors work.


Single-agent
------------

Classic
~~~~~~~

Classic environments that are based on IsaacGymEnvs implementation of MuJoCo-style environments.

.. table::
    :widths: 25 30 25 20

    +------------------+-----------------------------+-------------------------------------------------------------------------+------------------------------+
    | World            | Environment ID              | Description                                                             | Presets                      |
    +==================+=============================+=========================================================================+==============================+
    | |humanoid|       | |humanoid-link|             | Move towards a direction with the MuJoCo humanoid robot                 | **physics=**                 |
    |                  |                             |                                                                         | ``physx``,                   |
    |                  | |humanoid-direct-link|      |                                                                         | ``newton_mjwarp``,           |
    |                  |                             |                                                                         | ``ovphysx`` (direct only)    |
    +------------------+-----------------------------+-------------------------------------------------------------------------+------------------------------+
    | |ant|            | |ant-link|                  | Move towards a direction with the MuJoCo ant robot                      | **physics=**                 |
    |                  |                             |                                                                         | ``physx``,                   |
    |                  | |ant-direct-link|           |                                                                         | ``newton_mjwarp``,           |
    |                  |                             |                                                                         | ``newton_kamino``,           |
    |                  |                             |                                                                         | ``ovphysx`` (direct only)    |
    +------------------+-----------------------------+-------------------------------------------------------------------------+------------------------------+
    | |cartpole|       | |cartpole-link|             | Move the cart to keep the pole upwards in the classic cartpole control  | **physics=**                 |
    |                  |                             |                                                                         | ``physx``,                   |
    |                  | |cartpole-direct-link|      |                                                                         | ``newton_mjwarp``,           |
    |                  |                             |                                                                         | ``newton_kamino``,           |
    |                  |                             |                                                                         | ``ovphysx`` (direct only)    |
    +------------------+-----------------------------+-------------------------------------------------------------------------+------------------------------+
    | |cartpole|       | |cartpole-camera-presets|   | Move the cart to keep the pole upwards in the classic cartpole control  | **physics=**                 |
    |                  |                             | and perceptive inputs. Select data type via ``presets=``. Requires      | ``physx``,                   |
    |                  |                             | running with ``--enable_cameras``.                                      | ``newton_mjwarp``,           |
    |                  |                             |                                                                         | ``ovphysx`` (direct only)    |
    |                  |                             |                                                                         | **renderer=**                |
    |                  |                             |                                                                         | ``isaacsim_rtx_renderer``,   |
    |                  |                             |                                                                         | ``newton_renderer``,         |
    |                  |                             |                                                                         | ``ovrtx_renderer``           |
    |                  |                             |                                                                         | **presets=** ``rgb``,        |
    |                  |                             |                                                                         | ``depth``, ``albedo``,       |
    |                  |                             |                                                                         | ``semantic_segmentation``,   |
    |                  |                             |                                                                         | ``simple_shading_*``         |
    +------------------+-----------------------------+-------------------------------------------------------------------------+------------------------------+
    | |cartpole|       | |cartpole-camera-link|      | Move the cart to keep the pole upwards in the classic cartpole control  | **physics=**                 |
    |                  |                             | from raw RGB/depth observations or features extracted by pre-trained    | ``physx``,                   |
    |                  |                             | frozen vision encoders. Select pipeline via ``presets=``. Requires      | ``newton_mjwarp``            |
    |                  |                             | running with ``--enable_cameras``.                                      | **presets=** ``rgb``,        |
    |                  |                             |                                                                         | ``depth``, ``resnet18``,     |
    |                  |                             |                                                                         | ``theia_tiny``               |
    +------------------+-----------------------------+-------------------------------------------------------------------------+------------------------------+

.. |humanoid| image:: ../_static/tasks/classic/humanoid.jpg
.. |ant| image:: ../_static/tasks/classic/ant.jpg
.. |cartpole| image:: ../_static/tasks/classic/cartpole.jpg

.. |humanoid-link| replace:: `Isaac-Humanoid <../../../source/isaaclab_tasks/isaaclab_tasks/core/locomotion/humanoid/humanoid_manager_env_cfg.py>`__
.. |ant-link| replace:: `Isaac-Ant <../../../source/isaaclab_tasks/isaaclab_tasks/core/locomotion/ant/ant_manager_env_cfg.py>`__
.. |cartpole-link| replace:: `Isaac-Cartpole <../../../source/isaaclab_tasks/isaaclab_tasks/core/cartpole/cartpole_manager_env_cfg.py>`__
.. |cartpole-camera-presets| replace:: `Isaac-Cartpole-Camera-Direct <../../../source/isaaclab_tasks/isaaclab_tasks/core/cartpole/cartpole_direct_camera_env_cfg.py>`__
.. |cartpole-camera-link| replace:: `Isaac-Cartpole-Camera <../../../source/isaaclab_tasks/isaaclab_tasks/core/cartpole/cartpole_manager_camera_env_cfg.py>`__


.. |humanoid-direct-link| replace:: `Isaac-Humanoid-Direct <../../../source/isaaclab_tasks/isaaclab_tasks/core/locomotion/humanoid/humanoid_direct_env.py>`__
.. |ant-direct-link| replace:: `Isaac-Ant-Direct <../../../source/isaaclab_tasks/isaaclab_tasks/core/locomotion/ant/ant_direct_env.py>`__
.. |cartpole-direct-link| replace:: `Isaac-Cartpole-Direct <../../../source/isaaclab_tasks/isaaclab_tasks/core/cartpole/cartpole_direct_env.py>`__

Manipulation
~~~~~~~~~~~~

Environments based on fixed-arm manipulation tasks.

For many of these tasks, we include configurations with different arm action spaces. For example,
for the lift-cube environment:

* |lift-cube-link|: Franka arm with joint position control
* |lift-cube-ik-abs-link|: Franka arm with absolute IK control
* |lift-cube-ik-rel-link|: Franka arm with relative IK control

.. table::
    :widths: 25 30 25 20

    +-------------------------+------------------------------+-----------------------------------------------------------------------------+------------------------------+
    | World                   | Environment ID               | Description                                                                 | Presets                      |
    +=========================+==============================+=============================================================================+==============================+
    | |reach-franka|          | |reach-franka-link|          | Move the end-effector to a sampled target pose with the Franka robot        | **physics=** ``physx``,      |
    |                         |                              |                                                                             | ``newton_mjwarp``            |
    +-------------------------+------------------------------+-----------------------------------------------------------------------------+------------------------------+
    | |reach-ur10|            | |reach-ur10-link|            | Move the end-effector to a sampled target pose with the UR10 robot          | **physics=** ``physx``,      |
    |                         |                              |                                                                             | ``newton_mjwarp``            |
    +-------------------------+------------------------------+-----------------------------------------------------------------------------+------------------------------+
    | |deploy-reach-ur10e|    | |deploy-reach-ur10e-link|    | Move the end-effector to a sampled target pose with the UR10e robot         |                              |
    |                         |                              | This policy has been deployed to a real robot                               |                              |
    +-------------------------+------------------------------+-----------------------------------------------------------------------------+------------------------------+
    | |lift-cube|             | |lift-cube-link|             | Pick a cube and bring it to a sampled target position with the Franka robot |                              |
    +-------------------------+------------------------------+-----------------------------------------------------------------------------+------------------------------+
    | |lift-soft-franka|      | |lift-soft-franka-link|      | Pick a deformable soft body and bring it to a sampled target position with  | **physics=** ``physx``,      |
    |                         |                              | the Franka robot                                                            | ``newton_mjwarp_vbd``        |
    +-------------------------+------------------------------+-----------------------------------------------------------------------------+------------------------------+
    | |stack-cube|            | |stack-cube-link|            | Stack three cubes (bottom to top: blue, red, green) with the Franka robot.  |                              |
    |                         |                              | Blueprint env used for the NVIDIA Isaac GR00T blueprint for synthetic       |                              |
    |                         | |stack-cube-bp-link|         | manipulation motion generation                                              |                              |
    +-------------------------+------------------------------+-----------------------------------------------------------------------------+------------------------------+
    | |surface-gripper|       | |long-suction-link|          | Stack three cubes (bottom to top: blue, red, green)                         |                              |
    |                         |                              | with the UR10 arm and long surface gripper                                  |                              |
    |                         | |short-suction-link|         | or short surface gripper (cpu only).                                        |                              |
    +-------------------------+------------------------------+-----------------------------------------------------------------------------+------------------------------+
    | |cabi-franka|           | |cabi-franka-link|           | Grasp the handle of a cabinet's drawer and open it with the Franka robot    | **physics=** ``physx``,      |
    |                         |                              |                                                                             | ``newton_mjwarp``            |
    |                         | |franka-direct-link|         |                                                                             |                              |
    +-------------------------+------------------------------+-----------------------------------------------------------------------------+------------------------------+
    | |cube-allegro|          | |cube-allegro-link|          | In-hand reorientation of a cube using Allegro hand                          | **physics=** ``physx``,      |
    |                         |                              |                                                                             | ``newton_mjwarp``,           |
    |                         | |allegro-direct-link|        |                                                                             | ``ovphysx`` (direct only)    |
    +-------------------------+------------------------------+-----------------------------------------------------------------------------+------------------------------+
    | |cube-shadow|           | |cube-shadow-link|           | In-hand reorientation of a cube using Shadow hand                           | **physics=** ``physx``,      |
    |                         |                              |                                                                             | ``newton_mjwarp``            |
    |                         | |cube-shadow-ff-link|        |                                                                             |                              |
    |                         |                              |                                                                             |                              |
    |                         | |cube-shadow-lstm-link|      |                                                                             |                              |
    +-------------------------+------------------------------+-----------------------------------------------------------------------------+------------------------------+
    | |cube-shadow|           | |cube-shadow-vis-link|       | In-hand reorientation of a cube using Shadow hand using perceptive inputs.  | **physics=** ``physx``,      |
    |                         |                              | Requires running with ``--enable_cameras``.                                 | ``newton_mjwarp``            |
    |                         |                              |                                                                             | **renderer=**                |
    |                         |                              |                                                                             | ``isaacsim_rtx_renderer``,   |
    |                         |                              |                                                                             | ``newton_renderer``,         |
    |                         |                              |                                                                             | ``ovrtx_renderer``           |
    |                         |                              |                                                                             | **presets=** ``rgb``,        |
    |                         |                              |                                                                             | ``depth``, ``albedo``,       |
    |                         |                              |                                                                             | ``full``,                    |
    |                         |                              |                                                                             | ``semantic_segmentation``,   |
    |                         |                              |                                                                             | ``simple_shading_*``         |
    +-------------------------+------------------------------+-----------------------------------------------------------------------------+------------------------------+
    | |gr1_pick_place|        | |gr1_pick_place-link|        | Pick up and place an object in a basket with a GR-1 humanoid robot          | **physics=** ``physx``       |
    +-------------------------+------------------------------+-----------------------------------------------------------------------------+------------------------------+
    | |gr1_pp_waist|          | |gr1_pp_waist-link|          | Pick up and place an object in a basket with a GR-1 humanoid robot          | **physics=** ``physx``       |
    |                         |                              | with waist degrees-of-freedom enables that provides a wider reach space.    |                              |
    +-------------------------+------------------------------+-----------------------------------------------------------------------------+------------------------------+
    | |g1_pick_place|         | |g1_pick_place-link|         | Pick up and place an object in a basket with a Unitree G1 humanoid robot    | **physics=** ``physx``       |
    +-------------------------+------------------------------+-----------------------------------------------------------------------------+------------------------------+
    | |g1_pick_place_fixed|   | |g1_pick_place_fixed-link|   | Pick up and place an object in a basket with a Unitree G1 humanoid robot    | **physics=** ``physx``       |
    |                         |                              | with three-fingered hands. Robot is set up with the base fixed in place.    |                              |
    +-------------------------+------------------------------+-----------------------------------------------------------------------------+------------------------------+
    | |g1_pick_place_lm|      | |g1_pick_place_lm-link|      | Pick up and place an object in a basket with a Unitree G1 humanoid robot    | **physics=** ``physx``       |
    |                         |                              | with three-fingered hands and in-place locomanipulation capabilities        |                              |
    |                         |                              | enabled (i.e. Robot lower body balances in-place while upper body is        |                              |
    |                         |                              | controlled via Inverse Kinematics).                                         |                              |
    +-------------------------+------------------------------+-----------------------------------------------------------------------------+------------------------------+
    | |kuka-allegro-lift|     | |kuka-allegro-lift-link|     | Pick up a primitive shape on the table and lift it to target position.      | **physics=** ``physx``,      |
    |                         |                              | Supports state, single-camera, and dual-camera observation modes via        | ``newton_mjwarp``            |
    |                         |                              | ``presets=single_camera`` / ``presets=duo_camera`` (see RL table below).    | **renderer=**                |
    |                         |                              |                                                                             | ``isaacsim_rtx_renderer``,   |
    |                         |                              |                                                                             | ``newton_renderer``,         |
    |                         |                              |                                                                             | ``ovrtx_renderer``           |
    |                         |                              |                                                                             | **presets=** ``state``,      |
    |                         |                              |                                                                             | ``single_camera``,           |
    |                         |                              |                                                                             | ``duo_camera``,              |
    |                         |                              |                                                                             | ``rgb{64,128,256}``,         |
    |                         |                              |                                                                             | ``depth/albedo{..}``,        |
    |                         |                              |                                                                             | ``simple_shading_*{..}``     |
    +-------------------------+------------------------------+-----------------------------------------------------------------------------+------------------------------+
    | |kuka-allegro-reorient| | |kuka-allegro-reorient-link| | Pick up a primitive shape on the table and orient it to target pose.        | **physics=** ``physx``,      |
    |                         |                              | Supports state, single-camera, and dual-camera observation modes via        | ``newton_mjwarp``            |
    |                         |                              | ``presets=single_camera`` / ``presets=duo_camera`` (see RL table below).    | **renderer=**                |
    |                         |                              |                                                                             | ``isaacsim_rtx_renderer``,   |
    |                         |                              |                                                                             | ``newton_renderer``,         |
    |                         |                              |                                                                             | ``ovrtx_renderer``           |
    |                         |                              |                                                                             | **presets=** ``state``,      |
    |                         |                              |                                                                             | ``single_camera``,           |
    |                         |                              |                                                                             | ``duo_camera``,              |
    |                         |                              |                                                                             | ``rgb{64,128,256}``,         |
    |                         |                              |                                                                             | ``depth/albedo{..}``,        |
    |                         |                              |                                                                             | ``simple_shading_*{..}``     |
    +-------------------------+------------------------------+-----------------------------------------------------------------------------+------------------------------+
    | |galbot_stack|          | |galbot_stack-link|          | Stack three cubes (bottom to top: blue, red, green) with the left arm of    | **physics=** ``physx``       |
    |                         |                              | a Galbot humanoid robot                                                     |                              |
    +-------------------------+------------------------------+-----------------------------------------------------------------------------+------------------------------+
    | |agibot_place_mug|      | |agibot_place_mug-link|      | Pick up and place a mug upright with a Agibot A2D humanoid robot            |                              |
    +-------------------------+------------------------------+-----------------------------------------------------------------------------+------------------------------+
    | |agibot_place_toy|      | |agibot_place_toy-link|      | Pick up and place an object in a box with a Agibot A2D humanoid robot       |                              |
    +-------------------------+------------------------------+-----------------------------------------------------------------------------+------------------------------+
    | |reach_openarm_bi|      | |reach_openarm_bi-link|      | Move the end-effector to sampled target poses with the OpenArm robot        |                              |
    +-------------------------+------------------------------+-----------------------------------------------------------------------------+------------------------------+
    | |reach_openarm_uni|     | |reach_openarm_uni-link|     | Move the end-effector to a sampled target pose with the OpenArm robot       |                              |
    +-------------------------+------------------------------+-----------------------------------------------------------------------------+------------------------------+
    | |lift_openarm_uni|      | |lift_openarm_uni-link|      | Pick a cube and bring it to a sampled target position with the OpenArm robot|                              |
    +-------------------------+------------------------------+-----------------------------------------------------------------------------+------------------------------+
    | |cabi_openarm_uni|      | |cabi_openarm_uni-link|      | Grasp the handle of a cabinet's drawer and open it with the OpenArm robot   |                              |
    +-------------------------+------------------------------+-----------------------------------------------------------------------------+------------------------------+
    | |g1_assemble_trocar|    | |g1_assemble_trocar-link|    | Assemble trocar with a Unitree G1 humanoid robot with Dex3 hands            |                              |
    +-------------------------+------------------------------+-----------------------------------------------------------------------------+------------------------------+

.. |reach-franka| image:: ../_static/tasks/manipulation/franka_reach.jpg
.. |reach-ur10| image:: ../_static/tasks/manipulation/ur10_reach.jpg
.. |deploy-reach-ur10e| image:: ../_static/tasks/manipulation/ur10e_reach.jpg
.. |lift-cube| image:: ../_static/tasks/manipulation/franka_lift.jpg
.. |lift-soft-franka| image:: ../_static/newton/franka-mjwarp-vbd-coupling.png
.. |cabi-franka| image:: ../_static/tasks/manipulation/franka_open_drawer.jpg
.. |cube-allegro| image:: ../_static/tasks/manipulation/allegro_cube.jpg
.. |cube-shadow| image:: ../_static/tasks/manipulation/shadow_cube.jpg
.. |stack-cube| image:: ../_static/tasks/manipulation/franka_stack.jpg
.. |gr1_pick_place| image:: ../_static/tasks/manipulation/gr-1_pick_place.jpg
.. |g1_pick_place| image:: ../_static/tasks/manipulation/g1_pick_place.jpg
.. |g1_pick_place_fixed| image:: ../_static/tasks/manipulation/g1_pick_place_fixed_base.jpg
.. |g1_pick_place_lm| image:: ../_static/tasks/manipulation/g1_pick_place_locomanipulation.jpg
.. |surface-gripper| image:: ../_static/tasks/manipulation/ur10_stack_surface_gripper.jpg
.. |gr1_pp_waist| image:: ../_static/tasks/manipulation/gr-1_pick_place_waist.jpg
.. |galbot_stack| image:: ../_static/tasks/manipulation/galbot_stack_cube.jpg
.. |agibot_place_mug| image:: ../_static/tasks/manipulation/agibot_place_mug.jpg
.. |agibot_place_toy| image:: ../_static/tasks/manipulation/agibot_place_toy.jpg
.. |kuka-allegro-lift| image:: ../_static/tasks/manipulation/kuka_allegro_lift.jpg
.. |kuka-allegro-reorient| image:: ../_static/tasks/manipulation/kuka_allegro_reorient.jpg
.. |reach_openarm_bi| image:: ../_static/tasks/manipulation/openarm_bi_reach.jpg
.. |reach_openarm_uni| image:: ../_static/tasks/manipulation/openarm_uni_reach.jpg
.. |lift_openarm_uni| image:: ../_static/tasks/manipulation/openarm_uni_lift.jpg
.. |cabi_openarm_uni| image:: ../_static/tasks/manipulation/openarm_uni_open_drawer.jpg
.. |g1_assemble_trocar| image:: ../_static/tasks/manipulation/g1_assemble_trocar.jpg

.. |reach-franka-link| replace:: `Isaac-Reach-Franka <../../../source/isaaclab_tasks/isaaclab_tasks/core/reach/config/franka/joint_pos_env_cfg.py>`__
.. |reach-ur10-link| replace:: `Isaac-Reach-UR10 <../../../source/isaaclab_tasks/isaaclab_tasks/core/reach/config/ur_10/joint_pos_env_cfg.py>`__
.. |deploy-reach-ur10e-link| replace:: `Isaac-Deploy-Reach-UR10e-v0 <../../../source/isaaclab_tasks/isaaclab_tasks/contrib/deploy/reach/config/ur_10e/joint_pos_env_cfg.py>`__
.. |lift-cube-link| replace:: `Isaac-Lift-Cube-Franka <../../../source/isaaclab_tasks/isaaclab_tasks/core/lift/config/franka/joint_pos_env_cfg.py>`__
.. |lift-cube-ik-abs-link| replace:: `Isaac-Lift-Cube-Franka-IK-Abs-v0 <../../../source/isaaclab_tasks/isaaclab_tasks/contrib/lift/config/franka/ik_abs_env_cfg.py>`__
.. |lift-cube-ik-rel-link| replace:: `Isaac-Lift-Cube-Franka-IK-Rel-v0 <../../../source/isaaclab_tasks/isaaclab_tasks/contrib/lift/config/franka/ik_rel_env_cfg.py>`__
.. |lift-soft-franka-link| replace:: `Isaac-Lift-Soft-Franka <../../../source/isaaclab_tasks/isaaclab_tasks/core/lift/config/franka_soft/franka_soft_env_cfg.py>`__
.. |cabi-franka-link| replace:: `Isaac-Open-Drawer-Franka-v0 <../../../source/isaaclab_tasks/isaaclab_tasks/core/cabinet/config/franka/joint_pos_env_cfg.py>`__
.. |franka-direct-link| replace:: `Isaac-Franka-Cabinet-Direct-v0 <../../../source/isaaclab_tasks/isaaclab_tasks/core/franka_cabinet/franka_cabinet_env.py>`__
.. |cube-allegro-link| replace:: `Isaac-Repose-Cube-Allegro-v0 <../../../source/isaaclab_tasks/isaaclab_tasks/core/inhand/config/allegro_hand/allegro_env_cfg.py>`__
.. |allegro-direct-link| replace:: `Isaac-Repose-Cube-Allegro-Direct-v0 <../../../source/isaaclab_tasks/isaaclab_tasks/core/allegro_hand/allegro_hand_env_cfg.py>`__
.. |stack-cube-link| replace:: `Isaac-Stack-Cube-Franka-v0 <../../../source/isaaclab_tasks/isaaclab_tasks/contrib/stack/config/franka/stack_joint_pos_env_cfg.py>`__
.. |stack-cube-bp-link| replace:: `Isaac-Stack-Cube-Franka-IK-Rel-Blueprint-v0 <../../../source/isaaclab_tasks/isaaclab_tasks/contrib/stack/config/franka/stack_ik_rel_blueprint_env_cfg.py>`__
.. |gr1_pick_place-link| replace:: `Isaac-PickPlace-GR1T2-Abs-v0 <../../../source/isaaclab_tasks/isaaclab_tasks/contrib/pick_place/pickplace_gr1t2_env_cfg.py>`__
.. |g1_pick_place-link| replace:: `Isaac-PickPlace-G1-InspireFTP-Abs-v0 <../../../source/isaaclab_tasks/isaaclab_tasks/contrib/pick_place/pickplace_unitree_g1_inspire_hand_env_cfg.py>`__
.. |g1_pick_place_fixed-link| replace:: `Isaac-PickPlace-FixedBaseUpperBodyIK-G1-Abs-v0 <../../../source/isaaclab_tasks/isaaclab_tasks/contrib/locomanip_pick_place/fixed_base_upper_body_ik_g1_env_cfg.py>`__
.. |g1_pick_place_lm-link| replace:: `Isaac-PickPlace-Locomanipulation-G1-Abs-v0 <../../../source/isaaclab_tasks/isaaclab_tasks/contrib/locomanip_pick_place/locomanipulation_g1_env_cfg.py>`__
.. |long-suction-link| replace:: `Isaac-Stack-Cube-UR10-Long-Suction-IK-Rel-v0 <../../../source/isaaclab_tasks/isaaclab_tasks/contrib/stack/config/ur10_gripper/stack_ik_rel_env_cfg.py>`__
.. |short-suction-link| replace:: `Isaac-Stack-Cube-UR10-Short-Suction-IK-Rel-v0 <../../../source/isaaclab_tasks/isaaclab_tasks/contrib/stack/config/ur10_gripper/stack_ik_rel_env_cfg.py>`__
.. |gr1_pp_waist-link| replace:: `Isaac-PickPlace-GR1T2-WaistEnabled-Abs-v0 <../../../source/isaaclab_tasks/isaaclab_tasks/contrib/pick_place/pickplace_gr1t2_waist_enabled_env_cfg.py>`__
.. |galbot_stack-link| replace:: `Isaac-Stack-Cube-Galbot-Left-Arm-Gripper-RmpFlow-v0 <../../../source/isaaclab_tasks/isaaclab_tasks/contrib/stack/config/galbot/stack_rmp_rel_env_cfg.py>`__
.. |kuka-allegro-lift-link| replace:: `Isaac-Dexsuite-Kuka-Allegro-Lift-v0 <../../../source/isaaclab_tasks/isaaclab_tasks/core/dexsuite/config/kuka_allegro/dexsuite_kuka_allegro_env_cfg.py>`__
.. |kuka-allegro-reorient-link| replace:: `Isaac-Dexsuite-Kuka-Allegro-Reorient-v0 <../../../source/isaaclab_tasks/isaaclab_tasks/core/dexsuite/config/kuka_allegro/dexsuite_kuka_allegro_env_cfg.py>`__
.. |cube-shadow-link| replace:: `Isaac-Repose-Cube-Shadow-Direct-v0 <../../../source/isaaclab_tasks/isaaclab_tasks/core/shadow_hand/shadow_hand_env_cfg.py>`__
.. |cube-shadow-ff-link| replace:: `Isaac-Repose-Cube-Shadow-OpenAI-FF-Direct-v0 <../../../source/isaaclab_tasks/isaaclab_tasks/core/shadow_hand/shadow_hand_env_cfg.py>`__
.. |cube-shadow-lstm-link| replace:: `Isaac-Repose-Cube-Shadow-OpenAI-LSTM-Direct-v0 <../../../source/isaaclab_tasks/isaaclab_tasks/core/shadow_hand/shadow_hand_env_cfg.py>`__
.. |cube-shadow-vis-link| replace:: `Isaac-Repose-Cube-Shadow-Vision-Direct-v0 <../../../source/isaaclab_tasks/isaaclab_tasks/core/shadow_hand/shadow_hand_vision_env.py>`__
.. |agibot_place_mug-link| replace:: `Isaac-Place-Mug-Agibot-Left-Arm-RmpFlow-v0 <../../../source/isaaclab_tasks/isaaclab_tasks/contrib/place/config/agibot/place_upright_mug_rmp_rel_env_cfg.py>`__
.. |agibot_place_toy-link| replace:: `Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0 <../../../source/isaaclab_tasks/isaaclab_tasks/contrib/place/config/agibot/place_toy2box_rmp_rel_env_cfg.py>`__
.. |reach_openarm_bi-link| replace:: `Isaac-Reach-OpenArm-Bi-v0 <../../../source/isaaclab_tasks/isaaclab_tasks/contrib/reach/config/openarm/bimanual/joint_pos_env_cfg.py>`__
.. |reach_openarm_uni-link| replace:: `Isaac-Reach-OpenArm-v0 <../../../source/isaaclab_tasks/isaaclab_tasks/contrib/reach/config/openarm/unimanual/joint_pos_env_cfg.py>`__
.. |lift_openarm_uni-link| replace:: `Isaac-Lift-Cube-OpenArm-v0 <../../../source/isaaclab_tasks/isaaclab_tasks/contrib/lift/config/openarm/joint_pos_env_cfg.py>`__
.. |cabi_openarm_uni-link| replace:: `Isaac-Open-Drawer-OpenArm-v0 <../../../source/isaaclab_tasks/isaaclab_tasks/contrib/cabinet/config/openarm/joint_pos_env_cfg.py>`__
.. |g1_assemble_trocar-link| replace:: `Isaac-Assemble-Trocar-G129-Dex3-v0 <../../../source/isaaclab_tasks/isaaclab_tasks/contrib/assemble_trocar/g129_dex3_env_cfg.py>`__


Contact-rich Manipulation
~~~~~~~~~~~~~~~~~~~~~~~~~

Environments based on contact-rich manipulation tasks such as peg insertion, gear meshing and nut-bolt fastening.

These tasks share the same task configurations and control options. You can switch between them by specifying the task name.
For example:

* |factory-peg-link|: Peg insertion with the Franka arm
* |factory-gear-link|: Gear meshing with the Franka arm
* |factory-nut-link|: Nut-Bolt fastening with the Franka arm

.. table::
    :widths: 25 30 25 20

    +--------------------+-------------------------+-----------------------------------------------------------------------------+------------------------------+
    | World              | Environment ID          | Description                                                                 | Presets                      |
    +====================+=========================+=============================================================================+==============================+
    | |factory-peg|      | |factory-peg-link|      | Insert peg into the socket with the Franka robot                            |                              |
    +--------------------+-------------------------+-----------------------------------------------------------------------------+------------------------------+
    | |factory-gear|     | |factory-gear-link|     | Insert and mesh gear into the base with other gears, using the Franka robot |                              |
    +--------------------+-------------------------+-----------------------------------------------------------------------------+------------------------------+
    | |factory-nut|      | |factory-nut-link|      | Thread the nut onto the first 2 threads of the bolt, using the Franka robot |                              |
    +--------------------+-------------------------+-----------------------------------------------------------------------------+------------------------------+

.. |factory-peg| image:: ../_static/tasks/factory/peg_insert.jpg
.. |factory-gear| image:: ../_static/tasks/factory/gear_mesh.jpg
.. |factory-nut| image:: ../_static/tasks/factory/nut_thread.jpg

.. |factory-peg-link| replace:: `Isaac-Factory-PegInsert-Direct-v0 <../../../source/isaaclab_tasks/isaaclab_tasks/contrib/factory/factory_env_cfg.py>`__
.. |factory-gear-link| replace:: `Isaac-Factory-GearMesh-Direct-v0 <../../../source/isaaclab_tasks/isaaclab_tasks/contrib/factory/factory_env_cfg.py>`__
.. |factory-nut-link| replace:: `Isaac-Factory-NutThread-Direct-v0 <../../../source/isaaclab_tasks/isaaclab_tasks/contrib/factory/factory_env_cfg.py>`__

AutoMate
~~~~~~~~

Environments based on 100 diverse assembly tasks, each involving the insertion of a plug into a socket. These tasks share a common configuration and differ by th geometry and properties of the parts.

You can switch between tasks by specifying the corresponding asset ID. Available asset IDs include:

'00004', '00007', '00014', '00015', '00016', '00021', '00028', '00030', '00032', '00042', '00062', '00074', '00077', '00078', '00081', '00083', '00103', '00110', '00117', '00133', '00138', '00141', '00143', '00163', '00175', '00186', '00187', '00190', '00192', '00210', '00211', '00213', '00255', '00256', '00271', '00293', '00296', '00301', '00308', '00318', '00319', '00320', '00329', '00340', '00345', '00346', '00360', '00388', '00410', '00417', '00422', '00426', '00437', '00444', '00446', '00470', '00471', '00480', '00486', '00499', '00506', '00514', '00537', '00553', '00559', '00581', '00597', '00614', '00615', '00638', '00648', '00649', '00652', '00659', '00681', '00686', '00700', '00703', '00726', '00731', '00741', '00755', '00768', '00783', '00831', '00855', '00860', '00863', '01026', '01029', '01036', '01041', '01053', '01079', '01092', '01102', '01125', '01129', '01132', '01136'.

We provide environments for both disassembly and assembly.

.. attention::

  CUDA is recommended for running the AutoMate environments. If running with Nvidia driver 570 on Linux with architecture x86_64, we follow the below steps to install CUDA 12.8. This allows for computing rewards in AutoMate environments with CUDA. If you have a different operation system or architecture, please refer to the `CUDA installation page <https://developer.nvidia.com/cuda-12-8-0-download-archive>`_ for additional instruction.

  .. code-block:: bash

      wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda_12.8.0_570.86.10_linux.run
      sudo sh cuda_12.8.0_570.86.10_linux.run --toolkit

  When using conda, cuda toolkit can be installed with:

  .. code-block:: bash

      conda install cudatoolkit

  With 580 drivers on Linux with architecture x86_64, we install CUDA 13 and additionally install several packages. Please ensure that the pytorch version is compatible with the CUDA version.

  .. code-block:: bash

      wget https://developer.download.nvidia.com/compute/cuda/13.0.2/local_installers/cuda_13.0.2_580.95.05_linux.run
      sudo sh cuda_13.0.2_580.95.05_linux.run --toolkit
      pip install numba-cuda[cu13] coverage==7.6.1


* |disassembly-link|: The plug starts inserted in the socket. A low-level controller lifts the plug out and moves it to a random position. This process is purely scripted and does not involve any learned policy. Therefore, it does not require policy training or evaluation. The resulting trajectories serve as demonstrations for the reverse process, i.e., learning to assemble. To run disassembly for a specific task: ``python source/isaaclab_tasks/isaaclab_tasks/contrib/automate/run_disassembly_w_id.py --assembly_id=ASSEMBLY_ID --disassembly_dir=DISASSEMBLY_DIR``. All generated trajectories are saved to a local directory ``DISASSEMBLY_DIR``.
* |assembly-link|: The goal is to insert the plug into the socket. You can use this environment to train a policy via reinforcement learning or evaluate a pre-trained checkpoint.

  * To train an assembly policy, we run the command ``python source/isaaclab_tasks/isaaclab_tasks/contrib/automate/run_w_id.py --assembly_id=ASSEMBLY_ID --train``. We can customize the training process using the optional flags: ``--headless`` to run without opening the GUI windows, ``--max_iterations=MAX_ITERATIONS`` to set the number of training iterations, ``--num_envs=NUM_ENVS`` to set the number of parallel environments during training, ``--seed=SEED`` to assign the random seed. The policy checkpoints will be saved automatically during training in the directory ``logs/rl_games/Assembly/test``.
  * To evaluate an assembly policy, we run the command ``python source/isaaclab_tasks/isaaclab_tasks/contrib/automate/run_w_id.py --assembly_id=ASSEMBLY_ID --checkpoint=CHECKPOINT --log_eval``. The evaluation results are stored in ``evaluation_{ASSEMBLY_ID}.h5``.

.. table::
    :widths: 25 30 25 20

    +--------------------+-------------------------+-----------------------------------------------------------------------------+------------------------------+
    | World              | Environment ID          | Description                                                                 | Presets                      |
    +====================+=========================+=============================================================================+==============================+
    | |disassembly|      | |disassembly-link|      | Lift a plug out of the socket with the Franka robot                         |                              |
    +--------------------+-------------------------+-----------------------------------------------------------------------------+------------------------------+
    | |assembly|         | |assembly-link|         | Insert a plug into its corresponding socket with the Franka robot           |                              |
    +--------------------+-------------------------+-----------------------------------------------------------------------------+------------------------------+

.. |assembly| image:: ../_static/tasks/automate/00004.jpg
.. |disassembly| image:: ../_static/tasks/automate/01053_disassembly.jpg

.. |assembly-link| replace:: `Isaac-AutoMate-Assembly-Direct-v0 <../../../source/isaaclab_tasks/isaaclab_tasks/contrib/automate/assembly_env_cfg.py>`__
.. |disassembly-link| replace:: `Isaac-AutoMate-Disassembly-Direct-v0 <../../../source/isaaclab_tasks/isaaclab_tasks/contrib/automate/disassembly_env_cfg.py>`__

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
    :widths: 25 30 25 20

    +--------------------+-------------------------+-----------------------------------------------------------------------------+------------------------------+
    | World              | Environment ID          | Description                                                                 | Presets                      |
    +====================+=========================+=============================================================================+==============================+
    | |forge-peg|        | |forge-peg-link|        | Insert peg into the socket with the Franka robot                            |                              |
    +--------------------+-------------------------+-----------------------------------------------------------------------------+------------------------------+
    | |forge-gear|       | |forge-gear-link|       | Insert and mesh gear into the base with other gears, using the Franka robot |                              |
    +--------------------+-------------------------+-----------------------------------------------------------------------------+------------------------------+
    | |forge-nut|        | |forge-nut-link|        | Thread the nut onto the first 2 threads of the bolt, using the Franka robot |                              |
    +--------------------+-------------------------+-----------------------------------------------------------------------------+------------------------------+

.. |forge-peg| image:: ../_static/tasks/factory/peg_insert.jpg
.. |forge-gear| image:: ../_static/tasks/factory/gear_mesh.jpg
.. |forge-nut| image:: ../_static/tasks/factory/nut_thread.jpg

.. |forge-peg-link| replace:: `Isaac-Forge-PegInsert-Direct-v0 <../../../source/isaaclab_tasks/isaaclab_tasks/contrib/forge/forge_env_cfg.py>`__
.. |forge-gear-link| replace:: `Isaac-Forge-GearMesh-Direct-v0 <../../../source/isaaclab_tasks/isaaclab_tasks/contrib/forge/forge_env_cfg.py>`__
.. |forge-nut-link| replace:: `Isaac-Forge-NutThread-Direct-v0 <../../../source/isaaclab_tasks/isaaclab_tasks/contrib/forge/forge_env_cfg.py>`__


Locomotion
~~~~~~~~~~

Environments based on legged locomotion tasks.

.. table::
    :widths: 25 30 25 20

    +------------------------------+----------------------------------------------+------------------------------------------------------------------------------+------------------------------+
    | World                        | Environment ID                               | Description                                                                  | Presets                      |
    +==============================+==============================================+==============================================================================+==============================+
    | |velocity-flat-anymal-b|     | |velocity-flat-anymal-b-link|                | Track a velocity command on flat terrain with the Anymal B robot             | **physics=** ``physx``,      |
    |                              |                                              |                                                                              | ``newton_mjwarp``            |
    +------------------------------+----------------------------------------------+------------------------------------------------------------------------------+------------------------------+
    | |velocity-rough-anymal-b|    | |velocity-rough-anymal-b-link|               | Track a velocity command on rough terrain with the Anymal B robot            | **physics=** ``physx``,      |
    |                              |                                              |                                                                              | ``newton_mjwarp``            |
    +------------------------------+----------------------------------------------+------------------------------------------------------------------------------+------------------------------+
    | |velocity-flat-anymal-c|     | |velocity-flat-anymal-c-link|                | Track a velocity command on flat terrain with the Anymal C robot             | **physics=** ``physx``,      |
    |                              |                                              |                                                                              | ``newton_mjwarp``            |
    |                              |                                              |                                                                              | (Manager-based only)         |
    |                              | |velocity-flat-anymal-c-direct-link|         |                                                                              |                              |
    +------------------------------+----------------------------------------------+------------------------------------------------------------------------------+------------------------------+
    | |velocity-rough-anymal-c|    | |velocity-rough-anymal-c-link|               | Track a velocity command on rough terrain with the Anymal C robot            | **physics=** ``physx``,      |
    |                              |                                              |                                                                              | ``newton_mjwarp``            |
    |                              |                                              |                                                                              | (Manager-based only)         |
    |                              | |velocity-rough-anymal-c-direct-link|        |                                                                              |                              |
    +------------------------------+----------------------------------------------+------------------------------------------------------------------------------+------------------------------+
    | |velocity-flat-anymal-d|     | |velocity-flat-anymal-d-link|                | Track a velocity command on flat terrain with the Anymal D robot             | **physics=** ``physx``,      |
    |                              |                                              |                                                                              | ``newton_mjwarp``,           |
    |                              |                                              |                                                                              | ``ovphysx``                  |
    +------------------------------+----------------------------------------------+------------------------------------------------------------------------------+------------------------------+
    | |velocity-rough-anymal-d|    | |velocity-rough-anymal-d-link|               | Track a velocity command on rough terrain with the Anymal D robot            | **physics=** ``physx``,      |
    |                              |                                              |                                                                              | ``newton_mjwarp``            |
    +------------------------------+----------------------------------------------+------------------------------------------------------------------------------+------------------------------+
    | |velocity-flat-unitree-a1|   | |velocity-flat-unitree-a1-link|              | Track a velocity command on flat terrain with the Unitree A1 robot           | **physics=** ``physx``,      |
    |                              |                                              |                                                                              | ``newton_mjwarp``            |
    +------------------------------+----------------------------------------------+------------------------------------------------------------------------------+------------------------------+
    | |velocity-rough-unitree-a1|  | |velocity-rough-unitree-a1-link|             | Track a velocity command on rough terrain with the Unitree A1 robot          | **physics=** ``physx``,      |
    |                              |                                              |                                                                              | ``newton_mjwarp``            |
    +------------------------------+----------------------------------------------+------------------------------------------------------------------------------+------------------------------+
    | |velocity-flat-unitree-go1|  | |velocity-flat-unitree-go1-link|             | Track a velocity command on flat terrain with the Unitree Go1 robot          | **physics=** ``physx``,      |
    |                              |                                              |                                                                              | ``newton_mjwarp``            |
    +------------------------------+----------------------------------------------+------------------------------------------------------------------------------+------------------------------+
    | |velocity-rough-unitree-go1| | |velocity-rough-unitree-go1-link|            | Track a velocity command on rough terrain with the Unitree Go1 robot         | **physics=** ``physx``,      |
    |                              |                                              |                                                                              | ``newton_mjwarp``            |
    +------------------------------+----------------------------------------------+------------------------------------------------------------------------------+------------------------------+
    | |velocity-flat-unitree-go2|  | |velocity-flat-unitree-go2-link|             | Track a velocity command on flat terrain with the Unitree Go2 robot          | **physics=** ``physx``,      |
    |                              |                                              |                                                                              | ``newton_mjwarp``            |
    +------------------------------+----------------------------------------------+------------------------------------------------------------------------------+------------------------------+
    | |velocity-rough-unitree-go2| | |velocity-rough-unitree-go2-link|            | Track a velocity command on rough terrain with the Unitree Go2 robot         | **physics=** ``physx``,      |
    |                              |                                              |                                                                              | ``newton_mjwarp``            |
    +------------------------------+----------------------------------------------+------------------------------------------------------------------------------+------------------------------+
    | |velocity-flat-spot|         | |velocity-flat-spot-link|                    | Track a velocity command on flat terrain with the Boston Dynamics Spot robot | **physics=** ``physx``,      |
    |                              |                                              |                                                                              | ``newton_mjwarp``            |
    +------------------------------+----------------------------------------------+------------------------------------------------------------------------------+------------------------------+
    | |velocity-flat-h1|           | |velocity-flat-h1-link|                      | Track a velocity command on flat terrain with the Unitree H1 robot           | **physics=** ``physx``,      |
    |                              |                                              |                                                                              | ``newton_mjwarp``            |
    +------------------------------+----------------------------------------------+------------------------------------------------------------------------------+------------------------------+
    | |velocity-rough-h1|          | |velocity-rough-h1-link|                     | Track a velocity command on rough terrain with the Unitree H1 robot          | **physics=** ``physx``,      |
    |                              |                                              |                                                                              | ``newton_mjwarp``            |
    +------------------------------+----------------------------------------------+------------------------------------------------------------------------------+------------------------------+
    | |velocity-flat-g1|           | |velocity-flat-g1-link|                      | Track a velocity command on flat terrain with the Unitree G1 robot           | **physics=** ``physx``,      |
    |                              |                                              |                                                                              | ``newton_mjwarp``            |
    +------------------------------+----------------------------------------------+------------------------------------------------------------------------------+------------------------------+
    | |velocity-rough-g1|          | |velocity-rough-g1-link|                     | Track a velocity command on rough terrain with the Unitree G1 robot          | **physics=** ``physx``,      |
    |                              |                                              |                                                                              | ``newton_mjwarp``            |
    +------------------------------+----------------------------------------------+------------------------------------------------------------------------------+------------------------------+
    | |velocity-flat-digit|        | |velocity-flat-digit-link|                   | Track a velocity command on flat terrain with the Agility Digit robot        | **physics=** ``physx``       |
    +------------------------------+----------------------------------------------+------------------------------------------------------------------------------+------------------------------+
    | |velocity-rough-digit|       | |velocity-rough-digit-link|                  | Track a velocity command on rough terrain with the Agility Digit robot       | **physics=** ``physx``       |
    +------------------------------+----------------------------------------------+------------------------------------------------------------------------------+------------------------------+
    | |tracking-loco-manip-digit|  | |tracking-loco-manip-digit-link|             | Track a root velocity and hand pose command with the Agility Digit robot     | **physics=** ``physx``       |
    +------------------------------+----------------------------------------------+------------------------------------------------------------------------------+------------------------------+

.. |velocity-flat-anymal-b-link| replace:: `Isaac-Velocity-Flat-Anymal-B-v0 <../../../source/isaaclab_tasks/isaaclab_tasks/contrib/velocity/config/anymal_b/flat_env_cfg.py>`__
.. |velocity-rough-anymal-b-link| replace:: `Isaac-Velocity-Rough-Anymal-B-v0 <../../../source/isaaclab_tasks/isaaclab_tasks/contrib/velocity/config/anymal_b/rough_env_cfg.py>`__

.. |velocity-flat-anymal-c-link| replace:: `Isaac-Velocity-Flat-Anymal-C-v0 <../../../source/isaaclab_tasks/isaaclab_tasks/contrib/velocity/config/anymal_c/flat_env_cfg.py>`__
.. |velocity-rough-anymal-c-link| replace:: `Isaac-Velocity-Rough-Anymal-C-v0 <../../../source/isaaclab_tasks/isaaclab_tasks/contrib/velocity/config/anymal_c/rough_env_cfg.py>`__

.. |velocity-flat-anymal-c-direct-link| replace:: `Isaac-Velocity-Flat-Anymal-C-Direct-v0 <../../../source/isaaclab_tasks/isaaclab_tasks/contrib/anymal_c_direct/anymal_c_env.py>`__
.. |velocity-rough-anymal-c-direct-link| replace:: `Isaac-Velocity-Rough-Anymal-C-Direct-v0 <../../../source/isaaclab_tasks/isaaclab_tasks/contrib/anymal_c_direct/anymal_c_env.py>`__

.. |velocity-flat-anymal-d-link| replace:: `Isaac-Velocity-Flat-Anymal-D-v0 <../../../source/isaaclab_tasks/isaaclab_tasks/core/velocity/config/anymal_d/flat_env_cfg.py>`__
.. |velocity-rough-anymal-d-link| replace:: `Isaac-Velocity-Rough-Anymal-D-v0 <../../../source/isaaclab_tasks/isaaclab_tasks/core/velocity/config/anymal_d/rough_env_cfg.py>`__

.. |velocity-flat-unitree-a1-link| replace:: `Isaac-Velocity-Flat-Unitree-A1-v0 <../../../source/isaaclab_tasks/isaaclab_tasks/contrib/velocity/config/a1/flat_env_cfg.py>`__
.. |velocity-rough-unitree-a1-link| replace:: `Isaac-Velocity-Rough-Unitree-A1-v0 <../../../source/isaaclab_tasks/isaaclab_tasks/contrib/velocity/config/a1/rough_env_cfg.py>`__

.. |velocity-flat-unitree-go1-link| replace:: `Isaac-Velocity-Flat-Unitree-Go1-v0 <../../../source/isaaclab_tasks/isaaclab_tasks/contrib/velocity/config/go1/flat_env_cfg.py>`__
.. |velocity-rough-unitree-go1-link| replace:: `Isaac-Velocity-Rough-Unitree-Go1-v0 <../../../source/isaaclab_tasks/isaaclab_tasks/contrib/velocity/config/go1/rough_env_cfg.py>`__

.. |velocity-flat-unitree-go2-link| replace:: `Isaac-Velocity-Flat-Unitree-Go2-v0 <../../../source/isaaclab_tasks/isaaclab_tasks/core/velocity/config/go2/flat_env_cfg.py>`__
.. |velocity-rough-unitree-go2-link| replace:: `Isaac-Velocity-Rough-Unitree-Go2-v0 <../../../source/isaaclab_tasks/isaaclab_tasks/core/velocity/config/go2/rough_env_cfg.py>`__

.. |velocity-flat-spot-link| replace:: `Isaac-Velocity-Flat-Spot-v0 <../../../source/isaaclab_tasks/isaaclab_tasks/core/velocity/config/spot/flat_env_cfg.py>`__

.. |velocity-flat-h1-link| replace:: `Isaac-Velocity-Flat-H1-v0 <../../../source/isaaclab_tasks/isaaclab_tasks/core/velocity/config/h1/flat_env_cfg.py>`__
.. |velocity-rough-h1-link| replace:: `Isaac-Velocity-Rough-H1-v0 <../../../source/isaaclab_tasks/isaaclab_tasks/core/velocity/config/h1/rough_env_cfg.py>`__

.. |velocity-flat-g1-link| replace:: `Isaac-Velocity-Flat-G1-v0 <../../../source/isaaclab_tasks/isaaclab_tasks/core/velocity/config/g1/flat_env_cfg.py>`__
.. |velocity-rough-g1-link| replace:: `Isaac-Velocity-Rough-G1-v0 <../../../source/isaaclab_tasks/isaaclab_tasks/core/velocity/config/g1/rough_env_cfg.py>`__

.. |velocity-flat-digit-link| replace:: `Isaac-Velocity-Flat-Digit-v0 <../../../source/isaaclab_tasks/isaaclab_tasks/core/velocity/config/digit/flat_env_cfg.py>`__
.. |velocity-rough-digit-link| replace:: `Isaac-Velocity-Rough-Digit-v0 <../../../source/isaaclab_tasks/isaaclab_tasks/core/velocity/config/digit/rough_env_cfg.py>`__
.. |tracking-loco-manip-digit-link| replace:: `Isaac-Tracking-LocoManip-Digit-v0 <../../../source/isaaclab_tasks/isaaclab_tasks/contrib/locomanip_tracking/config/digit/loco_manip_env_cfg.py>`__

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

.. note::

   Agility Digit environments use closed-loop articulations (achilles rod, toe
   push-rods) that do not run correctly on ``newton_mjwarp``. Use ``physx`` for
   Digit-based tasks; see :ref:`known-issues-closed-loop-newton` for details.

Navigation
~~~~~~~~~~

.. table::
    :widths: 25 30 25 20

    +----------------+---------------------+-----------------------------------------------------------------------------+------------------------------+
    | World          | Environment ID      | Description                                                                 | Presets                      |
    +================+=====================+=============================================================================+==============================+
    | |anymal_c_nav| | |anymal_c_nav-link| | Navigate towards a target x-y position and heading with the ANYmal C robot. | **physics=** ``physx``,      |
    |                |                     |                                                                             | ``newton_mjwarp``            |
    +----------------+---------------------+-----------------------------------------------------------------------------+------------------------------+

.. |anymal_c_nav-link| replace:: `Isaac-Navigation-Flat-Anymal-C-v0 <../../../source/isaaclab_tasks/isaaclab_tasks/contrib/navigation/config/anymal_c/navigation_env_cfg.py>`__

.. |anymal_c_nav| image:: ../_static/tasks/navigation/anymal_c_nav.jpg


Multirotor
~~~~~~~~~~

.. note::
    The multirotor entry provides an environment configuration for flying the ARL robot.
    See the `drone_arl` folder and the ARL robot config
    (`ARL_ROBOT_1_CFG`) in the codebase for details.

.. |arl_robot_track_position_state_based-link| replace:: `Isaac-TrackPositionNoObstacles-ARL-Robot-1-v0 <../../../source/isaaclab_tasks/isaaclab_tasks/contrib/drone_arl/track_position_state_based/config/arl_robot_1/track_position_state_based_env_cfg.py>`__

.. |arl_robot_track_position_state_based| image:: ../_static/tasks/drone_arl/arl_robot_1_track_position_state_based.jpg

.. |arl_robot_navigation-link| replace:: `Isaac-Navigation-3DObstacles-ARL-Robot-1-v0 <../../../source/isaaclab_tasks/isaaclab_tasks/contrib/drone_arl/navigation/config/arl_robot_1/floating_obstacles_env_cfg.py>`__

.. |arl_robot_navigation| image:: ../_static/tasks/drone_arl/arl_robot_1_navigation.jpg

.. table::
    :widths: 25 30 25 20

    +----------------------------------------+---------------------------------------------+----------------------------------------------------------------------------------------+-----------------------+
    | World                                  | Environment ID                              | Description                                                                            | Presets               |
    +========================================+=============================================+========================================================================================+=======================+
    | |arl_robot_track_position_state_based| | |arl_robot_track_position_state_based-link| | Setpoint position control for the ARL robot using the track_position_state_based task. |                       |
    +----------------------------------------+---------------------------------------------+----------------------------------------------------------------------------------------+-----------------------+
    | |arl_robot_navigation|                 | |arl_robot_navigation-link|                 | Navigate through 3D obstacles with the ARL robot using depth camera sensing.           |                       |
    +----------------------------------------+---------------------------------------------+----------------------------------------------------------------------------------------+-----------------------+


Others
~~~~~~

.. note::

    Adversarial Motion Priors (AMP) training is only available with the `skrl` library, as it is the only one of the currently
    integrated libraries that supports it out-of-the-box (for the other libraries, it is necessary to implement the algorithm and architectures).
    See the `skrl's AMP Documentation <https://skrl.readthedocs.io/en/latest/api/agents/amp.html>`_ for more information.
    The AMP algorithm can be activated by adding the command line input ``--algorithm AMP`` to the train/play script.

    For evaluation, the play script's command line input ``--real-time`` allows the interaction loop between the environment and the agent to run in real time, if possible.

.. table::
    :widths: 25 30 25 20

    +----------------+---------------------------+-----------------------------------------------------------------------------+------------------------------+
    | World          | Environment ID            | Description                                                                 | Presets                      |
    +================+===========================+=============================================================================+==============================+
    | |humanoid_amp| | |humanoid_amp_dance-link| | Move a humanoid robot by imitating different pre-recorded human animations  |                              |
    |                |                           | (Adversarial Motion Priors).                                                |                              |
    |                | |humanoid_amp_run-link|   |                                                                             |                              |
    |                |                           |                                                                             |                              |
    |                | |humanoid_amp_walk-link|  |                                                                             |                              |
    +----------------+---------------------------+-----------------------------------------------------------------------------+------------------------------+

.. |humanoid_amp_dance-link| replace:: `Isaac-Humanoid-AMP-Dance-Direct-v0 <../../../source/isaaclab_tasks/isaaclab_tasks/contrib/humanoid_amp/humanoid_amp_env_cfg.py>`__
.. |humanoid_amp_run-link| replace:: `Isaac-Humanoid-AMP-Run-Direct-v0 <../../../source/isaaclab_tasks/isaaclab_tasks/contrib/humanoid_amp/humanoid_amp_env_cfg.py>`__
.. |humanoid_amp_walk-link| replace:: `Isaac-Humanoid-AMP-Walk-Direct-v0 <../../../source/isaaclab_tasks/isaaclab_tasks/contrib/humanoid_amp/humanoid_amp_env_cfg.py>`__

.. |humanoid_amp| image:: ../_static/tasks/others/humanoid_amp.jpg

Spaces showcase
~~~~~~~~~~~~~~~

The |cartpole_showcase| folder contains showcase tasks (based on the *Cartpole* and *Cartpole-Camera* Direct tasks)
for the definition/use of the various Gymnasium observation and action spaces supported in Isaac Lab.

.. |cartpole_showcase| replace:: `cartpole_showcase <../../../source/isaaclab_tasks/isaaclab_tasks/contrib/cartpole_showcase>`__

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
    :widths: 25 30 25 20

    +------------------------+------------------------------------+-----------------------------------------------------------------------------------------------------------------------+------------------------------+
    | World                  | Environment ID                     | Description                                                                                                           | Presets                      |
    +========================+====================================+=======================================================================================================================+==============================+
    | |cart-double-pendulum| | |cart-double-pendulum-direct-link| | Move the cart and the pendulum to keep the last one upwards in the classic inverted double pendulum on a cart control |                              |
    +------------------------+------------------------------------+-----------------------------------------------------------------------------------------------------------------------+------------------------------+

.. |cart-double-pendulum| image:: ../_static/tasks/classic/cart_double_pendulum.jpg

.. |cart-double-pendulum-direct-link| replace:: `Isaac-Cart-Double-Pendulum-Direct <../../../source/isaaclab_tasks/isaaclab_tasks/core/cart_double_pendulum/cart_double_pendulum_env.py>`__

Manipulation
~~~~~~~~~~~~

Environments based on fixed-arm manipulation tasks.

.. table::
    :widths: 25 30 25 20

    +----------------------+--------------------------------+--------------------------------------------------------+------------------------------+
    | World                | Environment ID                 | Description                                            | Presets                      |
    +======================+================================+========================================================+==============================+
    | |shadow-hand-over|   | |shadow-hand-over-direct-link| | Passing an object from one hand over to the other hand |                              |
    +----------------------+--------------------------------+--------------------------------------------------------+------------------------------+

.. |shadow-hand-over| image:: ../_static/tasks/manipulation/shadow_hand_over.jpg

.. |shadow-hand-over-direct-link| replace:: `Isaac-Shadow-Hand-Over-Direct-v0 <../../../source/isaaclab_tasks/isaaclab_tasks/core/shadow_hand_over/shadow_hand_over_env.py>`__

|

.. _comprehensive-environment-list:

Comprehensive List of Environments
==================================

For environments that have a different task name listed under ``Inference Task Name``, please use the Inference Task Name
provided when running ``play.py`` or any inferencing workflows. These tasks provide more suitable configurations for
inferencing, including reading from an already trained checkpoint and disabling runtime perturbations used for training.

.. START-AUTO-GENERATED: comprehensive-environment-list

.. list-table::
    :widths: 18 17 10 22 33

    * - **Task Name**
      - **Inference Task Name**
      - **Workflow**
      - **RL Library**
      - **Presets**
    * - Isaac-Ant-Direct-Warp-v0
      -
      - Direct
      - **rl_games** (PPO), **rsl_rl** (PPO), **skrl** (PPO)
      -
    * - Isaac-Ant-Direct
      -
      - Direct
      - **rl_games** (PPO), **rsl_rl** (PPO), **skrl** (PPO)
      - **physics=** ``newton_kamino``, ``newton_mjwarp``, ``ovphysx``, ``physx``
    * - Isaac-Ant-Warp-v0
      -
      - Manager Based
      - **rl_games** (PPO), **rsl_rl** (PPO), **skrl** (PPO), **sb3** (PPO)
      -
    * - Isaac-Ant
      -
      - Manager Based
      - **rl_games** (PPO), **rsl_rl** (PPO), **skrl** (PPO), **sb3** (PPO)
      - **physics=** ``newton_kamino``, ``newton_mjwarp``, ``physx``
    * - Isaac-Assemble-Trocar-G129-Dex3-v0
      - Isaac-Assemble-Trocar-G129-Dex3-Eval-v0
      - Manager Based
      - **rlinf** (PPO)
      -
    * - Isaac-AutoMate-Assembly-Direct-v0
      -
      - Direct
      - **rl_games** (PPO)
      -
    * - Isaac-AutoMate-Disassembly-Direct-v0
      -
      - Direct
      - **rl_games** (PPO)
      -
    * - Isaac-Cart-Double-Pendulum-Direct
      -
      - Direct
      - **rl_games** (PPO), **skrl** (PPO, IPPO, MAPPO)
      -
    * - Isaac-Cartpole
      -
      - Manager Based
      - **rl_games** (PPO), **rsl_rl** (PPO), **skrl** (PPO), **sb3** (PPO)
      - **physics=** ``newton_kamino``, ``newton_mjwarp``, ``physx``
    * - Isaac-Cartpole-Camera
      -
      - Manager Based
      - **rl_games** (PPO, FEATURE)
      - | **physics=** ``newton_kamino``, ``newton_mjwarp``, ``physx``
          | **presets=** ``depth``, ``resnet18``, ``rgb``, ``theia_tiny``
    * - Isaac-Cartpole-Camera-Direct
      -
      - Direct
      - **rl_games** (PPO), **skrl** (PPO)
      - | **physics=** ``newton_mjwarp``, ``ovphysx``, ``physx``
          | **renderer=** ``isaacsim_rtx_renderer``, ``newton_renderer``, ``ovrtx_renderer``
          | **presets=** ``albedo``, ``depth``, ``rgb``, ``semantic_segmentation``, ``simple_shading_constant_diffuse``, ``simple_shading_diffuse_mdl``, ``simple_shading_full_mdl``
    * - Isaac-Cartpole-Camera-Showcase-Direct
      -
      - Direct
      - **skrl** (PPO, BOX_BOX, BOX_DISCRETE, BOX_MULTIDISCRETE, DICT_BOX, DICT_DISCRETE, DICT_MULTIDISCRETE, TUPLE_BOX, TUPLE_DISCRETE, TUPLE_MULTIDISCRETE)
      - **presets=** ``box_box``, ``box_discrete``, ``box_multidiscrete``, ``dict_box``, ``dict_discrete``, ``dict_multidiscrete``, ``tuple_box``, ``tuple_discrete``, ``tuple_multidiscrete``
    * - Isaac-Cartpole-Direct
      -
      - Direct
      - **rl_games** (PPO), **rsl_rl** (PPO), **skrl** (PPO), **sb3** (PPO)
      - **physics=** ``newton_kamino``, ``newton_mjwarp``, ``ovphysx``, ``physx``
    * - Isaac-Cartpole-Direct-Warp-v0
      -
      - Direct
      - **rl_games** (PPO), **rsl_rl** (PPO), **skrl** (PPO), **sb3** (PPO)
      -
    * - Isaac-Cartpole-Showcase-Direct
      -
      - Direct
      - **skrl** (PPO, BOX_BOX, BOX_DISCRETE, BOX_MULTIDISCRETE, DICT_BOX, DICT_DISCRETE, DICT_MULTIDISCRETE, DISCRETE_BOX, DISCRETE_DISCRETE, DISCRETE_MULTIDISCRETE, MULTIDISCRETE_BOX, MULTIDISCRETE_DISCRETE, MULTIDISCRETE_MULTIDISCRETE, TUPLE_BOX, TUPLE_DISCRETE, TUPLE_MULTIDISCRETE)
      - | **physics=** ``newton_kamino``, ``newton_mjwarp``, ``ovphysx``, ``physx``
          | **presets=** ``box_box``, ``box_discrete``, ``box_multidiscrete``, ``dict_box``, ``dict_discrete``, ``dict_multidiscrete``, ``discrete_box``, ``discrete_discrete``, ``discrete_multidiscrete``, ``multidiscrete_box``, ``multidiscrete_discrete``, ``multidiscrete_multidiscrete``, ``tuple_box``, ``tuple_discrete``, ``tuple_multidiscrete``
    * - Isaac-Deploy-GearAssembly-Rizon4s-Grav-ROS-Inference-v0
      -
      - Manager Based
      - **rsl_rl** (PPO)
      -
    * - Isaac-Deploy-GearAssembly-Rizon4s-Grav-v0
      - Isaac-Deploy-GearAssembly-Rizon4s-Grav-Play-v0
      - Manager Based
      - **rsl_rl** (PPO)
      -
    * - Isaac-Deploy-GearAssembly-UR10e-2F140-ROS-Inference-v0
      -
      - Manager Based
      - **rsl_rl** (PPO)
      -
    * - Isaac-Deploy-GearAssembly-UR10e-2F140-v0
      - Isaac-Deploy-GearAssembly-UR10e-2F140-Play-v0
      - Manager Based
      - **rsl_rl** (PPO)
      -
    * - Isaac-Deploy-GearAssembly-UR10e-2F85-ROS-Inference-v0
      -
      - Manager Based
      - **rsl_rl** (PPO)
      -
    * - Isaac-Deploy-GearAssembly-UR10e-2F85-v0
      - Isaac-Deploy-GearAssembly-UR10e-2F85-Play-v0
      - Manager Based
      - **rsl_rl** (PPO)
      -
    * - Isaac-Deploy-Reach-Rizon4s-ROS-Inference-v0
      -
      - Manager Based
      - **rsl_rl** (PPO)
      -
    * - Isaac-Deploy-Reach-Rizon4s-v0
      - Isaac-Deploy-Reach-Rizon4s-Play-v0
      - Manager Based
      - **rsl_rl** (PPO)
      -
    * - Isaac-Deploy-Reach-UR10e-ROS-Inference-v0
      -
      - Manager Based
      - **rsl_rl** (PPO)
      -
    * - Isaac-Deploy-Reach-UR10e-v0
      - Isaac-Deploy-Reach-UR10e-Play-v0
      - Manager Based
      - **rsl_rl** (PPO)
      -
    * - Isaac-Dexsuite-Kuka-Allegro-Lift-v0
      - Isaac-Dexsuite-Kuka-Allegro-Lift-Play-v0
      - Manager Based
      - **rsl_rl** (PPO)
      - | **physics=** ``newton_mjwarp``, ``physx``
          | **renderer=** ``isaacsim_rtx_renderer``, ``newton_renderer``, ``ovrtx_renderer``
          | **presets=** ``albedo128``, ``albedo256``, ``albedo64``, ``cube``, ``depth128``, ``depth256``, ``depth64``, ``duo_camera``, ``raycaster_depth128``, ``raycaster_depth256``, ``raycaster_depth64``, ``rgb128``, ``rgb256``, ``rgb64``, ``semantic_segmentation128``, ``semantic_segmentation256``, ``semantic_segmentation64``, ``shapes``, ``simple_shading_constant_diffuse128``, ``simple_shading_constant_diffuse256``, ``simple_shading_constant_diffuse64``, ``simple_shading_diffuse_mdl128``, ``simple_shading_diffuse_mdl256``, ``simple_shading_diffuse_mdl64``, ``simple_shading_full_mdl128``, ``simple_shading_full_mdl256``, ``simple_shading_full_mdl64``, ``single_camera``, ``state``
    * - Isaac-Dexsuite-Kuka-Allegro-Reorient-v0
      - Isaac-Dexsuite-Kuka-Allegro-Reorient-Play-v0
      - Manager Based
      - **rsl_rl** (PPO)
      - | **physics=** ``newton_mjwarp``, ``physx``
          | **renderer=** ``isaacsim_rtx_renderer``, ``newton_renderer``, ``ovrtx_renderer``
          | **presets=** ``albedo128``, ``albedo256``, ``albedo64``, ``cube``, ``depth128``, ``depth256``, ``depth64``, ``duo_camera``, ``raycaster_depth128``, ``raycaster_depth256``, ``raycaster_depth64``, ``rgb128``, ``rgb256``, ``rgb64``, ``semantic_segmentation128``, ``semantic_segmentation256``, ``semantic_segmentation64``, ``shapes``, ``simple_shading_constant_diffuse128``, ``simple_shading_constant_diffuse256``, ``simple_shading_constant_diffuse64``, ``simple_shading_diffuse_mdl128``, ``simple_shading_diffuse_mdl256``, ``simple_shading_diffuse_mdl64``, ``simple_shading_full_mdl128``, ``simple_shading_full_mdl256``, ``simple_shading_full_mdl64``, ``single_camera``, ``state``
    * - Isaac-ExhaustPipe-GR1T2-Pink-IK-Abs-v0
      -
      - Manager Based
      -
      -
    * - Isaac-Factory-GearMesh-Direct-v0
      -
      - Direct
      - **rl_games** (PPO)
      -
    * - Isaac-Factory-NutThread-Direct-v0
      -
      - Direct
      - **rl_games** (PPO)
      -
    * - Isaac-Factory-PegInsert-Direct-v0
      -
      - Direct
      - **rl_games** (PPO)
      -
    * - Isaac-Forge-GearMesh-Direct-v0
      -
      - Direct
      - **rl_games** (PPO)
      -
    * - Isaac-Forge-NutThread-Direct-v0
      -
      - Direct
      - **rl_games** (PPO)
      -
    * - Isaac-Forge-PegInsert-Direct-v0
      -
      - Direct
      - **rl_games** (PPO)
      -
    * - Isaac-Franka-Cabinet-Direct-v0
      -
      - Direct
      - **rl_games** (PPO), **rsl_rl** (PPO), **skrl** (PPO)
      -
    * - Isaac-Humanoid-AMP-Dance-Direct-v0
      -
      - Direct
      - **skrl** (AMP)
      -
    * - Isaac-Humanoid-AMP-Run-Direct-v0
      -
      - Direct
      - **skrl** (AMP)
      -
    * - Isaac-Humanoid-AMP-Walk-Direct-v0
      -
      - Direct
      - **skrl** (AMP)
      -
    * - Isaac-Humanoid-Direct-Warp-v0
      -
      - Direct
      - **rl_games** (PPO), **rsl_rl** (PPO), **skrl** (PPO)
      -
    * - Isaac-Humanoid-Direct
      -
      - Direct
      - **rl_games** (PPO), **rsl_rl** (PPO), **skrl** (PPO)
      - **physics=** ``newton_mjwarp``, ``ovphysx``, ``physx``
    * - Isaac-Humanoid
      -
      - Manager Based
      - **rl_games** (PPO), **rsl_rl** (PPO), **skrl** (PPO), **sb3** (PPO)
      - **physics=** ``newton_mjwarp``, ``physx``
    * - Isaac-Lift-Cloth-Franka
      -
      - Manager Based
      - **rsl_rl** (PPO)
      - **physics=** ``newton_mjwarp_vbd``
    * - Isaac-Lift-Cube-Franka-IK-Abs-v0
      -
      - Manager Based
      -
      -
    * - Isaac-Lift-Cube-Franka-IK-Rel-v0
      -
      - Manager Based
      -
      -
    * - Isaac-Lift-Cube-Franka
      - Isaac-Lift-Cube-Franka-Play
      - Manager Based
      - **rl_games** (PPO), **rsl_rl** (PPO), **skrl** (PPO), **sb3** (PPO)
      -
    * - Isaac-Lift-Cube-OpenArm-v0
      - Isaac-Lift-Cube-OpenArm-Play-v0
      - Manager Based
      - **rl_games** (PPO), **rsl_rl** (PPO)
      -
    * - Isaac-Lift-Soft-Franka
      -
      - Manager Based
      - **rsl_rl** (PPO)
      - **physics=** ``newton_mjwarp_vbd``, ``physx``
    * - Isaac-Navigation-3DObstacles-ARL-Robot-1-v0
      - Isaac-Navigation-3DObstacles-ARL-Robot-1-Play-v0
      - Manager Based
      - **rl_games** (PPO), **rsl_rl** (PPO), **skrl** (PPO)
      -
    * - Isaac-Navigation-Flat-Anymal-C-v0
      - Isaac-Navigation-Flat-Anymal-C-Play-v0
      - Manager Based
      - **rsl_rl** (PPO), **skrl** (PPO)
      -
    * - Isaac-NutPour-GR1T2-Pink-IK-Abs-v0
      -
      - Manager Based
      -
      -
    * - Isaac-Open-Drawer-Franka-IK-Abs-v0
      -
      - Manager Based
      -
      -
    * - Isaac-Open-Drawer-Franka-IK-Rel-v0
      -
      - Manager Based
      -
      -
    * - Isaac-Open-Drawer-Franka-v0
      - Isaac-Open-Drawer-Franka-Play-v0
      - Manager Based
      - **rl_games** (PPO), **rsl_rl** (PPO), **skrl** (PPO)
      -
    * - Isaac-Open-Drawer-OpenArm-v0
      - Isaac-Open-Drawer-OpenArm-Play-v0
      - Manager Based
      - **rl_games** (PPO), **rsl_rl** (PPO)
      -
    * - Isaac-PickPlace-FixedBaseUpperBodyIK-G1-Abs-v0
      -
      - Manager Based
      -
      -
    * - Isaac-PickPlace-G1-InspireFTP-Abs-v0
      -
      - Manager Based
      -
      -
    * - Isaac-PickPlace-GR1T2-Abs-v0
      -
      - Manager Based
      -
      -
    * - Isaac-PickPlace-GR1T2-WaistEnabled-Abs-v0
      -
      - Manager Based
      -
      -
    * - Isaac-PickPlace-Locomanipulation-G1-Abs-v0
      -
      - Manager Based
      -
      -
    * - Isaac-Place-Mug-Agibot-Left-Arm-RmpFlow-v0
      -
      - Manager Based
      -
      -
    * - Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0
      -
      - Manager Based
      -
      -
    * - Isaac-Reach-Franka-IK-Abs-v0
      -
      - Manager Based
      -
      - **physics=** ``newton_mjwarp``, ``physx``
    * - Isaac-Reach-Franka-IK-Rel-v0
      -
      - Manager Based
      -
      - **physics=** ``newton_mjwarp``, ``physx``
    * - Isaac-Reach-Franka-OSC
      - Isaac-Reach-Franka-OSC-Play
      - Manager Based
      - **rsl_rl** (PPO)
      - **physics=** ``newton_mjwarp``, ``physx``
    * - Isaac-Reach-Franka
      - Isaac-Reach-Franka-Play
      - Manager Based
      - **rl_games** (PPO), **rsl_rl** (PPO), **skrl** (PPO)
      - **physics=** ``newton_mjwarp``, ``physx``
    * - Isaac-Reach-OpenArm-Bi-v0
      - Isaac-Reach-OpenArm-Bi-Play-v0
      - Manager Based
      - **rl_games** (PPO), **rsl_rl** (PPO)
      -
    * - Isaac-Reach-OpenArm-v0
      - Isaac-Reach-OpenArm-Play-v0
      - Manager Based
      - **rl_games** (PPO), **rsl_rl** (PPO), **skrl** (PPO)
      -
    * - Isaac-Reach-UR10
      - Isaac-Reach-UR10-Play
      - Manager Based
      - **rl_games** (PPO), **rsl_rl** (PPO), **skrl** (PPO)
      - **physics=** ``newton_mjwarp``, ``physx``
    * - Isaac-Repose-Cube-Allegro-Direct-Warp-v0
      -
      - Direct
      - **rl_games** (PPO), **rsl_rl** (PPO), **skrl** (PPO)
      -
    * - Isaac-Repose-Cube-Allegro-Direct-v0
      -
      - Direct
      - **rl_games** (PPO), **rsl_rl** (PPO), **skrl** (PPO)
      - **physics=** ``newton_mjwarp``, ``ovphysx``, ``physx``
    * - Isaac-Repose-Cube-Allegro-v0
      - Isaac-Repose-Cube-Allegro-Play-v0
      - Manager Based
      - **rl_games** (PPO), **rsl_rl** (PPO), **skrl** (PPO)
      -
    * - Isaac-Repose-Cube-Shadow-Direct-v0
      -
      - Direct
      - **rl_games** (PPO), **rsl_rl** (PPO), **skrl** (PPO)
      - **physics=** ``newton_mjwarp``, ``physx``
    * - Isaac-Repose-Cube-Shadow-OpenAI-FF-Direct-v0
      -
      - Direct
      - **rl_games** (PPO), **rsl_rl** (PPO), **skrl** (PPO)
      - **physics=** ``newton_mjwarp``, ``physx``
    * - Isaac-Repose-Cube-Shadow-OpenAI-LSTM-Direct-v0
      -
      - Direct
      - **rl_games** (PPO)
      - **physics=** ``newton_mjwarp``, ``physx``
    * - Isaac-Repose-Cube-Shadow-Vision-Direct-v0
      - Isaac-Repose-Cube-Shadow-Vision-Direct-Play-v0
      - Direct
      - **rl_games** (VISION), **rsl_rl** (PPO)
      - | **physics=** ``newton_mjwarp``, ``physx``
          | **renderer=** ``isaacsim_rtx_renderer``, ``newton_renderer``, ``ovrtx_renderer``
          | **presets=** ``albedo``, ``depth``, ``full``, ``rgb``, ``semantic_segmentation``, ``simple_shading_constant_diffuse``, ``simple_shading_diffuse_mdl``, ``simple_shading_full_mdl``
    * - Isaac-Shadow-Hand-Over-Direct-v0
      -
      - Direct
      - **rl_games** (PPO), **skrl** (PPO, IPPO, MAPPO)
      - **physics=** ``newton_mjwarp``, ``physx``
    * - Isaac-Stack-Cube-Bin-Franka-IK-Rel-Mimic-v0
      -
      - Manager Based
      -
      -
    * - Isaac-Stack-Cube-BlueGreen-Franka-IK-Rel-v0
      -
      - Manager Based
      -
      -
    * - Isaac-Stack-Cube-BlueGreenRed-Franka-IK-Rel-v0
      -
      - Manager Based
      -
      -
    * - Isaac-Stack-Cube-Franka-IK-Abs-v0
      -
      - Manager Based
      -
      -
    * - Isaac-Stack-Cube-Franka-IK-Rel-Blueprint-v0
      -
      - Manager Based
      -
      -
    * - Isaac-Stack-Cube-Franka-IK-Rel-Skillgen-v0
      -
      - Manager Based
      -
      -
    * - Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Cosmos-v0
      -
      - Manager Based
      -
      -
    * - Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-v0
      -
      - Manager Based
      -
      -
    * - Isaac-Stack-Cube-Franka-IK-Rel-v0
      -
      - Manager Based
      -
      -
    * - Isaac-Stack-Cube-Franka-v0
      -
      - Manager Based
      -
      -
    * - Isaac-Stack-Cube-Galbot-Left-Arm-Gripper-RmpFlow-v0
      -
      - Manager Based
      -
      -
    * - Isaac-Stack-Cube-Galbot-Left-Arm-Gripper-Visuomotor-v0
      -
      - Manager Based
      -
      -
    * - Isaac-Stack-Cube-Galbot-Right-Arm-Suction-RmpFlow-v0
      -
      - Manager Based
      -
      -
    * - Isaac-Stack-Cube-Instance-Randomize-Franka-IK-Rel-v0
      -
      - Manager Based
      -
      -
    * - Isaac-Stack-Cube-Instance-Randomize-Franka-v0
      -
      - Manager Based
      -
      -
    * - Isaac-Stack-Cube-RedGreen-Franka-IK-Rel-v0
      -
      - Manager Based
      -
      -
    * - Isaac-Stack-Cube-RedGreenBlue-Franka-IK-Rel-v0
      -
      - Manager Based
      -
      -
    * - Isaac-Stack-Cube-UR10-Long-Suction-IK-Rel-v0
      -
      - Manager Based
      -
      -
    * - Isaac-Stack-Cube-UR10-Short-Suction-IK-Rel-v0
      -
      - Manager Based
      -
      -
    * - Isaac-TrackPositionNoObstacles-ARL-Robot-1-v0
      - Isaac-TrackPositionNoObstacles-ARL-Robot-1-Play-v0
      - Manager Based
      - **rl_games** (PPO), **rsl_rl** (PPO), **skrl** (PPO)
      -
    * - Isaac-Tracking-LocoManip-Digit-v0
      - Isaac-Tracking-LocoManip-Digit-Play-v0
      - Manager Based
      - **rsl_rl** (PPO)
      - **physics=** ``newton_mjwarp``, ``physx``
    * - Isaac-Velocity-Flat-Anymal-B-v0
      - Isaac-Velocity-Flat-Anymal-B-Play-v0
      - Manager Based
      - **rsl_rl** (PPO), **skrl** (PPO)
      - **physics=** ``newton_mjwarp``, ``physx``
    * - Isaac-Velocity-Flat-Anymal-C-Direct-v0
      -
      - Direct
      - **rl_games** (PPO), **rsl_rl** (PPO), **skrl** (PPO)
      -
    * - Isaac-Velocity-Flat-Anymal-C-v0
      - Isaac-Velocity-Flat-Anymal-C-Play-v0
      - Manager Based
      - **rl_games** (PPO), **rsl_rl** (PPO), **skrl** (PPO)
      - **physics=** ``newton_mjwarp``, ``physx``
    * - Isaac-Velocity-Flat-Anymal-D-v0
      - Isaac-Velocity-Flat-Anymal-D-Play-v0
      - Manager Based
      - **rsl_rl** (PPO, DISTILLATION, DISTILLATION_RECURRENT, RECURRENT), **skrl** (PPO)
      - **physics=** ``newton_mjwarp``, ``ovphysx``, ``physx``
    * - Isaac-Velocity-Flat-Cassie-v0
      - Isaac-Velocity-Flat-Cassie-Play-v0
      - Manager Based
      - **rsl_rl** (PPO), **skrl** (PPO)
      - **physics=** ``newton_mjwarp``, ``physx``
    * - Isaac-Velocity-Flat-Digit-v0
      - Isaac-Velocity-Flat-Digit-Play-v0
      - Manager Based
      - **rsl_rl** (PPO)
      - **physics=** ``newton_mjwarp``, ``physx``
    * - Isaac-Velocity-Flat-G1-v0
      - Isaac-Velocity-Flat-G1-Play-v0
      - Manager Based
      - **rsl_rl** (PPO), **skrl** (PPO)
      - **physics=** ``newton_mjwarp``, ``physx``
    * - Isaac-Velocity-Flat-H1-v0
      - Isaac-Velocity-Flat-H1-Play-v0
      - Manager Based
      - **rsl_rl** (PPO), **skrl** (PPO)
      - **physics=** ``newton_mjwarp``, ``physx``
    * - Isaac-Velocity-Flat-Spot-v0
      - Isaac-Velocity-Flat-Spot-Play-v0
      - Manager Based
      - **rsl_rl** (PPO), **skrl** (PPO)
      - **physics=** ``newton_mjwarp``, ``physx``
    * - Isaac-Velocity-Flat-Unitree-A1-v0
      - Isaac-Velocity-Flat-Unitree-A1-Play-v0
      - Manager Based
      - **rsl_rl** (PPO), **skrl** (PPO), **sb3** (PPO)
      - **physics=** ``newton_mjwarp``, ``physx``
    * - Isaac-Velocity-Flat-Unitree-Go1-v0
      - Isaac-Velocity-Flat-Unitree-Go1-Play-v0
      - Manager Based
      - **rsl_rl** (PPO), **skrl** (PPO)
      - **physics=** ``newton_mjwarp``, ``physx``
    * - Isaac-Velocity-Flat-Unitree-Go2-v0
      - Isaac-Velocity-Flat-Unitree-Go2-Play-v0
      - Manager Based
      - **rsl_rl** (PPO), **skrl** (PPO)
      - **physics=** ``newton_mjwarp``, ``physx``
    * - Isaac-Velocity-Rough-Anymal-B-v0
      - Isaac-Velocity-Rough-Anymal-B-Play-v0
      - Manager Based
      - **rsl_rl** (PPO), **skrl** (PPO)
      - **physics=** ``newton_mjwarp``, ``physx``
    * - Isaac-Velocity-Rough-Anymal-C-Direct-v0
      -
      - Direct
      - **rl_games** (PPO), **rsl_rl** (PPO), **skrl** (PPO)
      -
    * - Isaac-Velocity-Rough-Anymal-C-v0
      - Isaac-Velocity-Rough-Anymal-C-Play-v0
      - Manager Based
      - **rl_games** (PPO), **rsl_rl** (PPO), **skrl** (PPO)
      - **physics=** ``newton_mjwarp``, ``physx``
    * - Isaac-Velocity-Rough-Anymal-D-v0
      - Isaac-Velocity-Rough-Anymal-D-Play-v0
      - Manager Based
      - **rsl_rl** (PPO), **skrl** (PPO)
      - **physics=** ``newton_mjwarp``, ``physx``
    * - Isaac-Velocity-Rough-Cassie-v0
      - Isaac-Velocity-Rough-Cassie-Play-v0
      - Manager Based
      - **rsl_rl** (PPO), **skrl** (PPO)
      - **physics=** ``newton_mjwarp``, ``physx``
    * - Isaac-Velocity-Rough-Digit-v0
      - Isaac-Velocity-Rough-Digit-Play-v0
      - Manager Based
      - **rsl_rl** (PPO)
      - **physics=** ``newton_mjwarp``, ``physx``
    * - Isaac-Velocity-Rough-G1-v0
      - Isaac-Velocity-Rough-G1-Play-v0
      - Manager Based
      - **rsl_rl** (PPO), **skrl** (PPO)
      - **physics=** ``newton_mjwarp``, ``physx``
    * - Isaac-Velocity-Rough-H1-v0
      - Isaac-Velocity-Rough-H1-Play-v0
      - Manager Based
      - **rsl_rl** (PPO), **skrl** (PPO)
      - **physics=** ``newton_mjwarp``, ``physx``
    * - Isaac-Velocity-Rough-Unitree-A1-v0
      - Isaac-Velocity-Rough-Unitree-A1-Play-v0
      - Manager Based
      - **rsl_rl** (PPO), **skrl** (PPO), **sb3** (PPO)
      - **physics=** ``newton_mjwarp``, ``physx``
    * - Isaac-Velocity-Rough-Unitree-Go1-v0
      - Isaac-Velocity-Rough-Unitree-Go1-Play-v0
      - Manager Based
      - **rsl_rl** (PPO), **skrl** (PPO)
      - **physics=** ``newton_mjwarp``, ``physx``
    * - Isaac-Velocity-Rough-Unitree-Go2-v0
      - Isaac-Velocity-Rough-Unitree-Go2-Play-v0
      - Manager Based
      - **rsl_rl** (PPO), **skrl** (PPO)
      - **physics=** ``newton_mjwarp``, ``physx``

.. END-AUTO-GENERATED: comprehensive-environment-list
