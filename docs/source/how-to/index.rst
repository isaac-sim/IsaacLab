.. _tutorials:
.. _how-to:

How-to Guides
=============

.. container:: guide-browser-intro

   Find step-by-step guides for building with Isaac Lab. Search by topic, then open a guide without leaving this page.

.. raw:: html

   <div class="guide-browser-toolbar" data-guide-browser>
     <label class="guide-search">
       <i class="fa-solid fa-magnifying-glass" aria-hidden="true"></i>
       <span class="visually-hidden">Search guides</span>
       <input type="search" data-guide-search placeholder="Search all guides" autocomplete="off">
     </label>
     <span class="guide-result-count" data-guide-count aria-live="polite"></span>
   </div>
   <p class="guide-empty-state" data-guide-empty hidden>No guides match this search.</p>
   <section class="guide-viewer" data-guide-viewer hidden aria-label="Selected guide">
     <div class="guide-viewer-header">
       <button type="button" class="guide-back-button" data-guide-back>
         <i class="fa-solid fa-arrow-left" aria-hidden="true"></i>
         Back to all guides
       </button>
       <a class="guide-open-page" data-guide-open-page target="_blank" rel="noopener">Open in a new tab</a>
     </div>
     <div class="guide-viewer-content">
       <p class="guide-loading" data-guide-loading role="status">Loading guide...</p>
       <iframe class="guide-frame" data-guide-frame title="Selected guide" hidden></iframe>
     </div>
   </section>

.. container:: guide-list

   .. container:: guide-group

      .. rubric:: Simulation Fundamentals

      .. container:: guide-entry

         :doc:`Creating an empty scene </source/tutorials/00_sim/create_empty>`

         Launch an empty simulation and learn the core startup sequence.

      .. container:: guide-entry

         :doc:`Spawning prims into the scene </source/tutorials/00_sim/spawn_prims>`

         Add lights, ground planes, and primitive shapes to a simulation stage.

      .. container:: guide-entry

         :doc:`Deep-dive into AppLauncher </source/tutorials/00_sim/launch_app>`

         Configure and launch simulation applications from Python and the command line.

   .. container:: guide-group

      .. rubric:: Assets

      .. container:: guide-entry

         :doc:`Adding a new robot to Isaac Lab </source/tutorials/01_assets/add_new_robot>`

         Bring a robot asset into Isaac Lab and define its articulation configuration.

      .. container:: guide-entry

         :doc:`Interacting with a rigid object </source/tutorials/01_assets/run_rigid_object>`

         Create, reset, and command a rigid object through the simulation API.

      .. container:: guide-entry

         :doc:`Interacting with an articulation </source/tutorials/01_assets/run_articulation>`

         Work with joint state, commands, and articulation data.

      .. container:: guide-entry

         :doc:`Interacting with a deformable object </source/tutorials/01_assets/run_deformable_object>`

         Spawn and manipulate deformable bodies in a scene.

      .. container:: guide-entry

         :doc:`Interacting with a surface gripper </source/tutorials/01_assets/run_surface_gripper>`

         Attach and release rigid objects with a surface gripper.

      .. container:: guide-entry

         :doc:`Importing a new asset </source/how-to/import_new_asset>`

         Convert URDF, MJCF, or mesh assets into USD for use in Isaac Lab.

      .. container:: guide-entry

         :doc:`Writing an asset configuration </source/how-to/write_articulation_cfg>`

         Turn an imported robot into a reusable articulation configuration.

      .. container:: guide-entry

         :doc:`Robot configurations </source/how-to/robots>`

         Understand the structure and conventions of supported robot configurations.

      .. container:: guide-entry

         :doc:`Making a physics prim fixed </source/how-to/make_fixed_prim>`

         Convert a floating asset into a fixed object in the simulation.

      .. container:: guide-entry

         :doc:`Spawning multiple assets </source/how-to/multi_asset_spawning>`

         Batch rigid objects and vary asset configurations across environments.

   .. container:: guide-group

      .. rubric:: Scenes and Cloning

      .. container:: guide-entry

         :doc:`Using the interactive scene </source/tutorials/02_scene/create_scene>`

         Compose assets and sensors with the higher-level interactive scene interface.

      .. container:: guide-entry

         :doc:`Cloning environments </source/how-to/cloning>`

         Choose cloning strategies, build heterogeneous scenes, and filter collisions.

   .. container:: guide-group

      .. rubric:: Environments and Training

      .. container:: guide-entry

         :doc:`Creating a manager-based base environment </source/tutorials/03_envs/create_manager_base_env>`

         Build a non-RL environment from reusable manager terms.

      .. container:: guide-entry

         :doc:`Creating a manager-based RL environment </source/tutorials/03_envs/create_manager_rl_env>`

         Add rewards, terminations, curricula, and commands for reinforcement learning.

      .. container:: guide-entry

         :doc:`Creating a direct workflow RL environment </source/tutorials/03_envs/create_direct_rl_env>`

         Implement an RL task with direct control over the environment loop.

      .. container:: guide-entry

         :doc:`Registering an environment </source/tutorials/03_envs/register_rl_env_gym>`

         Register an Isaac Lab task with Gymnasium and expose its configurations.

      .. container:: guide-entry

         :doc:`Training with an RL agent </source/tutorials/03_envs/run_rl_training>`

         Launch training and inference with a supported reinforcement learning library.

      .. container:: guide-entry

         :doc:`Configuring an RL agent </source/tutorials/03_envs/configuring_rl_training>`

         Customize agent settings and training hyperparameters.

      .. container:: guide-entry

         :doc:`Modifying an existing direct RL environment </source/tutorials/03_envs/modify_direct_rl_env>`

         Extend and adjust a direct workflow task without rebuilding it from scratch.

      .. container:: guide-entry

         :doc:`Policy inference in a USD environment </source/tutorials/03_envs/policy_inference_in_usd>`

         Run a trained policy against an environment defined in a USD stage.

      .. container:: guide-entry

         :doc:`Wrapping environments </source/how-to/wrap_rl_env>`

         Adapt Isaac Lab environments to external reinforcement learning interfaces.

      .. container:: guide-entry

         :doc:`Adding your own learning library </source/how-to/add_own_library>`

         Integrate an additional learning framework with Isaac Lab tasks.

      .. container:: guide-entry

         :doc:`Running scripted state machines </source/how-to/run_state_machines>`

         Drive environments with deterministic state-machine policies.

      .. container:: guide-entry

         :doc:`Curriculum utilities </source/how-to/curriculums>`

         Change environment parameters dynamically during training.

      .. container:: guide-entry

         :doc:`Transferring policies between PhysX and Newton </source/how-to/transfer_policies_between_physx_and_newton>`

         Validate and evaluate policies across the supported physics backends.

   .. container:: guide-group

      .. rubric:: Sensors, Cameras, and Rendering

      .. container:: guide-entry

         :doc:`Adding sensors on a robot </source/tutorials/04_sensors/add_sensors_on_robot>`

         Add camera, ray-caster, and contact sensors to an environment.

      .. container:: guide-entry

         :doc:`Saving rendered images and 3D re-projection </source/how-to/save_camera_output>`

         Save camera outputs and reconstruct point clouds from depth images.

      .. container:: guide-entry

         :doc:`Finding how many cameras to train with </source/how-to/estimate_how_many_cameras_can_run>`

         Estimate camera throughput and memory limits for your hardware.

      .. container:: guide-entry

         :doc:`Configuring RTX rendering settings </source/how-to/configure_rendering>`

         Tune RTX rendering quality and performance options.

      .. container:: guide-entry

         :doc:`Capturing sensor frames during training </source/how-to/capture_sensor_frames>`

         Record selected sensor outputs from a running training job.

   .. container:: guide-group

      .. rubric:: Controllers

      .. container:: guide-entry

         :doc:`Using a task-space controller </source/tutorials/05_controllers/run_diff_ik>`

         Control a robot end effector with differential inverse kinematics.

      .. container:: guide-entry

         :doc:`Using an operational space controller </source/tutorials/05_controllers/run_osc>`

         Apply operational-space control to a robot manipulator.

   .. container:: guide-group

      .. rubric:: Simulation and Data

      .. container:: guide-entry

         :doc:`Working with ProxyArray </source/how-to/proxy_array>`

         Read and write simulation state through dual NumPy and Warp access.

      .. container:: guide-entry

         :doc:`Simulation performance </source/how-to/simulation_performance>`

         Diagnose bottlenecks and improve simulation throughput.

   .. container:: guide-group

      .. rubric:: Teleoperation

      .. container:: guide-entry

         :doc:`Setting up Isaac Teleop with CloudXR </source/how-to/cloudxr_teleoperation>`

         Connect XR devices through CloudXR for immersive teleoperation.

      .. container:: guide-entry

         :doc:`Setting up Haply teleoperation </source/how-to/haply_teleoperation>`

         Use Haply devices for robot control with directional force feedback.

   .. container:: guide-group

      .. rubric:: Tools and Workflows

      .. container:: guide-entry

         :doc:`Recording animations of simulations </source/how-to/record_animation>`

         Capture simulation state and export an animation.

      .. container:: guide-entry

         :doc:`Mastering Omniverse for robotics </source/how-to/master_omniverse>`

         Find Omniverse workflows and resources relevant to Isaac Lab.

      .. container:: guide-entry

         :doc:`Profiling Isaac Lab with Nsight Systems </source/how-to/profile_with_nsys>`

         Capture and inspect runtime traces with NVIDIA Nsight Systems.

.. container:: guide-browser-note

   .. note::

      This collection is a work in progress. If a question is not answered here, open an issue on the
      `Isaac Lab GitHub repository <https://github.com/isaac-sim/IsaacLab>`_.
