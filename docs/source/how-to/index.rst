.. _how-to:

How-to Guides
=============

This section includes guides that help you use Isaac Lab. These are intended for users who
have already worked through the tutorials and are looking for more information on how to
use Isaac Lab. If you are new to Isaac Lab, we recommend you start with the tutorials.

.. note::

    This section is a work in progress. If you have a question that is not answered here,
    please open an issue on our `GitHub page <https://github.com/isaac-sim/IsaacLab>`_.

Importing a New Asset
---------------------

Importing an asset into Isaac Lab is a common task. It contains two steps: importing the asset into
a USD format and then setting up the configuration object for the asset. The following guide explains
how to import a new asset into Isaac Lab.

.. toctree::
    :maxdepth: 1

    import_new_asset
    write_articulation_cfg
    robots

Creating a Fixed Asset
----------------------

Often you may want to create a fixed asset in your scene. For instance, making a floating base robot
a fixed base robot. This guide goes over the various considerations and steps to create a fixed asset.

.. toctree::
    :maxdepth: 1

    make_fixed_prim

Spawning Multiple Assets
------------------------

This guide explains how to batch rigid objects into a collection and configure different asset
variants across environments.

.. toctree::
    :maxdepth: 1

    multi_asset_spawning

Cloning Environments
--------------------

This guide explains how Isaac Lab's template-based cloning system works, including
cloning strategies, heterogeneous environments, and collision filtering.

.. toctree::
    :maxdepth: 1

    cloning

Saving Camera Output
--------------------

This guide explains how to save the camera output in Isaac Lab.

.. toctree::
    :maxdepth: 1

    save_camera_output

Estimate How Many Cameras Can Run On Your Machine
-------------------------------------------------

This guide demonstrates how to estimate the number of cameras one can run on their machine under the desired parameters.

.. toctree::
    :maxdepth: 1

    estimate_how_many_cameras_can_run

Configure Rendering
-------------------

This guide demonstrates how to customize the RTX rendering settings.

.. toctree::
    :maxdepth: 1

    configure_rendering


Working with Simulation Data
----------------------------

This guide explains how to consume asset and sensor data through the Torch and Warp representations
provided by :class:`~isaaclab.utils.warp.ProxyArray`, and how to keep retained views valid.

.. toctree::
    :maxdepth: 1

    proxy_array


Interfacing with Environments
-----------------------------

These guides explain how to interface with reinforcement learning environments in Isaac Lab.

.. toctree::
    :maxdepth: 1

    wrap_rl_env
    add_own_library
    run_state_machines


Transferring Policies Between Physics Backends
-----------------------------------------------

This guide explains how to validate and evaluate policies trained in PhysX and deployed in Newton,
and policies trained in Newton and deployed in PhysX.

.. toctree::
    :maxdepth: 1

    transfer_policies_between_physx_and_newton


Working with Physics Backends
-----------------------------

These guides help prepare assets and tasks for the supported physics backends.

.. toctree::
    :maxdepth: 1

    prepare_asset_for_newton
    /source/how-to/native_physics_api/index

For experimental Newton solver and Warp-environment workflows, see
:ref:`newton-using-vbd`, :ref:`newton-using-mpm`, :ref:`newton-using-cables`,
:ref:`warp-environments`, and :ref:`warp-env-migration`.

.. toctree::
    :hidden:

    /source/overview/core-concepts/physical-backends/joint_and_body_ordering
    /source/overview/core-concepts/physical-backends/newton/using-vbd-solver
    /source/overview/core-concepts/physical-backends/newton/using-mpm
    /source/overview/core-concepts/physical-backends/newton/using-cables
    /source/overview/core-concepts/physical-backends/newton/warp-environments
    /source/overview/core-concepts/physical-backends/newton/warp-env-migration

Solver Tuning
-------------

These guides diagnose and tune solver-specific behavior after backend, task,
and asset validation.

.. toctree::
   :maxdepth: 1

   solver_tuning/index

Recording an Animation and Video
--------------------------------

This guide explains how to record an animation and capture sensor frames in Isaac Lab. For
recording training video, see :doc:`/source/features/record_video`.

.. toctree::
    :maxdepth: 1

    record_animation
    capture_sensor_frames


Dynamically Modifying Environment Parameters With CurriculumTerm
----------------------------------------------------------------

This guide explains how to dynamically modify environment parameters during training in Isaac Lab.
It covers the use of curriculum utilities to change environment parameters at runtime.

.. toctree::
    :maxdepth: 1

    curriculums


Mastering Omniverse
-------------------

Omniverse is a powerful platform that provides a wide range of features. This guide links to
additional resources that help you use Omniverse features in Isaac Lab.

.. toctree::
    :maxdepth: 1

    master_omniverse


Setting up Isaac Teleop with CloudXR
------------------------------------

This guide explains how to install Isaac Teleop, start the CloudXR runtime, and connect XR
devices for immersive teleoperation in Isaac Lab.

.. toctree::
    :maxdepth: 1

    cloudxr_teleoperation


Setting up Haply Teleoperation
------------------------------

This guide explains how to use Haply Inverse3 and VerseGrip devices for robot teleoperation
with directional force feedback in Isaac Lab.

.. toctree::
    :maxdepth: 1

    haply_teleoperation


Profiling Isaac Lab with Nsight Systems
---------------------------------------

This guide explains how to profile Isaac Lab tasks with NVIDIA Nsight Systems for runtime performance analysis.

.. toctree::
    :maxdepth: 1

    profile_with_nsys
