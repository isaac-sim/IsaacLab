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

Creating a Fixed Asset
----------------------

Often you may want to create a fixed asset in your scene. For instance, making a floating base robot
a fixed base robot. This guide goes over the various considerations and steps to create a fixed asset.

.. toctree::
    :maxdepth: 1

    make_fixed_prim

Spawning Multiple Assets
------------------------

This guide explains how to import and configure different assets in each environment. This is
useful when you want to create diverse environments with different objects.

.. toctree::
    :maxdepth: 1

    multi_asset_spawning

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


Drawing Markers
---------------

This guide explains how to use the :class:`~isaaclab.markers.VisualizationMarkers` class to draw markers in
Isaac Lab.

.. toctree::
    :maxdepth: 1

    draw_markers


Interfacing with Environments
-----------------------------

These guides explain how to interface with reinforcement learning environments in Isaac Lab.

.. toctree::
    :maxdepth: 1

    wrap_rl_env
    add_own_library


Recording an Animation and Video
--------------------------------

This guide explains how to record an animation and video in Isaac Lab.

.. toctree::
    :maxdepth: 1

    record_animation
    record_video

Mastering Omniverse
-------------------

Omniverse is a powerful platform that provides a wide range of features. This guide links to
additional resources that help you use Omniverse features in Isaac Lab.

.. toctree::
    :maxdepth: 1

    master_omniverse
