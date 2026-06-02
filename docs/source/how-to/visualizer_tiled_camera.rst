.. _how-to-visualizer-tiled-camera:

Using Visualizer Tiled Cameras
==============================

.. currentmodule:: isaaclab

This guide is accompanied by the ``run_tiled_camera_visualizer.py`` script in the
``IsaacLab/scripts/tutorials/07_visualizers`` directory.

For general visualizer documentation, see :doc:`/source/overview/core-concepts/visualization`.

The visualizer tiled camera view is a live monitoring and debugging tool. It opens a
non-interactive image panel in the Kit or Newton visualizer and shows one RGB tile per
selected environment. This is separate from camera observations used by policies.

This guide shows two tiled camera use cases: generated cameras following moving Anymal-D
robots in Kit, and existing wrist-mounted robot cameras shown in Newton.

.. dropdown:: Code for run_tiled_camera_visualizer.py
   :icon: code

   .. literalinclude:: ../../../scripts/tutorials/07_visualizers/run_tiled_camera_visualizer.py
      :language: python
      :emphasize-lines: 79-91
      :linenos:


Kit visualizer
--------------

The Kit visualizer uses an Omniverse viewport for the tiled camera panel. The script
automatically enables camera support when ``--viz kit`` is selected.

.. code-block:: bash

   ./isaaclab.sh -p scripts/tutorials/07_visualizers/run_tiled_camera_visualizer.py \
     --task Isaac-Velocity-Rough-Anymal-D-Play-v0 \
     --num_envs 64 \
     --viz kit \
     --enable_cameras

For this example, the script creates generated cameras that follow the base link of each
Anymal-D robot. To change the number of tiles, followed prim, or camera offset, edit
the highlighted ``KitVisualizerCfg`` fields in the script.

To display the tiled camera panel, select the ``Visualizer Tiled Camera`` viewport from
the viewport selection menu.

.. figure:: ../_static/visualizers/tiled_camera_kit_anymal_activate.jpg
   :width: 100%
   :alt: Kit visualizer tiled camera panel for Anymal-D robots

   Kit visualizer with generated tiled cameras following Anymal-D robots. The annotated
   circle and arrow should highlight the viewport selection menu used to activate the
   ``Visualizer Tiled Camera`` panel.


Newton visualizer
-----------------

The Newton visualizer uses a lightweight OpenGL window. This example uses the Dexsuite
Kuka-Allegro lift environment with its wrist-mounted camera, which gives a close-up view
of the hand and object from each selected environment:

.. code-block:: bash

   ./isaaclab.sh -p scripts/tutorials/07_visualizers/run_tiled_camera_visualizer.py \
     --task Isaac-Dexsuite-Kuka-Allegro-Lift-Play-v0 \
     --num_envs 64 \
     --viz newton \
     presets=duo_camera,rgb128,newton_renderer

For this example, the script maps the tiled camera panel to the existing wrist camera
sensor at ``/World/envs/env_.*/Robot/ee_link/palm_link/Camera``. To change the number of
tiles or use a different existing camera, edit the highlighted ``NewtonVisualizerCfg``
fields in the script.

To show or hide the tiled camera panel, use the ``Visualizer Tiled Camera`` entry in the
Tiled Camera View dropdown in the left sidebar.

.. figure:: ../_static/visualizers/tiled_camera_newton_kuka_wrist_activate.jpg
   :width: 100%
   :alt: Newton visualizer tiled camera panel for Kuka-Allegro wrist cameras

   Newton visualizer with existing wrist-mounted Kuka-Allegro cameras in the tiled panel.
   The annotated circle and arrow should highlight the Tiled Camera View dropdown used to
   activate the ``Visualizer Tiled Camera`` panel.


Configuration notes
-------------------

To customize tiled camera behavior, edit the highlighted ``VisualizerCfg`` fields in
``run_tiled_camera_visualizer.py``:

* For generated cameras, ``tiled_cam_target_prim_path`` chooses the followed prim and
  ``tiled_cam_eye`` sets the camera offset from that prim.
* For existing scene cameras, ``tiled_cam_prim_path`` must match an Isaac Lab
  :class:`~isaaclab.sensors.Camera` sensor in the selected task.
* ``tiled_cam_num`` controls how many environment tiles are shown.


Troubleshooting
---------------

* If a generated view fails with a missing prim error, check that
  ``tiled_cam_target_prim_path`` resolves in each selected environment. Common template
  forms include ``/World/envs/*/...`` and ``/World/envs/env_.*/...``.
* If an existing-camera view reports that no Isaac Lab camera owns the prim, check that
  ``tiled_cam_prim_path`` matches a :class:`~isaaclab.sensors.Camera` sensor in the task.
* If ``rerun`` or ``viser`` is selected, use ``--viz kit`` or ``--viz newton`` instead.
  The tiled camera panel is currently implemented for Kit and Newton.
* If the view is too expensive, reduce ``tiled_cam_num``, ``--num_envs``, or the camera
  resolution. The visualizer caps the tiled panel at 100 tiles.


See also
--------

* :doc:`/source/overview/core-concepts/visualization` - visualizer configuration and UI controls.
* :doc:`/source/how-to/configure_rendering` - selecting rendering presets and quality modes.
