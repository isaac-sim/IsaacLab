Creating Markers in Orbit
=========================

In this tutorial, we will explore how to create different types of markers in `Orbit` using a Python script.
The script demonstrates the creation of markers with various shapes and visual properties.

Please ensure you have gone through the previous tutorials, especially creating an empty scene for a foundational understanding.


The Code
~~~~~~~~

The tutorial corresponds to the ``markers.py`` script in the ``orbit/source/standalone/demos`` directory.
Let's take a look at the Python script:

.. literalinclude:: ../../../../source/standalone/demos/markers.py
   :language: python
   :linenos:

The Code Explained
~~~~~~~~~~~~~~~~~~

Creating and spawning markers
-----------------------------

The :function:`spawn_markers` function creates different types of markers with specified configurations.
For example, we include frames, arrows, cubes, spheres, cylinders, cones, and meshes.
The function returns a :obj:`VisualizationMarkers` object.

.. literalinclude:: ../../../../source/standalone/demos/markers.py
   :language: python
   :lines: 37-84
   :linenos:
   :lineno-start: 37

Main simulation logic
---------------------

The `main` function sets up the simulation context, camera view, and spawns lights into the stage.
It then creates instances of the markers and places them in a grid pattern.
The markers are rotated around the z-axis during the simulation for visualization purposes.

.. literalinclude:: ../../../../source/standalone/demos/markers.py
   :language: python
   :lines: 86-111
   :linenos:
   :lineno-start: 86

Executing the Script
~~~~~~~~~~~~~~~~~~~~

To run the script, execute the following command:

.. code-block:: bash

  ./orbit.sh -p source/standalone/demos/markers.py


The simulation should start, and you can observe the different types of markers arranged in a grid pattern.
To stop the simulation, close the window, press the ``STOP`` button in the UI, or use ``Ctrl+C`` in the terminal.

This tutorial provides a foundation for working with markers in Orbit.
You can further customize markers by adjusting their configurations and exploring additional options
available in the Orbit API.
