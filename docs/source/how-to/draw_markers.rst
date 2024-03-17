Creating Visualization Markers
==============================

.. currentmodule:: omni.isaac.orbit

Visualization markers are useful to debug the state of the environment. They can be used to visualize
the frames, commands, and other information in the simulation.

While Isaac Sim provides its own :mod:`omni.isaac.debug_draw` extension, it is limited to rendering only
points, lines and splines. For cases, where you need to render more complex shapes, you can use the
:class:`markers.VisualizationMarkers` class.

This guide is accompanied by a sample script ``markers.py`` in the ``orbit/source/standalone/demos`` directory.

.. dropdown:: Code for markers.py
   :icon: code

   .. literalinclude:: ../../../source/standalone/demos/markers.py
      :language: python
      :emphasize-lines: 49-97, 112-113, 142-148
      :linenos:



Configuring the markers
-----------------------

The :class:`~markers.VisualizationMarkersCfg` class provides a simple interface to configure
different types of markers. It takes in the following parameters:

- :attr:`~markers.VisualizationMarkersCfg.prim_path`: The corresponding prim path for the marker class.
- :attr:`~markers.VisualizationMarkersCfg.markers`: A dictionary specifying the different marker prototypes
  handled by the class. The key is the name of the marker prototype and the value is its spawn configuration.

.. note::

   In case the marker prototype specifies a configuration with physics properties, these are removed.
   This is because the markers are not meant to be simulated.

Here we show all the different types of markers that can be configured. These range from simple shapes like
cones and spheres to more complex geometries like a frame or arrows. The marker prototypes can also be
configured from USD files.

.. literalinclude:: ../../../source/standalone/demos/markers.py
   :language: python
   :lines: 49-97
   :dedent:


Drawing the markers
-------------------

To draw the markers, we call the :class:`~markers.VisualizationMarkers.visualize` method. This method takes in
as arguments the pose of the markers and the corresponding marker prototypes to draw.

.. literalinclude:: ../../../source/standalone/demos/markers.py
   :language: python
   :lines: 142-148
   :dedent:


Executing the Script
--------------------

To run the accompanying script, execute the following command:

.. code-block:: bash

  ./orbit.sh -p source/standalone/demos/markers.py

The simulation should start, and you can observe the different types of markers arranged in a grid pattern.
The markers will rotating around their respective axes. Additionally every few rotations, they will
roll forward on the grid.

To stop the simulation, close the window, or use ``Ctrl+C`` in the terminal.
