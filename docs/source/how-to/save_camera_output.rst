.. _how-to-save-images-and-3d-reprojection:


Saving rendered images and 3D re-projection
===========================================

.. currentmodule:: omni.isaac.orbit

This how-to demonstrates an efficient saving of rendered images and the projection of depth images into 3D Space.

It is accompanied with the ``run_usd_camera.py`` script in the ``orbit/source/standalone/tutorials/04_sensors``
directory. For an introduction to sensors, please check the :ref:`tutorial-add-sensors-on-robot` tutorials.


Saving the Images
-----------------

To save the images, we use the basic write class from Omniverse Replicator. This class allows us to save the
images in a numpy format. For more information on the basic writer, please check the
`documentation <https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/writer_examples.html>`_.

.. literalinclude:: ../../../source/standalone/tutorials/04_sensors/run_usd_camera.py
   :language: python
   :lines: 135-137
   :linenos:
   :lineno-start: 135

While stepping the simulator, the images can be saved to the defined folder.
Since the BasicWriter only supports saving data using NumPy format, we first need to convert the PyTorch sensors
to NumPy arrays before packing them in a dictionary.

.. literalinclude:: ../../../source/standalone/tutorials/04_sensors/run_usd_camera.py
   :language: python
   :lines: 172-193
   :linenos:
   :lineno-start: 172

Projection into 3D Space
------------------------

In addition, we provide utilities to project the depth image into 3D Space.
The re-projection operations are done using torch which allows us to use the GPU for faster computation.
The resulting point cloud is visualized using the :mod:`omni.isaac.debug_draw` extension from Isaac Sim.

.. literalinclude:: ../../../source/standalone/tutorials/04_sensors/run_usd_camera.py
   :language: python
   :lines: 197-229
   :linenos:
   :lineno-start: 197
