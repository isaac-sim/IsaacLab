.. _how-to-save-images-and-3d-reprojection:


Saving rendered images and 3D re-projection
===========================================

.. currentmodule:: omni.isaac.orbit

This guide accompanied with the ``run_usd_camera.py`` script in the ``orbit/source/standalone/tutorials/04_sensors``
directory.

.. dropdown:: Code for run_usd_camera.py
   :icon: code

   .. literalinclude:: ../../../source/standalone/tutorials/04_sensors/run_usd_camera.py
      :language: python
      :emphasize-lines: 137-139, 172-196, 200-204, 214-232
      :linenos:


Saving using Replicator Basic Writer
------------------------------------

To save camera outputs, we use the basic write class from Omniverse Replicator. This class allows us to save the
images in a numpy format. For more information on the basic writer, please check the
`documentation <https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/writer_examples.html>`_.

.. literalinclude:: ../../../source/standalone/tutorials/04_sensors/run_usd_camera.py
   :language: python
   :lines: 137-139
   :dedent:

While stepping the simulator, the images can be saved to the defined folder. Since the BasicWriter only supports
saving data using NumPy format, we first need to convert the PyTorch sensors to NumPy arrays before packing
them in a dictionary.

.. literalinclude:: ../../../source/standalone/tutorials/04_sensors/run_usd_camera.py
   :language: python
   :lines: 172-192
   :dedent:

After this step, we can save the images using the BasicWriter.

.. literalinclude:: ../../../source/standalone/tutorials/04_sensors/run_usd_camera.py
   :language: python
   :lines: 193-196
   :dedent:


Projection into 3D Space
------------------------

We include utilities to project the depth image into 3D Space. The re-projection operations are done using
PyTorch operations which allows faster computation.

.. literalinclude:: ../../../source/standalone/tutorials/04_sensors/run_usd_camera.py
   :language: python
   :lines: 200-204
   :dedent:

The resulting point cloud can be visualized using the :mod:`omni.isaac.debug_draw` extension from Isaac Sim.
This makes it easy to visualize the point cloud in the 3D space.

.. literalinclude:: ../../../source/standalone/tutorials/04_sensors/run_usd_camera.py
   :language: python
   :lines: 214-232
   :dedent:


Executing the script
--------------------

To run the accompanying script, execute the following command:

.. code-block:: bash

  ./orbit.sh -p source/standalone/tutorials/04_sensors/run_usd_camera.py --save --draw


The simulation should start, and you can observe different objects falling down. An output folder will be created
in the ``orbit/source/standalone/tutorials/04_sensors`` directory, where the images will be saved. Additionally,
you should see the point cloud in the 3D space drawn on the viewport.

To stop the simulation, close the window, press the ``STOP`` button in the UI, or use ``Ctrl+C`` in the terminal.
