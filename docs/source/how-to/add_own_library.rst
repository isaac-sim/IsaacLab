Adding your own learning library
================================

Isaac Lab comes pre-integrated with a number of libraries (such as RSL-RL, RL-Games, SKRL, Stable Baselines, etc.).
However, you may want to integrate your own library with Isaac Lab or use a different version of the libraries than
the one installed by Isaac Lab. This is possible as long as the library is available as Python package that supports
the Python version used by the underlying simulator. For instance, if you are using Isaac Sim 4.0.0 onwards, you need
to ensure that the library is available for Python 3.10.

Using a different version of a library
--------------------------------------

If you want to use a different version of a library than the one installed by Isaac Lab, you can install the library
by building it from source or using a different version of the library available on PyPI.

For instance, if you want to use your own modified version of the `rsl-rl`_ library, you can follow these steps:

1. Follow the instructions for installing Isaac Lab. This will install the default version of the ``rsl-rl`` library.
2. Clone the ``rsl-rl`` library from the GitHub repository:

   .. code-block:: bash

     git clone git@github.com:leggedrobotics/rsl_rl.git


3. Install the library in your Python environment:

   .. code-block:: bash

     # Assuming you are in the root directory of the Isaac Lab repository
     cd IsaacLab

     # Note: If you are using a virtual environment, make sure to activate it before running the following command
     ./isaaclab.sh -p -m pip install -e /path/to/rsl_rl

In this case, the ``rsl-rl`` library will be installed in the Python environment used by Isaac Lab. You can now use the
``rsl-rl`` library in your experiments. To check the library version and other details, you can use the following
command:

.. code-block:: bash

  ./isaaclab.sh -p -m pip show rsl-rl

This should now show the location of the ``rsl-rl`` library as the directory where you cloned the library.
For instance, if you cloned the library to ``/home/user/git/rsl_rl``, the output of the above command should be:

.. code-block:: bash

  Name: rsl_rl
  Version: 2.0.2
  Summary: Fast and simple RL algorithms implemented in pytorch
  Home-page: https://github.com/leggedrobotics/rsl_rl
  Author: ETH Zurich, NVIDIA CORPORATION
  Author-email:
  License: BSD-3
  Location: /home/user/git/rsl_rl
  Requires: torch, torchvision, numpy, GitPython, onnx
  Required-by:


Integrating a new library
-------------------------

Adding a new library to Isaac Lab is similar to using a different version of a library. You can install the library
in your Python environment and use it in your experiments. However, if you want to integrate the library with
Isaac Lab, you can will first need to make a wrapper for the library, as explained in
:ref:`how-to-env-wrappers`.

The following steps can be followed to integrate a new library with Isaac Lab:

1. Add your library as an extra-dependency in the ``setup.py`` for the extension ``isaaclab_tasks``.
   This will ensure that the library is installed when you install Isaac Lab or it will complain if the library is not
   installed or available.
2. Install your library in the Python environment used by Isaac Lab. You can do this by following the steps mentioned
   in the previous section.
3. Create a wrapper for the library. You can check the module :mod:`isaaclab_rl`
   for examples of wrappers for different libraries. You can create a new wrapper for your library and add it to the
   module. You can also create a new module for the wrapper if you prefer.
4. Create workflow scripts for your library to train and evaluate agents. You can check the existing workflow scripts
   in the ``scripts/reinforcement_learning`` directory for examples. You can create new workflow
   scripts for your library and add them to the directory.

Optionally, you can also add some tests and documentation for the wrapper. This will help ensure that the wrapper
works as expected and can guide users on how to use the wrapper.

* Add some tests to ensure that the wrapper works as expected and remains compatible with the library.
  These tests can be added to the ``source/isaaclab_rl/test`` directory.
* Add some documentation for the wrapper. You can add the API documentation to the
  ``docs/source/api/lab_tasks/isaaclab_rl.rst`` file.

.. _rsl-rl: https://github.com/leggedrobotics/rsl_rl
