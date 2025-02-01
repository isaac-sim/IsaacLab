Extension Development
=======================

Everything in Omniverse is either an extension, or a collection of extensions (an application). They are
modularized packages that form the atoms of the Omniverse ecosystem. Each extension
provides a set of functionalities that can be used by other extensions or
standalone applications. A folder is recognized as an extension if it contains
an ``extension.toml`` file in the ``config`` directory. More information on extensions can be found in the
`Omniverse documentation <https://docs.omniverse.nvidia.com/kit/docs/kit-manual/latest/guide/extensions_basic.html>`__.

Each extension in Isaac Lab is written as a python package and follows the following structure:

.. code:: bash

   <extension-name>
   ├── config
   │   └── extension.toml
   ├── docs
   │   ├── CHANGELOG.md
   │   └── README.md
   ├── <extension-name>
   │   ├── __init__.py
   │   ├── ....
   │   └── scripts
   ├── setup.py
   └── tests

The ``config/extension.toml`` file contains the metadata of the extension. This
includes the name, version, description, dependencies, etc. This information is used
by the Omniverse API to load the extension. The ``docs`` directory contains the documentation
for the extension with more detailed information about the extension and a CHANGELOG
file that contains the changes made to the extension in each version.

The ``<extension-name>`` directory contains the main python package for the extension.
It may also contains the ``scripts`` directory for keeping python-based applications
that are loaded into Omniverse when then extension is enabled using the
`Extension Manager <https://docs.omniverse.nvidia.com/kit/docs/kit-manual/latest/guide/extensions_basic.html>`__.

More specifically, when an extension is enabled, the python module specified in the
``config/extension.toml`` file is loaded and scripts that contains children of the
:class:`omni.ext.IExt` class are executed.

.. code:: python

   import omni.ext

   class MyExt(omni.ext.IExt):
      """My extension application."""

      def on_startup(self, ext_id):
         """Called when the extension is loaded."""
         pass

      def on_shutdown(self):
         """Called when the extension is unloaded.

         It releases all references to the extension and cleans up any resources.
         """
         pass

While loading extensions into Omniverse happens automatically, using the python package
in standalone applications requires additional steps. To simplify the build process and
avoiding the need to understand the `premake <https://premake.github.io/>`__
build system used by Omniverse, we directly use the `setuptools <https://setuptools.readthedocs.io/en/latest/>`__
python package to build the python module provided by the extensions. This is done by the
``setup.py`` file in the extension directory.

.. note::

   The ``setup.py`` file is not required for extensions that are only loaded into Omniverse
   using the `Extension Manager <https://docs.omniverse.nvidia.com/prod_extensions/prod_extensions/ext_extension-manager.html>`__.

Lastly, the ``tests`` directory contains the unit tests for the extension. These are written
using the `unittest <https://docs.python.org/3/library/unittest.html>`__ framework. It is
important to note that Omniverse also provides a similar
`testing framework <https://docs.omniverse.nvidia.com/kit/docs/kit-manual/104.0/guide/testing_exts_python.html>`__.
However, it requires going through the build process and does not support testing of the python module in
standalone applications.

Custom Extension Dependency Management
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Certain extensions may have dependencies which require the installation of additional packages before the extension
can be used. While Python dependencies are handled by the `setuptools <https://setuptools.readthedocs.io/en/latest/>`__
package and specified in the ``setup.py`` file, non-Python dependencies such as `ROS <https://www.ros.org/>`__
packages or `apt <https://en.wikipedia.org/wiki/APT_(software)>`__ packages are not handled by setuptools.
Handling these kinds of dependencies requires an additional procedure.

There are two types of dependencies that can be specified in the ``extension.toml`` file
under the ``isaac_lab_settings`` section:

1. **apt_deps**: A list of apt packages that need to be installed. These are installed using the
   `apt <https://ubuntu.com/server/docs/package-management>`__ package manager.
2. **ros_ws**: The path to the ROS workspace that contains the ROS packages. These are installed using
   the `rosdep <https://docs.ros.org/en/humble/Tutorials/Intermediate/Rosdep.html>`__ dependency manager.

As an example, the following ``extension.toml`` file specifies the dependencies for the extension:

.. code-block:: toml

   [isaac_lab_settings]
   # apt dependencies
   apt_deps = ["libboost-all-dev"]

   # ROS workspace
   # note: if this path is relative, it is relative to the extension directory's root
   ros_ws = "/home/user/catkin_ws"

These dependencies are installed using the ``install_deps.py`` script provided in the ``tools`` directory.
To install all dependencies for all extensions, run the following command:

.. code-block:: bash

   # execute from the root of the repository
   # the script expects the type of dependencies to install and the path to the extensions directory
   # available types are: 'apt', 'rosdep' and 'all'
   python tools/install_deps.py all ${ISAACLAB_PATH}/source

.. note::
   Currently, this script is automatically executed during the build process of the ``Dockerfile.base``
   and ``Dockerfile.ros2``. This ensures that all the 'apt' and 'rosdep' dependencies are installed
   before building the extensions respectively.


Standalone applications
~~~~~~~~~~~~~~~~~~~~~~~

In a typical Omniverse workflow, the simulator is launched first and then the extensions are
enabled. The loading of python modules and other python applications happens automagically, under the hood, and while this is the recommended
workflow, it is not always possible.

For example, consider robot reinforcement learning. It is essential to have complete control over the simulation step
and when things update instead of asynchronously waiting for the result. In
such cases, we require direct control of the simulation, and so it is necessary to write a standalone application. These applications are functionally similar in that they launch the simulator using the :class:`~isaaclab.app.AppLauncher` and
then control the simulation directly through the :class:`~isaaclab.sim.SimulationContext`. In these cases, python modules from extensions **must** be imported after the app is launched.  Doing so before the app is launched will cause missing module errors.

The following snippet shows how to write a standalone application:

.. code:: python

   """Launch Isaac Sim Simulator first."""

   from isaaclab.app import AppLauncher

   # launch omniverse app
   app_launcher = AppLauncher(headless=False)
   simulation_app = app_launcher.app


   """Rest everything follows."""

   from isaaclab.sim import SimulationContext

   if __name__ == "__main__":
      # get simulation context
      simulation_context = SimulationContext()
      # reset and play simulation
      simulation_context.reset()
      # step simulation
      simulation_context.step()
      # stop simulation
      simulation_context.stop()

      # close the simulation
      simulation_app.close()


It is necessary to launch the simulator before running any other code because extensions are hot-loaded
when the simulator starts. Many Omniverse modules become available only after the simulator is launched.
For further details, we recommend exploring the Isaac Lab :ref:`tutorials`.
