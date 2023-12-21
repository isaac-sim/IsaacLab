Importing a New Asset
=====================

.. currentmodule:: omni.isaac.orbit

NVIDIA Omniverse relies on the Universal Scene Description (USD) file format to
import and export assets. USD is an open source file format developed by Pixar
Animation Studios. It is a scene description format optimized for large-scale,
complex data sets. While this format is widely used in the film and animation
industry, it is less common in the robotics community.

To this end, NVIDIA has developed various importers that allow you to import
assets from other file formats into USD. These importers are available as
extensions to Omniverse Kit:

* **URDF Importer** - Import assets from URDF files.
* **MJCF Importer** - Import assets from MJCF files.
* **Asset Importer** - Import assets from various file formats, including
  OBJ, FBX, STL, and glTF.

The recommended workflow from NVIDIA is to use the above importers to convert
the asset into its USD representation. Once the asset is in USD format, you can
use the Omniverse Kit to edit the asset and export it to other file formats.

An important note to use assets for large-scale simulation is to ensure that they
are in `instanceable`_ format. This allows the asset to be efficiently loaded
into memory and used multiple times in a scene. Otherwise, the asset will be
loaded into memory multiple times, which can cause performance issues.
For more details on instanceable assets, please check the Isaac Sim `documentation`_.


Using URDF Importer
-------------------

Isaac Sim includes the URDF and MJCF importers by default. These importers support the
option to import assets as instanceable assets. By selecting this option, the
importer will create two USD files: one for all the mesh data and one for
all the non-mesh data (e.g. joints, rigid bodies, etc.). The prims in the mesh data file are
referenced in the non-mesh data file. This allows the mesh data (which is often bulky) to be
loaded into memory only once and used multiple times in a scene.

For using these importers from the GUI, please check the documentation for `MJCF importer`_ and
`URDF importer`_ respectively.

For using the URDF importers from Python scripts, we include a utility tool called ``convert_urdf.py``.
Internally, this script creates an instance of :class:`~sim.converters.UrdfConverterCfg` which
is then passed to the :class:`~sim.converters.UrdfConverter` class. The configuration class specifies
the default values for the importer. The important settings are:

* :attr:`~sim.converters.UrdfConverterCfg.fix_base` - Whether to fix the base of the robot.
  This depends on whether you have a floating-base or fixed-base robot.
* :attr:`~sim.converters.UrdfConverterCfg.make_instanceable` - Whether to create instanceable assets.
  Usually, this should be set to ``True``.
* :attr:`~sim.converters.UrdfConverterCfg.merge_fixed_joints` - Whether to merge the fixed joints.
  Usually, this should be set to ``True`` to reduce the asset complexity.
* :attr:`~sim.converters.UrdfConverterCfg.default_drive_type` - The drive-type for the joints.
  We recommend this to always be ``"none"``. This allows changing the drive configuration using the
  actuator models.
* :attr:`~sim.converters.UrdfConverterCfg.default_drive_stiffness` - The drive stiffness for the joints.
  We recommend this to always be ``0.0``. This allows changing the drive configuration using the
  actuator models.
* :attr:`~sim.converters.UrdfConverterCfg.default_drive_damping` - The drive damping for the joints.
  Similar to the stiffness, we recommend this to always be ``0.0``.

Example Usage
~~~~~~~~~~~~~

In this example, we use the pre-processed URDF file of the ANYmal-D robot. To check the
pre-process URDF, please check the file the `anymal.urdf`_. The main difference between the
pre-processed URDF and the original URDF are:

* We removed the ``<gazebo>`` tag from the URDF. This tag is not supported by the URDF importer.
* We removed the ``<transmission>`` tag from the URDF. This tag is not supported by the URDF importer.
* We removed various collision bodies from the URDF to reduce the complexity of the asset.
* We changed all the joint's damping and friction parameters to ``0.0``. This ensures that we can perform
  effort-control on the joints without PhysX adding additional damping.
* We added the ``<dont_collapse>`` tag to fixed joints. This ensures that the importer does
  not merge these fixed joints.

The following shows the steps to clone the repository and run the converter:

.. code-block:: bash

  # create a directory to clone
  mkdir ~/git && cd ~/git
  # clone a repository with URDF files
  git clone git@github.com:isaac-orbit/anymal_d_simple_description.git

  # go to top of the repository
  cd /path/to/orbit
  # run the converter
  ./orbit.sh -p source/standalone/tools/convert_urdf.py \
    ~/git/anymal_d_simple_description/urdf/anymal.urdf \
    source/extensions/omni.isaac.orbit_assets/data/Robots/ANYbotics/anymal_d.usd \
    --merge-joints \
    --make-instanceable


Executing the above script will create two USD files inside the
``source/extensions/omni.isaac.orbit_assets/data/Robots/ANYbotics/`` directory:

* ``anymal_d.usd`` - This is the main asset file. It contains all the non-mesh data.
* ``Props/instanceable_assets.usd`` - This is the mesh data file.

You can press play on the opened window to see the asset in the scene. The asset should "collapse"
if everything is working correctly. If it blows up, then it might be that you have self-collisions
present in the URDF.

To run the script headless, you can add the ``--headless`` flag. This will not open the GUI and
exit the script after the conversion is complete.


Using Mesh Importer
-------------------

Omniverse Kit includes the mesh converter tool that uses the ASSIMP library to import assets
from various mesh formats (e.g. OBJ, FBX, STL, glTF, etc.). The asset converter tool is available
as an extension to Omniverse Kit. Please check the `asset converter`_ documentation for more details.
However, unlike Isaac Sim's URDF and MJCF importers, the asset converter tool does not support
creating instanceable assets. This means that the asset will be loaded into memory multiple times
if it is used multiple times in a scene.

Thus, we include a utility tool called ``convert_mesh.py`` that uses the asset converter tool to
import the asset and then converts it into an instanceable asset. Internally, this script creates
an instance of :class:`~sim.converters.MeshConverterCfg` which is then passed to the
:class:`~sim.converters.MeshConverter` class. Since the mesh file does not contain any physics
information, the configuration class accepts different physics properties (such as mass, collision
shape, etc.) as input. Please check the documentation for :class:`~sim.converters.MeshConverterCfg`
for more details.

Example Usage
~~~~~~~~~~~~~

We use an OBJ file of a cube to demonstrate the usage of the mesh converter. The following shows
the steps to clone the repository and run the converter:

.. code-block:: bash

  # create a directory to clone
  mkdir ~/git && cd ~/git
  # clone a repository with URDF files
  git clone git@github.com:NVIDIA-Omniverse/IsaacGymEnvs.git

  # go to top of the repository
  cd /path/to/orbit
  # run the converter
  ./orbit.sh -p source/standalone/tools/convert_mesh.py \
    ~/git/IsaacGymEnvs/assets/trifinger/objects/meshes/cube_multicolor.obj \
    source/extensions/omni.isaac.orbit_assets/data/Props/CubeMultiColor/cube_multicolor.usd \
    --make-instanceable \
    --collision-approximation convexDecomposition \
    --mass 1.0

Similar to the URDF converter, executing the above script will create two USD files inside the
``source/extensions/omni.isaac.orbit_assets/data/Props/CubeMultiColor/`` directory. Additionally,
if you press play on the opened window, you should see the asset fall down under the influence
of gravity.

* If you do not set the ``--mass`` flag, then no rigid body properties will be added to the asset.
  It will be imported as a static asset.
* If you also do not set the ``--collision-approximation`` flag, then the asset will not have any collider
  properties as well and will be imported as a visual asset.


.. _instanceable: https://openusd.org/dev/api/_usd__page__scenegraph_instancing.html
.. _documentation: https://docs.omniverse.nvidia.com/isaacsim/latest/isaac_gym_tutorials/tutorial_gym_instanceable_assets.html
.. _MJCF importer: https://docs.omniverse.nvidia.com/isaacsim/latest/advanced_tutorials/tutorial_advanced_import_mjcf.html
.. _URDF importer: https://docs.omniverse.nvidia.com/isaacsim/latest/advanced_tutorials/tutorial_advanced_import_urdf.html
.. _anymal.urdf: https://github.com/isaac-orbit/anymal_d_simple_description/blob/master/urdf/anymal.urdf
.. _asset converter: https://docs.omniverse.nvidia.com/extensions/latest/ext_asset-converter.html
