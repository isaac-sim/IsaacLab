Importing a New Asset
=====================

.. currentmodule:: isaaclab

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
* **Mesh Importer** - Import assets from various file formats, including
  OBJ, FBX, STL, and glTF.

The recommended workflow from NVIDIA is to use the above importers to convert
the asset into its USD representation. Once the asset is in USD format, you can
use the Omniverse Kit to edit the asset and export it to other file formats. Isaac Sim includes
these importers by default. They can also be enabled manually in Omniverse Kit.


An important note to use assets for large-scale simulation is to ensure that they
are in `instanceable`_ format. This allows the asset to be efficiently loaded
into memory and used multiple times in a scene. Otherwise, the asset will be
loaded into memory multiple times, which can cause performance issues.
For more details on instanceable assets, please check the Isaac Sim `documentation`_.


Using URDF Importer
-------------------

For using the URDF importer in the GUI, please check the documentation at `URDF importer`_. For using the URDF importer from Python scripts, we include a utility tool called ``convert_urdf.py``. This script creates an instance of :class:`~sim.converters.UrdfConverterCfg` which
is then passed to the :class:`~sim.converters.UrdfConverter` class.

The URDF importer has various configuration parameters that can be set to control the behavior of the importer.
The default values for the importer's configuration parameters are specified are in the :class:`~sim.converters.UrdfConverterCfg` class, and they are listed below. We made a few commonly modified settings to be available as command-line arguments when calling the ``convert_urdf.py``, and they are marked with ``*`` in the list. For a comprehensive list of the configuration parameters, please check the the documentation at `URDF importer`_.

* :attr:`~sim.converters.UrdfConverterCfg.fix_base` * - Whether to fix the base of the robot.
  This depends on whether you have a floating-base or fixed-base robot. The command-line flag is
  ``--fix-base`` where when set, the importer will fix the base of the robot, otherwise it will default to floating-base.
* :attr:`~sim.converters.UrdfConverterCfg.root_link_name` - The link on which the PhysX articulation root is placed.
* :attr:`~sim.converters.UrdfConverterCfg.merge_fixed_joints` * - Whether to merge the fixed joints.
  Usually, this should be set to ``True`` to reduce the asset complexity. The command-line flag is
  ``--merge-joints`` where when set, the importer will merge the fixed joints, otherwise it will default to not merging the fixed joints.
* :attr:`~sim.converters.UrdfConverterCfg.joint_drive` - The configuration for the joint drives on the robot.

  * :attr:`~sim.converters.UrdfConverterCfg.JointDriveCfg.drive_type` - The drive type for the joints.
    This can be either ``"acceleration"`` or ``"force"``. We recommend using ``"force"`` for most cases.
  * :attr:`~sim.converters.UrdfConverterCfg.JointDriveCfg.target_type` - The target type for the joints.
    This can be either ``"none"``, ``"position"``, or ``"velocity"``. We recommend using ``"position"`` for most cases.
    Setting this to ``"none"`` will disable the drive and set the joint gains to 0.0.
  * :attr:`~sim.converters.UrdfConverterCfg.JointDriveCfg.gains` - The drive stiffness and damping gains for the joint.
    We support two ways to set the gains:

    * :attr:`~sim.converters.UrdfConverterCfg.JointDriveCfg.PDGainsCfg` - To directly set the stiffness and damping.
    * :attr:`~sim.converters.UrdfConverterCfg.JointDriveCfg.NaturalFrequencyGainsCfg` - To set the gains using the
      desired natural frequency response of the system.

For more detailed information on the configuration parameters, please check the documentation for :class:`~sim.converters.UrdfConverterCfg`.

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


.. tab-set::
   :sync-group: os

   .. tab-item:: :icon:`fa-brands fa-linux` Linux
      :sync: linux

      .. code-block:: bash

        # clone a repository with URDF files
        git clone git@github.com:isaac-orbit/anymal_d_simple_description.git

        # go to top of the Isaac Lab repository
        cd IsaacLab
        # run the converter
        ./isaaclab.sh -p scripts/tools/convert_urdf.py \
          ../anymal_d_simple_description/urdf/anymal.urdf \
          source/isaaclab_assets/data/Robots/ANYbotics/anymal_d.usd \
          --merge-joints \
          --joint-stiffness 0.0 \
          --joint-damping 0.0 \
          --joint-target-type none

   .. tab-item:: :icon:`fa-brands fa-windows` Windows
      :sync: windows

      .. code-block:: batch

        :: clone a repository with URDF files
        git clone git@github.com:isaac-orbit/anymal_d_simple_description.git

        :: go to top of the Isaac Lab repository
        cd IsaacLab
        :: run the converter
        isaaclab.bat -p scripts\tools\convert_urdf.py ^
          ..\anymal_d_simple_description\urdf\anymal.urdf ^
          source\isaaclab_assets\data\Robots\ANYbotics\anymal_d.usd ^
          --merge-joints ^
          --joint-stiffness 0.0 ^
          --joint-damping 0.0 ^
          --joint-target-type none

Executing the above script will create a USD file inside the
``source/isaaclab_assets/data/Robots/ANYbotics/`` directory:

* ``anymal_d.usd`` - This is the main asset file.


To run the script headless, you can add the ``--headless`` flag. This will not open the GUI and
exit the script after the conversion is complete.

You can press play on the opened window to see the asset in the scene. The asset should fall under gravity. If it blows up, then it might be that you have self-collisions present in the URDF.


.. figure:: ../_static/tutorials/tutorial_convert_urdf.jpg
    :align: center
    :figwidth: 100%
    :alt: result of convert_urdf.py



Using MJCF Importer
-------------------

Similar to the URDF Importer, the MJCF Importer also has a GUI interface. Please check the documentation at
`MJCF importer`_ for more details. For using the MJCF importer from Python scripts, we include a utility tool
called ``convert_mjcf.py``. This script creates an instance of :class:`~sim.converters.MjcfConverterCfg`
which is then passed to the :class:`~sim.converters.MjcfConverter` class.

The default values for the importer's configuration parameters are specified in the
:class:`~sim.converters.MjcfConverterCfg` class. The configuration parameters are listed below.
We made a few commonly modified settings to be available as command-line arguments when calling the
``convert_mjcf.py``, and they are marked with ``*`` in the list. For a comprehensive list of the configuration
parameters, please check the the documentation at `MJCF importer`_.

.. note::
   The MJCF importer was rewritten in Isaac Sim 5.0 to use the ``mujoco-usd-converter`` library.
   Settings such as ``fix_base``, ``import_sites``, ``import_inertia_tensor``, and ``make_instanceable``
   are no longer needed — the converter now handles these automatically based on the MJCF file content.

* :attr:`~sim.converters.MjcfConverterCfg.merge_mesh` * - Whether to merge meshes where possible to
  optimize the model. The command-line flag is ``--merge-mesh``.
* :attr:`~sim.converters.MjcfConverterCfg.collision_from_visuals` * - Whether to generate collision
  geometry from visual geometries. The command-line flag is ``--collision-from-visuals``.
* :attr:`~sim.converters.MjcfConverterCfg.collision_type` * - Type of collision geometry to use
  (e.g. ``"default"``, ``"Convex Hull"``, ``"Convex Decomposition"``). The command-line flag is
  ``--collision-type``.
* :attr:`~sim.converters.MjcfConverterCfg.self_collision` * - Whether to activate self-collisions
  between links of the articulation. The command-line flag is ``--self-collision``.


Example Usage
~~~~~~~~~~~~~

In this example, we use the MuJoCo model of the Unitree's H1 humanoid robot in the `mujoco_menagerie`_.

The following shows the steps to clone the repository and run the converter:


.. tab-set::
   :sync-group: os

   .. tab-item:: :icon:`fa-brands fa-linux` Linux
      :sync: linux

      .. code-block:: bash

        # clone a repository with MJCF files
        git clone git@github.com:google-deepmind/mujoco_menagerie.git

        # go to top of the Isaac Lab repository
        cd IsaacLab
        # run the converter
        ./isaaclab.sh -p scripts/tools/convert_mjcf.py \
          ../mujoco_menagerie/unitree_h1/h1.xml \
          source/isaaclab_assets/data/Robots/Unitree/h1.usd \
          --merge-mesh

   .. tab-item:: :icon:`fa-brands fa-windows` Windows
      :sync: windows

      .. code-block:: batch

        :: clone a repository with MJCF files
        git clone git@github.com:google-deepmind/mujoco_menagerie.git

        :: go to top of the Isaac Lab repository
        cd IsaacLab
        :: run the converter
        isaaclab.bat -p scripts\tools\convert_mjcf.py ^
          ..\mujoco_menagerie\unitree_h1\h1.xml ^
          source\isaaclab_assets\data\Robots\Unitree\h1.usd ^
          --merge-mesh

Executing the above script will create the USD file inside the
``source/isaaclab_assets/data/Robots/Unitree/`` directory:

* ``h1.usd`` - This is the converted USD asset file.

.. figure:: ../_static/tutorials/tutorial_convert_mjcf.jpg
    :align: center
    :figwidth: 100%
    :alt: result of convert_mjcf.py


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

.. tab-set::
   :sync-group: os

   .. tab-item:: :icon:`fa-brands fa-linux` Linux
      :sync: linux

      .. code-block:: bash

        # clone a repository with URDF files
        git clone git@github.com:NVIDIA-Omniverse/IsaacGymEnvs.git

        # go to top of the Isaac Lab repository
        cd IsaacLab
        # run the converter
        ./isaaclab.sh -p scripts/tools/convert_mesh.py \
          ../IsaacGymEnvs/assets/trifinger/objects/meshes/cube_multicolor.obj \
          source/isaaclab_assets/data/Props/CubeMultiColor/cube_multicolor.usd \
          --make-instanceable \
          --collision-approximation convexDecomposition \
          --mass 1.0

   .. tab-item:: :icon:`fa-brands fa-windows` Windows
      :sync: windows

      .. code-block:: batch

        :: clone a repository with URDF files
        git clone git@github.com:NVIDIA-Omniverse/IsaacGymEnvs.git

        :: go to top of the Isaac Lab repository
        cd IsaacLab
        :: run the converter
        isaaclab.bat -p scripts\tools\convert_mesh.py ^
          ..\IsaacGymEnvs\assets\trifinger\objects\meshes\cube_multicolor.obj ^
          source\isaaclab_assets\data\Props\CubeMultiColor\cube_multicolor.usd ^
          --make-instanceable ^
          --collision-approximation convexDecomposition ^
          --mass 1.0

You may need to press 'F' to zoom in on the asset after import.

Similar to the URDF and MJCF converter, executing the above script will create two USD files inside the
``source/isaaclab_assets/data/Props/CubeMultiColor/`` directory. Additionally,
if you press play on the opened window, you should see the asset fall down under the influence
of gravity.

* If you do not set the ``--mass`` flag, then no rigid body properties will be added to the asset.
  It will be imported as a static asset.
* If you also do not set the ``--collision-approximation`` flag, then the asset will not have any collider
  properties as well and will be imported as a visual asset.


.. figure:: ../_static/tutorials/tutorial_convert_mesh.jpg
    :align: center
    :figwidth: 100%
    :alt: result of convert_mesh.py


.. _instanceable: https://openusd.org/dev/api/_usd__page__scenegraph_instancing.html
.. _documentation: https://docs.isaacsim.omniverse.nvidia.com/latest/isaac_lab_tutorials/tutorial_instanceable_assets.html
.. _MJCF importer: https://docs.isaacsim.omniverse.nvidia.com/latest/importer_exporter/ext_isaacsim_asset_importer_mjcf.html
.. _URDF importer: https://docs.isaacsim.omniverse.nvidia.com/latest/importer_exporter/ext_isaacsim_asset_importer_urdf.html
.. _anymal.urdf: https://github.com/isaac-orbit/anymal_d_simple_description/blob/master/urdf/anymal.urdf
.. _asset converter: https://docs.omniverse.nvidia.com/extensions/latest/ext_asset-converter.html
.. _mujoco_menagerie: https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_h1
