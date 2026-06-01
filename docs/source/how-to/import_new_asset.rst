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

.. note::
   The URDF importer was upgraded to version 3.0 in Isaac Sim 6, replacing the previous C++
   binding-based API with a Python pipeline (``urdf-usd-converter``). Assets are now made
   instanceable by default — ``make_instanceable`` is no longer a configuration option.
   See the :doc:`/source/migration/migrating_to_isaaclab_3-0` for a full list of breaking changes.

The URDF importer has various configuration parameters that can be set to control the behavior of the importer.
The default values for the importer's configuration parameters are specified are in the :class:`~sim.converters.UrdfConverterCfg` class, and they are listed below. We made a few commonly modified settings to be available as command-line arguments when calling the ``convert_urdf.py``, and they are marked with ``*`` in the list. For a comprehensive list of the configuration parameters, please check the the documentation at `URDF importer`_.

Articulation and joint structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* :attr:`~sim.converters.UrdfConverterCfg.fix_base` * - Whether to fix the base of the robot.
  This depends on whether you have a floating-base or fixed-base robot. The command-line flag is
  ``--fix-base`` where when set, the importer will fix the base of the robot, otherwise it will default to floating-base.
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
      Both ``stiffness`` and ``damping`` accept a single float (applied uniformly).
    * :attr:`~sim.converters.UrdfConverterCfg.JointDriveCfg.NaturalFrequencyGainsCfg` - To set the gains using the
      desired natural frequency response of the system. **Deprecated in URDF importer 3.0** — use
      ``PDGainsCfg`` instead.

Geometry, collisions, and materials
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* :attr:`~sim.converters.UrdfConverterCfg.collision_from_visuals` - Whether to create collision geometry
  from visual geometry when no explicit ``<collision>`` is defined for a link. Defaults to ``False``.
* :attr:`~sim.converters.UrdfConverterCfg.collision_type` - The collision shape simplification to apply.
  One of ``"Convex Hull"`` (default), ``"Convex Decomposition"``, ``"Bounding Sphere"``, or ``"Bounding Cube"``.
* :attr:`~sim.converters.UrdfConverterCfg.self_collision` - Whether to activate self-collisions between
  links of the articulation. Defaults to ``False``.
* :attr:`~sim.converters.UrdfConverterCfg.merge_mesh` - Whether to merge meshes where possible to optimize
  the model. Defaults to ``False``.
* :attr:`~sim.converters.UrdfConverterCfg.link_density` - Default density in ``kg/m^3`` for links whose
  ``<inertial>`` properties are missing. ``0.0`` (default) leaves densities unchanged.

Asset resolution and output
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* :attr:`~sim.converters.UrdfConverterCfg.ros_package_paths` - List of ROS package name/path mappings used
  to resolve ``package://`` URLs in the URDF. Each entry is a dict with keys ``name`` and ``path``.
* :attr:`~sim.converters.UrdfConverterCfg.robot_type` - Robot type applied by the USD robot schema.
  Defaults to ``"Default"``. Must be one of: ``"Default"``, ``"End Effector"``, ``"Manipulator"``,
  ``"Humanoid"``, ``"Wheeled"``, ``"Holonomic"``, ``"Quadruped"``, ``"Mobile Manipulators"``, ``"Aerial"``.
* :attr:`~sim.converters.UrdfConverterCfg.run_asset_transformer` - Run the asset transformer to convert
  the flattened USD into a layered USD (interface USD + payloads). Defaults to ``True``.
* :attr:`~sim.converters.UrdfConverterCfg.run_multi_physics_conversion` - Also emit MuJoCo-compatible joint
  attributes alongside PhysX. Defaults to ``True``.
* :attr:`~sim.converters.UrdfConverterCfg.debug_mode` - Write intermediate conversion artifacts next to the
  output USD for inspection. Defaults to ``False``.

Deprecated (no-op in URDF importer 3.0)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following options are retained for backwards compatibility but are ignored by the URDF importer 3.0.
A warning is logged when they are set.

* :attr:`~sim.converters.UrdfConverterCfg.root_link_name` - The link on which the PhysX articulation root
  was previously placed.
* :attr:`~sim.converters.UrdfConverterCfg.convert_mimic_joints_to_normal_joints` - Convert mimic joints to
  normal joints during conversion.
* :attr:`~sim.converters.UrdfConverterCfg.replace_cylinders_with_capsules` - Replace cylinder shapes with
  capsule shapes during conversion.

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
* The ``<dont_collapse>`` URDF tag is **no longer supported** in URDF importer 3.0. Fixed joint
  merging is now a Python pre-processing step that merges all fixed joints when
  ``merge_fixed_joints`` is enabled. If you need to preserve a specific fixed joint, disable
  ``merge_fixed_joints`` entirely or restructure the URDF to use a non-fixed joint type
  (e.g. revolute with zero-range limits).

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
          source/isaaclab_assets/data/Robots/ANYbotics/ \
          --merge-joints \
          --joint-stiffness 0.0 \
          --joint-damping 0.0 \
          --joint-target-type none \
          --viz kit

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
          source\isaaclab_assets\data\Robots\ANYbotics\ ^
          --merge-joints ^
          --joint-stiffness 0.0 ^
          --joint-damping 0.0 ^
          --joint-target-type none ^
          --viz kit

Executing the above script will create a USD file inside the
``source/isaaclab_assets/data/Robots/ANYbotics/anymal/`` directory (the subdirectory name
is derived automatically from the robot name in the URDF):

* ``anymal.usda`` - This is the main asset file.

.. note::
   The URDF importer auto-deduplicates the per-robot subdirectory when it already exists.
   If you re-run the converter against the same ``usd_dir`` with a changed configuration
   (for example, flipping ``fix_base``), the importer writes to a new numbered folder
   (``anymal_1/``, ``anymal_2/``, …) rather than overwriting the previous output.
   :attr:`~sim.converters.UrdfConverter.usd_path` reflects whichever folder the importer
   actually used. Delete stale subdirectories manually (or wipe ``usd_dir``) if you do not
   want them to accumulate on disk.

The examples above pass ``--viz kit`` to open the GUI and inspect the converted asset.
To run the script headless and exit after the conversion is complete, omit ``--viz kit``.

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
   Settings such as ``import_sites``, ``import_inertia_tensor``, and ``make_instanceable`` are no
   longer needed — the converter now handles these automatically based on the MJCF file content.

Geometry, collisions, and materials
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* :attr:`~sim.converters.MjcfConverterCfg.merge_mesh` * - Whether to merge meshes where possible to
  optimize the model. The command-line flag is ``--merge-mesh``.
* :attr:`~sim.converters.MjcfConverterCfg.collision_from_visuals` * - Whether to generate collision
  geometry from visual geometries. The command-line flag is ``--collision-from-visuals``.
* :attr:`~sim.converters.MjcfConverterCfg.collision_type` * - The collision shape simplification to
  apply. One of ``"Convex Hull"`` (default), ``"Convex Decomposition"``, ``"Bounding Sphere"``, or
  ``"Bounding Cube"``. The command-line flag is ``--collision-type``.
* :attr:`~sim.converters.MjcfConverterCfg.self_collision` * - Whether to activate self-collisions
  between links of the articulation. The command-line flag is ``--self-collision``.

Articulation and physics
~~~~~~~~~~~~~~~~~~~~~~~~

* :attr:`~sim.converters.MjcfConverterCfg.fix_base` - Whether to add a fixed joint between the world
  and the root rigid-body link. Defaults to ``False``.
* :attr:`~sim.converters.MjcfConverterCfg.link_density` - Default density in ``kg/m^3`` for links whose
  ``<inertial>`` properties are missing in the MJCF. ``0.0`` (default) leaves densities unchanged.
* :attr:`~sim.converters.MjcfConverterCfg.import_physics_scene` * - Import physics scene properties
  (gravity, time step, etc.) from the MJCF file. Defaults to ``False``. The command-line flag is
  ``--import-physics-scene``.

Actuator overrides
~~~~~~~~~~~~~~~~~~

MuJoCo models actuators as an affine transformation ``tau = gain @ control + bias``. The following
options override the values parsed from the MJCF on a per-actuator basis. Each defaults to ``None``,
which leaves the parsed values unchanged.

* :attr:`~sim.converters.MjcfConverterCfg.override_gain_type` - The actuator gain type override (e.g.
  ``"fixed"``).
* :attr:`~sim.converters.MjcfConverterCfg.override_bias_type` - The actuator bias type override (e.g.
  ``"affine"``).
* :attr:`~sim.converters.MjcfConverterCfg.override_gain_prm` - The actuator gain parameter array override.
  Example for position control: ``[kp, 0, 0, 0, 0, 0, 0, 0, 0, 0]``.
* :attr:`~sim.converters.MjcfConverterCfg.override_bias_prm` - The actuator bias parameter array override.
  Example for position control: ``[0, -kp, -kd, 0, 0, 0, 0, 0, 0, 0]``.

Asset resolution and output
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* :attr:`~sim.converters.MjcfConverterCfg.robot_type` - Robot type applied by the USD robot schema.
  Defaults to ``"Default"``. Must be one of: ``"Default"``, ``"End Effector"``, ``"Manipulator"``,
  ``"Humanoid"``, ``"Wheeled"``, ``"Holonomic"``, ``"Quadruped"``, ``"Mobile Manipulators"``, ``"Aerial"``.
* :attr:`~sim.converters.MjcfConverterCfg.run_asset_transformer` - Run the asset transformer to convert
  the flattened USD into a layered USD (interface USD + payloads). Defaults to ``True``.
* :attr:`~sim.converters.MjcfConverterCfg.run_multi_physics_conversion` - Convert compatible MuJoCo
  attributes to PhysX attributes (e.g. actuator gains). Defaults to ``True``.
* :attr:`~sim.converters.MjcfConverterCfg.debug_mode` - Write intermediate conversion artifacts next to
  the output USD for inspection. Defaults to ``False``.

For more detailed information on the configuration parameters, please check the documentation for :class:`~sim.converters.MjcfConverterCfg`.


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
          --merge-mesh \
          --viz kit

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
          --merge-mesh ^
          --viz kit

Executing the above script will create the USD file inside the
``source/isaaclab_assets/data/Robots/Unitree/`` directory:

* ``h1.usd`` - This is the converted USD asset file.

.. note::
   The MJCF importer auto-deduplicates the per-robot subdirectory when it already exists,
   matching the URDF importer's behavior. If you re-run the converter against the same
   ``usd_dir`` with a changed configuration, the importer writes to a new numbered folder
   (``h1_1/``, ``h1_2/``, …) rather than overwriting the previous output.
   :attr:`~sim.converters.MjcfConverter.usd_path` reflects whichever folder the importer
   actually used. Delete stale subdirectories manually (or wipe ``usd_dir``) if you do not
   want them to accumulate on disk.

.. figure:: ../_static/tutorials/tutorial_convert_mjcf.jpg
    :align: center
    :figwidth: 100%
    :alt: result of convert_mjcf.py


.. _import-new-asset-ensure-drives-exist:

Ensuring joint drives exist on every joint
------------------------------------------

A common pitfall when porting a freshly-imported asset across physics backends
is that joints which actuate fine in PhysX silently do nothing in a
Newton-based backend (MuJoCo Warp, XPBD, Featherstone, Semi-implicit).

The URDF and MJCF importers both write a ``PhysicsDriveAPI`` to every
articulated joint, but the stiffness and damping on that drive are often left
at ``0`` — the assumption being that the actuator gains are authored at runtime
by an :class:`~isaaclab.actuators.ImplicitActuatorCfg` or
:class:`~isaaclab.actuators.IdealPDActuatorCfg`. This is the recommended way to
keep gains tunable per task without re-importing the USD.

PhysX creates a solver actuator for every joint regardless of the authored
gains, so the runtime writes from ``ImplicitActuatorCfg.stiffness`` /
``damping`` always take effect. Newton's USD importer, by contrast, only
materialises a solver actuator when the authored drive reports a non-zero
stiffness or damping — a joint whose authored gains are both zero is treated
as passive and is dropped from the actuator set, so subsequent runtime writes
have nothing to attach to.

The recommended fix is to opt the spawn config into the cross-backend bridge by
setting :attr:`~isaaclab.sim.schemas.JointDrivePropertiesCfg.ensure_drives_exist`
to ``True``:

.. code:: python

   spawn=sim_utils.UsdFileCfg(
       usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/Agility/Cassie/cassie.usd",
       joint_drive_props=sim_utils.JointDrivePropertiesCfg(ensure_drives_exist=True),
       ...
   )

When ``ensure_drives_exist=True``, every drive whose authored stiffness *and*
damping are both zero is updated with a minimal placeholder stiffness
(``1e-3``) before the simulation starts. This is enough for Newton's importer
to create the actuator; the actual gains are then overwritten by the actuator
model at runtime, so the placeholder has no effect on the simulated dynamics.
Drives whose authored gains are non-zero are left untouched.

This is how ``isaaclab_assets.CASSIE_CFG`` keeps Cassie working across both
PhysX and Newton — the asset ships with zero-gain drives because it relies on
``ImplicitActuatorCfg`` for the legs, and the spawn config enables
:attr:`~isaaclab.sim.schemas.JointDrivePropertiesCfg.ensure_drives_exist` so
that both backends see the same actuator set.

You can leave the flag at its default ``False`` for assets that author non-zero
drive gains in the USD itself, or for assets driven by an
:class:`~isaaclab.actuators.IdealPDActuatorCfg` that explicitly zeroes the
solver drive and applies torque externally.


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
