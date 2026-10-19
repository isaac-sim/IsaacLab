:orphan:

Mastering Omniverse for Robotics
================================

These resources cover USD scene authoring and PhysX setup in Omniverse tools.
They are useful when preparing assets for Isaac Lab or working in Isaac Sim. For
Isaac Lab's physics configuration interface, see :doc:`/source/concepts/schema_cfgs`.

The topics include:

- Navigating USD Composer and editing scenes.
- USD prims, schemas, references, payloads, and instancing.
- Applying PhysX schemas to prims.
- Creating prims and changing their attributes through Python.

Part 1: Using USD Composer
--------------------------

Start with the introductions, then choose the resources for the asset or scene
you are preparing.

Introduction to Omniverse and USD

-  `What is NVIDIA Omniverse? <https://youtu.be/dvdB-ndYJBM>`__
-  `What is the USD File Type? \| Getting Started in NVIDIA Omniverse <https://youtu.be/GOdyx-oSs2M>`__
-  `What Makes USD Unique in NVIDIA Omniverse <https://youtu.be/o2x-30-PTkw>`__

Using Omniverse USD Composer

-  `Introduction to Omniverse USD Composer <https://youtu.be/_30Pf3nccuE>`__
-  `Navigation Basics in Omniverse USD Composer <https://youtu.be/kb4ZA3TyMak>`__
-  `Lighting Basics in NVIDIA Omniverse USD Composer <https://youtu.be/c7qyI8pZvF4>`__
-  `Rendering Overview in NVIDIA Omniverse USD Composer <https://youtu.be/dCvq2ZyYmu4>`__

Materials and MDL

-  `Five Things to Know About Materials in NVIDIA Omniverse <https://youtu.be/C0HmcQXaENc>`__
-  `How to apply materials? <https://docs.omniverse.nvidia.com/materials-and-rendering/latest/materials.html#applying-materials>`__

Omniverse Physics and PhysX SDK

-  `Basics - Setting Up Physics and Toolbar Overview <https://youtu.be/nsJ0S9MycJI>`__
-  `Basics - Demos Overview <https://youtu.be/-y0-EVTj10s>`__
-  `Rigid Bodies - Mass Editing <https://youtu.be/GHl2RwWeRuM>`__
-  `Materials - Friction Restitution and Defaults <https://youtu.be/oTW81DltNiE>`__
-  `Overview of Simulation Ready Assets Physics in Omniverse <https://youtu.be/lFtEMg86lJc>`__

Importing assets

-  `Omniverse Create - Importing FBX Files \| NVIDIA Omniverse Tutorials <https://youtu.be/dQI0OpzfVHw>`__
-  `Omniverse Asset Importer <https://docs.omniverse.nvidia.com/extensions/latest/ext_asset-importer.html>`__
-  `Isaac Sim URDF impoter <https://docs.isaacsim.omniverse.nvidia.com/latest/importer_exporter/ext_isaacsim_asset_importer_urdf.html>`__


Part 2: Scripting in Omniverse
------------------------------

The above links mainly introduced how to use the USD Composer and its
functionalities through UI operations. However, often developers
need to write scripts to perform operations. This is especially true
when you want to automate certain tasks or create custom applications
that use Omniverse as a backend. This section will introduce you to
scripting in Omniverse.

USD is the main file format Omniverse operates with. So naturally, the
APIs (from OpenUSD) for modifying USD are at the core of Omniverse.
Most of the APIs are in C++ and Python bindings are provided for them.
Thus, to script in Omniverse, you need to understand the USD APIs.

.. note::

   While Isaac Sim and Isaac Lab try to "relieve" users from understanding
   the core USD concepts and APIs, understanding these basics still
   help a lot once you start diving inside the codebase and modifying
   it for your own application.

Before diving into USD scripting, it is good to get acquainted with the
terminologies used in USD. We recommend the following `introduction to
USD basics <https://www.sidefx.com/docs/houdini/solaris/usd.html>`__ by
Houdini, which is a 3D animation software.
Make sure to go through the following sections:

-  `Quick example <https://www.sidefx.com/docs/houdini/solaris/usd.html#quick-example>`__
-  `Attributes and primvars <https://www.sidefx.com/docs/houdini/solaris/usd.html#attrs>`__
-  `Composition <https://www.sidefx.com/docs/houdini/solaris/usd.html#compose>`__
-  `Schemas <https://www.sidefx.com/docs/houdini/solaris/usd.html#schemas>`__
-  `Instances <https://www.sidefx.com/docs/houdini/solaris/usd.html#instancing>`__
   and `Scene-graph Instancing <https://openusd.org/dev/api/_usd__page__scenegraph_instancing.html>`__

As a test of understanding, make sure you can answer the following:

-  What are prims? What is meant by a prim path in a stage?
-  How are attributes related to prims?
-  How are schemas related to prims?
-  What is the difference between attributes and schemas?
-  What is asset instancing?

Part 3: More Resources
----------------------

- `Omniverse Glossary of Terms <https://docs.isaacsim.omniverse.nvidia.com/latest/reference_material/reference_glossary.html>`__
- `Omniverse Code Samples <https://docs.omniverse.nvidia.com/dev-guide/latest/programmer_ref.html>`__
- `PhysX Limitations <https://docs.isaacsim.omniverse.nvidia.com/latest/physics/physics_resources.html>`__
- `PhysX Documentation <https://nvidia-omniverse.github.io/PhysX/physx/>`__.
