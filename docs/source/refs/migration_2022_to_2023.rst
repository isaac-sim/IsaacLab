Migration Guide (Isaac Sim)
===========================

Moving from Isaac Sim 2022.2.1 to 2023.1.0 and later brings in a number of changes to the
APIs and the way the application is built. This document outlines the changes
and how to migrate your code to the new APIs. Many of these changes attribute to
the underlying Omniverse Kit upgrade from 104.2 to 105.1. The new upgrade brings
the following notable changes:

* Update to USD 22.11
* Upgrading the Python from 3.7 to 3.10


Renaming of PhysX Flatcache to PhysX Fabric
-------------------------------------------

The PhysX Flatcache has been renamed to PhysX Fabric. The new name is more
descriptive of the functionality and is consistent with the naming convention
used by Omniverse called `Fabric`_. Consequently, the Python module name has
also been changed from :mod:`omni.physxflatcache` to :mod:`omni.physxfabric`.

Following this, on the Isaac Sim side, various renaming have occurred:

* The parameter passed to :class:`SimulationContext` constructor via the keyword :obj:`sim_params`
  now expects the key ``use_fabric`` instead of ``use_flatcache``.
* The Python attribute :attr:`SimulationContext.get_physics_context().use_flatcache` is now
  :attr:`SimulationContext.get_physics_context().use_fabric`.
* The Python function :meth:`SimulationContext.get_physics_context().enable_flatcache` is now
  :meth:`SimulationContext.get_physics_context().enable_fabric`.


Renaming of the URDF and MJCF Importers
---------------------------------------

Starting from Isaac Sim 2023.1, the URDF and MJCF importers have been renamed to be more consistent
with the other asset importers in Omniverse. The importers are now available on NVIDIA-Omniverse GitHub
as open source projects.

Due to the extension name change, the Python module names have also been changed:

* URDF Importer: :mod:`omni.importer.urdf` (previously :mod:`omni.isaac.urdf`)
* MJCF Importer: :mod:`omni.importer.mjcf` (previously :mod:`omni.isaac.mjcf`)


Deprecation of :class:`UsdLux.Light` API
----------------------------------------

As highlighted in the release notes of `USD 22.11`_, the ``UsdLux.Light`` API has
been deprecated in favor of the new ``UsdLuxLightAPI`` API. In the new API the attributes
are prefixed with ``inputs:``. For example, the ``intensity`` attribute is now available as
``inputs:intensity``.

The following example shows how to create a sphere light using the old API and
the new API.

.. dropdown:: Code for Isaac Sim 2022.2.1 and below
  :icon: code

  .. code-block:: python

      import isaacsim.core.utils.prims as prim_utils

      prim_utils.create_prim(
        "/World/Light/GreySphere",
        "SphereLight",
        translation=(4.5, 3.5, 10.0),
        attributes={"radius": 2.5, "intensity": 600.0, "color": (0.75, 0.75, 0.75)},
      )

.. dropdown:: Code for Isaac Sim 2023.1.0 and above
  :icon: code

  .. code-block:: python

      import isaacsim.core.utils.prims as prim_utils

      prim_utils.create_prim(
          "/World/Light/WhiteSphere",
          "SphereLight",
          translation=(-4.5, 3.5, 10.0),
          attributes={
            "inputs:radius": 2.5,
            "inputs:intensity": 600.0,
            "inputs:color": (1.0, 1.0, 1.0)
          },
      )


.. _Fabric: https://docs.omniverse.nvidia.com/kit/docs/usdrt/latest/docs/usd_fabric_usdrt.html
.. _`USD 22.11`: https://github.com/PixarAnimationStudios/OpenUSD/blob/release/CHANGELOG.md
