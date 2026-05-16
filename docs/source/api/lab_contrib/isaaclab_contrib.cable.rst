isaaclab_contrib.cable
======================

.. automodule:: isaaclab_contrib.cable

  .. rubric:: Classes

  .. autosummary::

    cable_object.CableObject
    cable_object.CableRegistryEntry
    cable_object_cfg.CableObjectCfg

  .. rubric:: Replicate-hook plumbing

  .. autosummary::

    cable_object.add_cable_entry_to_builder
    cable_object.add_registered_cables_to_builder
    cable_object.install_cable_builder_hooks

Cable Object
------------

.. autoclass:: isaaclab_contrib.cable.cable_object.CableObject
  :members:
  :inherited-members:
  :show-inheritance:

.. autoclass:: isaaclab_contrib.cable.cable_object_cfg.CableObjectCfg
  :members:
  :show-inheritance:
  :exclude-members: __init__

Replicate-Hook Plumbing
-----------------------

The cable registry / per-world builder hook mirrors the deformable contrib
pattern: :class:`CableObject` constructor appends a
:class:`CableRegistryEntry` to ``SimulationManager._cable_registry``, and the
hook installed by :func:`install_cable_builder_hooks` walks that registry once
per world during ``add_to_builder`` to call
:meth:`newton.ModelBuilder.add_rod_graph`.

.. autoclass:: isaaclab_contrib.cable.cable_object.CableRegistryEntry
  :members:
  :show-inheritance:
  :exclude-members: __init__

.. autofunction:: isaaclab_contrib.cable.cable_object.add_cable_entry_to_builder

.. autofunction:: isaaclab_contrib.cable.cable_object.add_registered_cables_to_builder

.. autofunction:: isaaclab_contrib.cable.cable_object.install_cable_builder_hooks
