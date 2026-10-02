isaaclab\_ov.cloner
====================

.. automodule:: isaaclab_ov.cloner

Clone contexts route assets to the representation that owns their copies:

* :class:`~isaaclab_ov.cloner.OvPhysxReplicateContext` prepares native physics replication.
* :class:`~isaaclab_ov.cloner.OvrtxReplicateContext` prepares copies for an OVRTX engine's internal scene.
* :class:`~isaaclab_ov.cloner.OvstageReplicateContext` prepares copies for simulation-owned OVStage resources,
  independently of their consumers. An OVRTX engine borrowing that stage does not clone it again.

.. _ov-clone-routing:

Clone routing
-------------

``OvrtxReplicateContext.prepare`` in ``isaaclab_ov/cloner/replicate.py`` selects the representation
before native resource initialization. Without environment overrides, OVPhysX uses native physics
cloning and an independent OVStage for OVRTX rendering; Newton uses native OVRTX cloning.

``ISAAC_LAB_OVRTX_USE_OVSTAGE=1`` selects independent OVStage rendering with either physics backend;
``0`` selects native OVRTX cloning. Set the flag before creating the simulation. Reset retains the
selected route. OVPhysX keeps its private physics stage and native cloning in both cases.

The rendering stage owner coordinates write ordinals and closes after its borrowers. Rendering on
an OVStage is synchronous.
