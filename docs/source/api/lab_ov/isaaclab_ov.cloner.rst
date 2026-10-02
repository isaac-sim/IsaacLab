isaaclab\_ov.cloner
====================

.. automodule:: isaaclab_ov.cloner

Clone contexts route assets to the representation that owns their copies:

* :class:`~isaaclab_ov.cloner.OvPhysxReplicateContext` prepares native physics replication.
* :class:`~isaaclab_ov.cloner.OvrtxReplicateContext` prepares copies for an OVRTX engine's internal scene.
* :class:`~isaaclab_ov.cloner.OvstageReplicateContext` prepares copies for simulation-owned OVStage resources,
  independently of their consumers. An OVRTX engine borrowing that stage does not clone it again.

OVPhysX and OVRTX configurations request their native clone contexts by default. Clone preparation
resolves the representation before native resource initialization. With
``ISAAC_LAB_OVRTX_USE_OVSTAGE=1``, it replaces the OVRTX route with
``OvstageReplicateContext`` and acquires an isolated rendering stage from the simulation registry.
OVPhysX retains its private stage and native cloning. The stage owner coordinates rendering write
ordinals and closes after its borrowers. Rendering on an OVStage is synchronous.
