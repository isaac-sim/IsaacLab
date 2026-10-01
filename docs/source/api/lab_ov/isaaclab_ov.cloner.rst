isaaclab\_ov.cloner
====================

.. automodule:: isaaclab_ov.cloner

Clone contexts route assets to the representation that owns their copies:

* :class:`~isaaclab_ov.cloner.OvPhysxReplicateContext` prepares native physics replication.
* :class:`~isaaclab_ov.cloner.OvrtxReplicateContext` prepares copies for an OVRTX engine's internal scene.
* :class:`~isaaclab_ov.cloner.OvstageReplicateContext` prepares copies for simulation-owned OVStage resources,
  independently of their consumers. An OVRTX engine borrowing that stage does not clone it again.

OVPhysX and OVRTX retain their native cloning paths by default. With
``ISAAC_LAB_OVRTX_USE_OVSTAGE=1``, Isaac Lab creates one shared
:class:`~isaaclab_ov.stage.OvstageBackend` populated with physics and rendering domains.
``OvstageReplicateContext`` replaces both native contexts and clones their combined asset routes once.
Physics and rendering attach to that same stage; the stage owner coordinates write ordinals and closes
only after both borrowers. Automatic sharing is deferred until an OVPhysX release includes the cold
binding and path-lookup performance fixes and startup parity is validated; OVPhysX 0.6.3 is affected.

For shared stages, configure cameras before the initial simulation reset. Camera overrides are authored
before the stage is populated. OVRTX cameras may have different product settings, but share one native renderer
and must agree on native logging and transform-cache settings. Rendering on the shared stage is synchronous.
