isaaclab\_ov.cloner
====================

.. automodule:: isaaclab_ov.cloner

Clone contexts route assets to the representation that owns their copies:

* :class:`~isaaclab_ov.cloner.OvPhysxReplicateContext` prepares native physics replication.
* :class:`~isaaclab_ov.cloner.OvrtxReplicateContext` prepares copies for an OVRTX engine's internal scene.
* :class:`~isaaclab_ov.cloner.OvstageReplicateContext` prepares copies for simulation-owned OVStage resources,
  independently of their consumers. An OVRTX engine borrowing that stage does not clone it again.

When OVPhysX and OVRTX are active together, Isaac Lab automatically creates one
:class:`~isaaclab_ov.stage.OvstageBackend` populated with physics and rendering domains.
``OvstageReplicateContext`` replaces both native contexts and clones their combined asset routes once.
Physics and rendering attach to that same stage; the stage owner coordinates write ordinals and closes
only after both borrowers. Independent OVPhysX and OVRTX scenes retain their native cloning paths.

Configure cameras before the initial simulation reset. Camera overrides are authored before the shared
stage is populated. OVRTX cameras may have different product settings, but share one native renderer
and must agree on native logging and transform-cache settings. Rendering on the shared stage is synchronous.
