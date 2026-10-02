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
selected route. This flag alone never enables physics/rendering stage sharing.

To share one OVStage with OVPhysX and OVRTX, set either:

* ``ISAAC_LAB_OVRTX_USE_OVSTAGE=1`` and ``ISAAC_LAB_OVPHYSX_USE_OVSTAGE=1``; or
* ``ISAAC_LAB_SHARE_OVSTAGE=1``, shorthand for enabling both.

Because OVStage rendering already defaults on with OVPhysX, setting only
``ISAAC_LAB_OVPHYSX_USE_OVSTAGE=1`` also selects sharing. Physics stage cloning requires OVPhysX and
OVStage rendering. Explicit ``0`` values disable their respective option; contradictory settings
raise an error instead of overriding one another. ``ISAAC_LAB_SHARE_OVSTAGE=0`` prohibits sharing
without changing the rendering default.

Sharing remains opt-in pending upstream cold-binding and shared-stage update performance fixes;
OVPhysX 0.6.3 has the measured startup regression. In shared mode, ``OvstageReplicateContext`` combines
both asset routes and clones one stage populated with physics and rendering domains before either
consumer attaches.

Configure shared-stage cameras before the initial simulation reset. Cameras may have different product
settings, but share one native renderer and must agree on native logging and transform-cache settings.

The rendering stage owner coordinates write ordinals and closes after its borrowers. Rendering on
an OVStage is synchronous.
