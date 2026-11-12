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
before native resource initialization. Without environment overrides, OVRTX uses native cloning with
both OVPhysX and Newton. OVPhysX also retains native physics cloning.

``ISAAC_LAB_OVRTX_USE_OVSTAGE=1`` selects independent OVStage rendering with either physics backend;
``0`` selects native OVRTX cloning. Set the flag before creating the simulation. Reset retains the
selected route. This flag alone never enables physics/rendering stage sharing.

To share one OVStage with OVPhysX and OVRTX, set ``ISAAC_LAB_SHARE_OVSTAGE=1``. Sharing requires
OVPhysX and selects OVStage rendering regardless of ``ISAAC_LAB_OVRTX_USE_OVSTAGE``.

Sharing remains experimental and opt-in. It retains native physics cloning and populates complete
original worlds in one stage before OVPhysX attaches. ``OvstageReplicateContext`` also receives the
physics asset routes so it can complete the rendering copies after native physics cloning and binding
acquisition. The selected route is retained across resets and subsequent renderer creation.

Configure shared-stage cameras and contact sensors before the initial simulation reset. Cameras may have
different product settings, but share one native renderer and must agree on native logging and transform-cache settings.
Full-stage physics imports, including deformable scenes that require authored environment copies, are
not supported in shared mode. Disable ``ISAAC_LAB_SHARE_OVSTAGE``
for those scenes; sharing rejects them before native population rather than dropping their USD overrides.

The stage owner coordinates write ordinals and releases its borrowers before rebuilding. Rendering on
an OVStage is synchronous.

Shared-stage attachment order
-----------------------------

Shared mode uses early physics attachment without an additional flag:

#. Populate complete original worlds and required render sources in the shared OVStage, then seal ordinal 1.
#. Attach OVPhysX to ordinal 1 and wait for attachment to complete.
#. Execute native physics clones, wait for completion, and run physics warmup.
#. Initialize asset and contact bindings on the manager's
   :attr:`~isaaclab.physics.PhysicsManager.binding_init_event`, then acquire scene-data bindings.
#. Clone the remaining render prims into the shared OVStage, apply environment placement, and seal ordinal 2.
#. Publish ``PHYSICS_READY``, allowing startup gravity callbacks and camera preparation to use the completed stage.
#. Attach OVRTX to the shared stage during renderer preparation and continue normal rendering.

Sensors select their initialization phase through the read-only
:attr:`~isaaclab.sensors.SensorBase.initialize_event` property. Ordinary sensors use ``PHYSICS_READY``;
OVPhysX contact sensors use the manager's earlier binding event.

Later physics edits consume their exact update ordinals and exclude the render-copy ordinal 2.
A failed render copy requires a stage rebuild before retry. A forced reset releases bindings before
their native owners and rebuilds the attachment sequence.

Attaching OVPhysX before cloning the remaining render prims reduced initialization work and host memory
in the measured workloads.
