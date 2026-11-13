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

.. raw:: html

   <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 880" role="img"
        aria-labelledby="shared-stage-flow-title shared-stage-flow-desc"
        style="display: block; width: 100%; max-width: 36rem; margin: 1.5rem auto;">
     <title id="shared-stage-flow-title">Shared OVStage attachment order</title>
     <desc id="shared-stage-flow-desc">
       Populate originals at ordinal 1, attach physics, complete native clones and warmup,
       acquire bindings, then clone render prims at ordinal 2 before publishing readiness
       and attaching OVRTX. The numbered steps below describe the same sequence.
     </desc>
     <defs>
       <marker id="shared-stage-flow-arrow" viewBox="0 0 10 10" refX="9" refY="5"
               markerWidth="7" markerHeight="7" orient="auto">
         <path d="M 0 0 L 10 5 L 0 10 Z" fill="currentColor"/>
       </marker>
     </defs>
     <g fill="none" stroke="currentColor" stroke-width="1.5">
       <rect x="10" y="10" width="580" height="70" rx="8"/>
       <rect x="10" y="110" width="580" height="70" rx="8"/>
       <rect x="10" y="210" width="580" height="70" rx="8"/>
       <rect x="10" y="310" width="580" height="70" rx="8"/>
       <rect x="10" y="410" width="580" height="70" rx="8"/>
       <rect x="10" y="510" width="580" height="70" rx="8"/>
       <rect x="10" y="610" width="580" height="70" rx="8"/>
       <rect x="10" y="710" width="580" height="70" rx="8"/>
       <rect x="10" y="810" width="580" height="60" rx="8"/>
       <g marker-end="url(#shared-stage-flow-arrow)">
         <path d="M 300 80 V 106"/>
         <path d="M 300 180 V 206"/>
         <path d="M 300 280 V 306"/>
         <path d="M 300 380 V 406"/>
         <path d="M 300 480 V 506"/>
         <path d="M 300 580 V 606"/>
         <path d="M 300 680 V 706"/>
         <path d="M 300 780 V 806"/>
       </g>
     </g>
     <g fill="currentColor" text-anchor="middle" font-family="sans-serif" font-size="18">
       <text x="300" y="39">Populate original worlds and required render sources</text>
       <text x="300" y="63" font-weight="bold">Seal ordinal 1</text>
       <text x="300" y="139">Attach OVPhysX at ordinal 1</text>
       <text x="300" y="163">Wait for completion</text>
       <text x="300" y="239">Execute native physics clones</text>
       <text x="300" y="263">Wait for completion</text>
       <text x="300" y="351">Run physics warmup</text>
       <text x="300" y="439">Initialize asset and contact bindings</text>
       <text x="300" y="463" font-family="monospace">binding_init_event</text>
       <text x="300" y="551">Acquire scene-data bindings</text>
       <text x="300" y="639">Clone remaining render prims and place environments</text>
       <text x="300" y="663" font-weight="bold">Seal ordinal 2 (rendering-only)</text>
       <text x="300" y="739" font-family="monospace">Publish PHYSICS_READY</text>
       <text x="300" y="763">Startup callbacks and camera preparation can proceed</text>
       <text x="300" y="835">Attach OVRTX during renderer preparation</text>
       <text x="300" y="859">Continue normal rendering</text>
     </g>
   </svg>

#. Populate complete original worlds and required render sources in the shared OVStage, then seal ordinal 1.
#. Attach OVPhysX to ordinal 1 and wait for attachment to complete.
#. Execute native physics clones, wait for completion, and run physics warmup.
#. Initialize asset and contact bindings on the manager's
   :attr:`~isaaclab.physics.PhysicsManager.binding_init_event`, then acquire scene-data bindings.
#. Clone the remaining render prims into the shared OVStage, apply environment placement, and seal ordinal 2.
#. Publish ``PHYSICS_READY``, allowing startup gravity callbacks and camera preparation to use the completed stage.
#. Attach OVRTX to the shared stage during renderer preparation and continue normal rendering.

CPU simulations populate all physics worlds before OVPhysX attachment and skip native physics cloning.
Deferred render copies still populate any missing render-only prims at ordinal 2.

Sensors select their initialization phase through the
:attr:`~isaaclab.sensors.SensorBase.initialize_event` attribute. Ordinary sensors use ``PHYSICS_READY``;
OVPhysX contact sensors use the manager's earlier binding event.

Later physics edits consume their exact update ordinals and exclude the render-copy ordinal 2.
A failed render copy requires a stage rebuild before retry. A forced reset releases bindings before
their native owners and rebuilds the attachment sequence.

Attaching OVPhysX before cloning the remaining render prims reduced initialization work and host memory
in the measured workloads.
