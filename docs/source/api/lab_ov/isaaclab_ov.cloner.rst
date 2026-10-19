isaaclab\_ov.cloner
====================

.. automodule:: isaaclab_ov.cloner

Clone contexts route assets to the representation that owns their copies:

* :class:`~isaaclab_ov.cloner.OvPhysxReplicateContext` prepares native physics replication.
* :class:`~isaaclab_ov.cloner.OvrtxReplicateContext` prepares copies for an OVRTX engine's internal scene.

.. _ov-clone-routing:

Clone routing
-------------

``OvrtxReplicateContext.prepare`` in ``isaaclab_ov/cloner/replicate.py`` acquires each renderer's
native OVRTX engine before replication. OVPhysX keeps a private OVStage for physics population;
its clones use the OVPhysX API. OVRTX imports its own USD scene and applies its clones with the
OVRTX API. Newton also uses native OVRTX cloning.

Each renderer configuration owns one engine and its camera products. Physics and rendering routes
remain independent, including when both consume the same authored scene.
