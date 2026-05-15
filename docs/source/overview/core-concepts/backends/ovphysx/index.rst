OvPhysX Backend
===============

.. warning::

    OvPhysX is **highly experimental** and is not recommended for general use yet.
    The public surface is changing rapidly while the backend is under active
    development. This page is a placeholder and will be expanded once the
    in-flight integration work lands on ``develop``.

OvPhysX is a kit-less variant of the PhysX backend. It drives PhysX directly
(without the Omniverse Kit runtime) and reads scene-level solver parameters
from the USD ``PhysicsScene`` prim rather than from a Python config. The Python
config :class:`~isaaclab_ovphysx.physics.OvPhysxCfg` only exposes the handful of
GPU buffer sizes that are not represented on the USD schema.

OvPhysX is selected through :class:`~isaaclab_ovphysx.physics.OvPhysxCfg`:

.. code-block:: python

    from isaaclab.sim import SimulationCfg
    from isaaclab_ovphysx.physics import OvPhysxCfg

    sim_cfg = SimulationCfg(physics=OvPhysxCfg())

Why use OvPhysX?
----------------

* **Kit-less execution.** OvPhysX avoids Omniverse Kit, which makes it a useful
  experimental path for headless deployments and for backends that don't need
  the Kit runtime stack.
* **USD-as-source-of-truth.** Solver parameters are taken from the
  ``PhysicsScene`` USD prim, so authoring tools that already manage USD scenes
  do not need a parallel Python config.

What works today
----------------

The asset and sensor surface tracks PhysX, but only a subset is implemented and
validated at the time of writing. Articulation, Rigid Object, and Contact
Sensor are the primary covered surfaces. Other sensors and renderer
combinations are partial — consult the in-flight pull requests below for the
current status:

* Articulation: `PR #5459 <https://github.com/isaac-sim/IsaacLab/pull/5459>`_
* RigidObject: `PR #5426 <https://github.com/isaac-sim/IsaacLab/pull/5426>`_

Status and follow-up
--------------------

This page is intentionally a stub. Once the in-flight OvPhysX work merges, this
section will be expanded with full installation, configuration, and supported
feature lists matching the other backends.

For architectural context, see :doc:`../../multi_backend_architecture`.
