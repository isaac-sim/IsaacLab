PhysX Backend
=============

`NVIDIA PhysX <https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/index.html>`_
is the historical default physics backend in Isaac Lab. It runs through
`NVIDIA Isaac Sim <https://docs.isaacsim.omniverse.nvidia.com>`_'s Omniverse Kit
runtime and supports GPU-accelerated rigid-body, articulation, soft-body, and
particle simulation. PhysX is the reference backend for behaviour parity in Isaac
Lab — assets, sensors, and tasks have all been validated against it first, and the
other backends are measured against PhysX behaviour.

PhysX is selected via :class:`~isaaclab_physx.physics.PhysxCfg`:

.. code-block:: python

    from isaaclab.sim import SimulationCfg
    from isaaclab_physx.physics import PhysxCfg

    sim_cfg = SimulationCfg(physics=PhysxCfg())

The PhysX backend uses the Temporal Gauss-Seidel (TGS) solver by default and also
exposes a Projective Gauss-Seidel (PGS) variant. Scene-level solver tuning, GPU
buffer sizing, and contact-handling parameters live on
:class:`~isaaclab_physx.physics.PhysxCfg`; per-actor settings remain on the USD
schema. See :doc:`configuration` for the most common knobs.

For an overview of how the multi-backend architecture works, see
:doc:`../../multi_backend_architecture`.


.. toctree::
  :maxdepth: 2
  :titlesonly:

  installation
  configuration
  supported-features
