Changed
^^^^^^^

* Extended the :attr:`~isaaclab_ov.renderers.OVRTXRendererCfg.use_ovrtx_cloning` path to support
  heterogeneous scenes as well as homogeneous ones. :meth:`~isaaclab_ov.renderers.OVRTXRenderer.prepare_stage`
  now exports only :class:`~isaaclab.cloner.ClonePlan` source prototypes plus global stage metadata, and
  replication uses OVRTX cloning API (``clone_usd``) for all rows in the published clone plan instead of
  cloning only ``/World/envs/env_0``.
