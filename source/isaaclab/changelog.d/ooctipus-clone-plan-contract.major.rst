Changed
^^^^^^^

* **Breaking:** Made :class:`~isaaclab.cloner.ClonePlan` the complete input to clone backends:
  ``env_ids`` and ``positions`` are required, shared prims are exact zero-mask rows instead of
  the former ``ClonePlan.global_paths`` field, and ``context_rows`` routes plan rows to
  registered contexts. Replace ``UsdReplicateContext.queue(...)`` or ``queue_mapping(...)``
  followed by ``replicate()`` with ``UsdReplicateContext.replicate(plan)``, and represent the
  former constructor ``global_paths`` as exact shared plan rows. Standalone raw USD callers may
  continue to use :func:`~isaaclab.cloner.usd_replicate`. Remove the former ``stage`` argument
  from :func:`~isaaclab.cloner.replicate` and :class:`~isaaclab.cloner.ReplicateSession`, and
  pass explicit positions to :func:`~isaaclab.cloner.clone_plan_from_env_0`. Camera renderer
  integrations now consume exact ``CameraRenderSpec.camera_prim_paths``; remove the former
  ``camera_path_relative_to_env_0`` argument.
