* Moved physics body pose publication on OVStage scenes from :class:`~isaaclab_ov.renderers.OVRTXRenderer` into the
  stage-owned :class:`~isaaclab_ov.stage_publishers.BodyPosePublisher`. Renderers that share a stage write each
  physics state once, and environment visibility is stage state that every camera on the stage sees.
