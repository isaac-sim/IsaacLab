* Used ordinary camera sensors for interactive RTX views, including global cameras in standalone scenes without a clone plan. Removed the separate perspective creation, capture, and resize paths; camera buffers retained their fixed resolution.
* Preserved runtime mesh and marker projection through the shared renderer and removed the unused viewer-only OVStage owner.
