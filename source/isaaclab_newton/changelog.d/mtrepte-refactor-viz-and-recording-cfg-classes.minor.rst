Changed
^^^^^^^

* **Breaking:** Removed ``isaaclab_newton.video_recording.recording_hooks`` (dead stub, never
  wired into the dispatch path). No migration needed.

* Added :meth:`~isaaclab_newton.physics.NewtonManager.video_capture_backend` classmethod
  (returns ``"newton_gl"``), used by
  :class:`~isaaclab.envs.utils.VideoRecorder` to select the Newton GL capture backend.
