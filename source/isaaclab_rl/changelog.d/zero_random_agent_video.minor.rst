Added
^^^^^

* Added ``--video``, ``--video_length``, and ``--video_interval`` CLI arguments to the zero and
  random checkpoint-free agents (:mod:`isaaclab_rl.entrypoints.simple_agents`), reusing the video
  recording infrastructure shared with the train and play entrypoints.
* Added :attr:`~isaaclab_rl.entrypoints.SimpleAgentRequest.video` to request video recording from the
  typed zero and random agent APIs.

Fixed
^^^^^

* Fixed ``--video`` playback stopping before every video recorder had finished its first clip. Playback
  now runs until the recorder whose first clip ends last is done, including its ``step_offset`` when
  ``--video_length`` is passed.
