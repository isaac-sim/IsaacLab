* Fixed RLinf playback launched with ``uv run`` failing when Ray attempted to upload working directories larger
  than 512 MiB. The upload is now disabled for both RLinf training and playback.
