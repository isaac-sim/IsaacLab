* Moved the community Ray dispatch and tuning tools to ``isaaclab_contrib.rl.ray``,
  packaged their Kubernetes templates, and added the checkout's ``ray`` dependency extra.
  Run tools with ``uv run --extra ray python -m isaaclab_contrib.rl.ray.<tool>``.
  The previous scripts, tuning configuration paths, and cluster templates remained
  available for one deprecation release. Legacy job submissions continued uploading
  files beside the scripts; module users could select their files with ``--working_dir``.
* Disabled Ray's automatic uv checkout upload for the prepared cluster workflow and
  propagated failed worker commands and aggregate jobs to the caller instead of reporting success.
* Fixed manual resource wrapping to honor the requested CPU, GPU, and memory allocation,
  including zero-GPU jobs, without conflicting with the default worker count.
