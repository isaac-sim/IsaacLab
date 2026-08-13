Added
^^^^^

* Added :func:`~isaaclab.utils.assets.unmirror_file_path`, which maps a locally cached asset copy
  written by :func:`~isaaclab.utils.assets.retrieve_file_path` back to the URL it was downloaded
  from. Exports of a stage that references cached copies can use it to name the source assets
  instead of machine-specific cache paths.
