Fixed
^^^^^

* Fixed the "Prefilling reset buffer" progress bar in
  :func:`~isaaclab_tasks.core.lift.mdp.events` rendering as garbled block
  characters in GitHub Actions logs. The :class:`tqdm` bar is now disabled when
  ``stderr`` is not a TTY; a plain :mod:`logging` message is emitted at the
  start and on completion so CI logs still show meaningful progress.
