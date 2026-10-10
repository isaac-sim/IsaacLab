* Added disabled automatic resets to ``ManagerBasedRLEnv``. Completed environments kept their final
  observations and stopped producing rewards, completion signals, and trajectory records until
  callers started another episode with ``reset()`` or ``reset_to()``. Physics continued simulating;
  same-step automatic resets remained the default.
* Added optional environment IDs to recorder manager step methods so callers could record only
  selected environments while recorder terms continued returning data for the full batch.
