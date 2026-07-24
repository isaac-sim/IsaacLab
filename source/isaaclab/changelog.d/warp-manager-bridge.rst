Changed
^^^^^^^

* Relaxed ``func`` annotation on :class:`~isaaclab.managers.ObservationTermCfg`,
  :class:`~isaaclab.managers.RewardTermCfg`, and
  :class:`~isaaclab.managers.TerminationTermCfg` to ``Callable[..., torch.Tensor | None]``
  so kernel-style ``func(env, out) -> None`` terms type-check alongside the
  existing torch return-tensor form.
