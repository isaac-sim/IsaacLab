* Extended ``ObservationManager.compute`` with keyword-only ``preview=False``. Custom overrides must accept
  and forward this argument. Stateful tensor processors must override ``preview`` to preserve their state;
  its default delegates to ``__call__`` for compatibility.
