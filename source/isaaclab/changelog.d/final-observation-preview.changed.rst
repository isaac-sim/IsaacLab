* Custom stateful tensor processors should override ``preview`` to preserve their state during terminal captures;
  the default delegates to ``__call__`` for compatibility.
