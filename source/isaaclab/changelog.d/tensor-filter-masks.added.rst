* Added optional device-resident ``env_mask`` arguments to tensor processor calls. Digital filters and
  integrators preserve the history and output of unselected batch entries without gathering GPU indices.
